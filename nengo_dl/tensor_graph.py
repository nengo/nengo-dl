"""
Manages all the data and build processes associated with the TensorFlow graph.

The TensorFlow graph is the symbolic description of the computations in the
network, which will be executed by the simulator.
"""

from collections import OrderedDict, defaultdict
import functools
import inspect
import logging
import warnings

from nengo import Connection, Process, Ensemble
from nengo.builder.operator import SimPyFunc, Reset
from nengo.builder.processes import SimProcess
from nengo.config import ConfigError
from nengo.ensemble import Neurons
from nengo.exceptions import SimulationError, ValidationError
from nengo.neurons import Direct
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training.tracking import base as trackable

from nengo_dl import builder, graph_optimizer, signals, utils, tensor_node, config
from nengo_dl.compat import (
    tf_compat,
    SparseMatrix,
    is_sparse,
    make_process_state,
    make_process_step,
)

logger = logging.getLogger(__name__)


class TensorGraph(keras.layers.Layer):
    """
    Manages the construction of the TensorFlow symbolic computation graph.

    Parameters
    ----------
    model : `~nengo.builder.Model`
        Pre-built Nengo model describing the network to be simulated
    dt : float
        Length of a simulator timestep, in seconds
    unroll_simulation : int
        Unroll simulation loop by explicitly building ``unroll_simulation``
        iterations into the computation graph
    dtype : str
        Floating point precision to use for simulation (e.g. "float32")
    minibatch_size : int
        The number of simultaneous inputs that will be passed through the
        network
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        Device on which to execute computations (if None then uses the
        default device as determined by TensorFlow)
    progress : `.utils.ProgressBar`
        Progress bar for optimization stage
    rng : `~numpy.random.mtrand.RandomState`
        Seeded random number generator
    """

    @trackable.no_automatic_dependency_tracking
    def __init__(
        self, model, dt, unroll_simulation, dtype, minibatch_size, device, progress, rng
    ):
        super().__init__(
            name="TensorGraph",
            dynamic=False,
            trainable=not config.get_setting(model, "inference_only", False),
            dtype=dtype,
            batch_size=minibatch_size,
        )

        self.model = model
        self.dt = dt
        self.unroll = unroll_simulation
        self.minibatch_size = minibatch_size
        self.device = device
        self.rng = rng
        self.inference_only = not self.trainable
        self.signals = signals.SignalDict(
            dtype, self.minibatch_size, self.inference_only
        )

        # find invariant inputs (nodes that don't receive any input other
        # than the simulation time). we'll compute these outside the simulation
        # and feed in the result.
        if self.model.toplevel is None:
            self.invariant_inputs = OrderedDict()
        else:
            self.invariant_inputs = OrderedDict(
                (n, n.output)
                for n in self.model.toplevel.all_nodes
                if n.size_in == 0 and not isinstance(n, tensor_node.TensorNode)
            )

        # remove input nodes because they are executed outside the simulation
        node_processes = [
            n.output for n in self.invariant_inputs if isinstance(n.output, Process)
        ]
        operators = [
            op
            for op in self.model.operators
            if not (
                (isinstance(op, SimPyFunc) and op.x is None)
                or (
                    isinstance(op, SimProcess)
                    and op.input is None
                    and op.process in node_processes
                )
            )
        ]

        # mark trainable signals
        self.mark_signals()

        logger.info("Initial plan length: %d", len(operators))

        # apply graph simplification functions
        simplifications = config.get_setting(
            model,
            "simplifications",
            [
                graph_optimizer.remove_constant_copies,
                graph_optimizer.remove_unmodified_resets,
                graph_optimizer.remove_zero_incs,
                graph_optimizer.remove_identity_muls,
            ],
        )

        with progress.sub("operator simplificaton", max_value=None):
            old_operators = []
            while len(old_operators) != len(operators) or any(
                x is not y for x, y in zip(operators, old_operators)
            ):
                old_operators = operators
                for simp in simplifications:
                    operators = simp(operators)

        # group mergeable operators
        planner = config.get_setting(model, "planner", graph_optimizer.tree_planner)

        with progress.sub("merging operators", max_value=None):
            plan = planner(operators)

        # TODO: we could also merge operators sequentially (e.g., combine
        # a copy and dotinc into one op), as long as the intermediate signal
        # is only written to by one op and read by one op

        # order signals/operators to promote contiguous reads
        sorter = config.get_setting(model, "sorter", graph_optimizer.order_signals)

        with progress.sub("ordering signals", max_value=None):
            sigs, self.plan = sorter(plan, n_passes=10)

        # create base arrays and map Signals to TensorSignals (views on those
        # base arrays)
        with progress.sub("creating signals", max_value=None):
            self.create_signals(sigs)

        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info(
            "Number of base arrays: %d, %d",
            *tuple(len(x) for x in self.base_arrays_init),
        )

    # @with_self
    def build_inputs(self):
        """
        Generates a set of Input layers that can be used as inputs to a
        TensorGraph layer.

        Returns
        -------
        n_steps : ``tf.keras.layers.Input``
            Input layer for specifying the number of simulation timesteps.
        inputs : dict of {`nengo:nengo.Node`: ``tf.keras.layers.Input``}
            Input layers for each of the Nodes in the network.
        """

        # number of steps to run
        n_steps = keras.layers.Input(
            shape=(1,), batch_size=self.minibatch_size, dtype="int32", name="n_steps"
        )

        # input placeholders
        inputs = OrderedDict(
            (
                n,
                keras.layers.Input(
                    shape=(None, n.size_out),
                    batch_size=self.minibatch_size,
                    dtype=self.dtype,
                    name="node_%d" % i
                    if n.label is None
                    else utils.sanitize_name(n.label),
                ),
            )
            for i, n in enumerate(self.invariant_inputs)
        )

        return n_steps, inputs

    # @with_self
    def build(self, input_shape=None):
        """
        Create any Variables used in the model.
        """

        super().build(input_shape)

        # variables for model parameters
        assert len(self.signals.base_params) == 0
        for k, v in self.base_arrays_init[True].items():
            self.signals.base_params[k] = self.add_weight(
                initializer=tf.initializers.constant(v),
                shape=v.shape,
                dtype=v.dtype,
                trainable=True,
                name="base_params/%s_%s" % (v.dtype, "_".join(str(x) for x in v.shape)),
            )

        logger.debug("created base param variables")
        logger.debug([str(x) for x in self.signals.base_params.values()])

        # variables to save the internal state of simulation between runs
        # note: place these on CPU because they'll only be accessed once at the
        # beginning of the simulation loop, and they can be quite large
        with tf.device("/cpu:0"):
            for k, v in self.base_arrays_init[False].items():
                self.signals.saved_state[k] = self.add_weight(
                    # TODO: don't make these a constant initializer because we
                    #  don't want to store those large arrays in the graph def
                    #  (double check that this is still a problem in TF2.0)
                    initializer=tf.initializers.constant(v),
                    shape=v.shape,
                    dtype=v.dtype,
                    trainable=False,
                    name="saved_state/%s_%s"
                    % (v.dtype, "_".join(str(x) for x in v.shape)),
                )

        logger.debug("created saved state variables")
        logger.debug([str(x) for x in self.signals.saved_state.values()])

    # @with_self
    # @tf.function  # TODO: get this working? does this help?
    @tf.autograph.experimental.do_not_convert  # TODO: enable autograph
    @trackable.no_automatic_dependency_tracking
    def call(self, inputs, training=False, progress=None):
        """
        Constructs a new graph to simulate the model.

        Parameters
        ----------
        inputs : list of ``tf.Tensor``
            Input placeholders for the network (must match the order defined in
            `.build_inputs`).
        training : bool
            Whether the network is being run in training or inference mode.
        progress : `.utils.ProgressBar`
            Progress bar for construction stage

        Returns
        -------
        probe_arrays : list of ``tf.Tensor``
            Tensors representing the output of all the Probes in the network (order
            corresponding to ``self.model.probes``, which is the order the Probes were
            instantiated).
        """

        super().call(inputs, training=training)

        if progress is None:
            progress = utils.NullProgressBar()

        # reset signaldict
        self.signals.reset()

        self.outputs = {}
        self.optimizers = {}

        # create these constants once here for reuse in different operators
        self.signals.training = training
        self.signals.dt = tf.constant(self.dt, self.dtype)
        self.signals.dt_val = self.dt  # store the actual value as well
        self.signals.zero = tf.constant(0, self.dtype)
        self.signals.one = tf.constant(1, self.dtype)

        # variable to track training step
        # if not self.inference_only:
        #     self.training_step = tf_compat.train.get_or_create_global_step()
        # else:
        #     self.training_step = None

        input_idx = 0
        self.steps_to_run = inputs[input_idx][0, 0]
        input_idx += 1

        # set up invariant inputs
        self.input_phs = {}
        for n in self.invariant_inputs:
            # specify batch dimension (keras sets it to None)
            inputs[input_idx].set_shape(
                [self.minibatch_size] + inputs[input_idx].get_shape().as_list()[1:]
            )

            self.input_phs[n] = inputs[input_idx]
            input_idx += 1

        # set up target placeholders
        # self.target_phs = {}
        # for p in self.model.probes:
        #     self.target_phs[p] = inputs[input_idx]
        #     input_idx += 1

        assert input_idx == len(inputs)

        # initialize op builder
        build_config = builder.BuildConfig(
            inference_only=self.inference_only,
            lif_smoothing=config.get_setting(self.model, "lif_smoothing"),
            cpu_only=self.device == "/cpu:0" or not utils.tf_gpu_installed,
            rng=self.rng,
            add_weight=self.add_weight,
        )
        self.op_builder = builder.Builder(self.plan, self.signals, build_config)

        # pre-build stage
        with progress.sub("pre-build stage", max_value=len(self.plan)) as sub:
            self.op_builder.build_pre(sub)

        # build stage
        with progress.sub("build stage", max_value=len(self.plan) * self.unroll) as sub:
            self._build_loop(sub)

        # update saved internal state
        updated_states = [
            var.assign(val)
            for var, val in zip(
                self.signals.saved_state.values(), self.final_internal_state
            )
        ]
        with tf.control_dependencies(updated_states):
            self.steps_run_and_save = tf.identity(self.steps_run)

        # logging
        logger.info(
            "Number of reads: %d", sum(x for x in self.signals.read_types.values())
        )
        for x in self.signals.read_types.items():
            logger.info("    %s: %d", *x)
        logger.info(
            "Number of writes: %d", sum(x for x in self.signals.write_types.values())
        )
        for x in self.signals.write_types.items():
            logger.info("    %s: %d", *x)

        # note: always return steps_run so that the simulation will run for the given
        # number of steps, even if there are no output probes
        output = [self.steps_run]
        output.extend(self.probe_arrays.values())
        return output

    def _build_step(self, progress):
        """
        Build the operators that execute a single simulation timestep
        into the graph.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for loop construction

        Returns
        -------
        probe_tensors : list of ``tf.Tensor``
            The Tensor objects representing the data required for each model
            Probe
        side_effects : list of ``tf.Tensor``
            The output Tensors of computations that may have side-effects
            (e.g., `~nengo.Node` functions), meaning that they
            must be executed each time step even if their output doesn't appear
            to be used in the simulation
        """

        # build operators
        side_effects = self.op_builder.build(progress)

        logger.debug("collecting probe tensors")
        probe_tensors = []
        for p in self.model.probes:
            probe_sig = self.model.sig[p]["in"]
            if probe_sig in self.signals:
                # TODO: better solution to avoid the forced_copy
                # we need to make sure that probe reads occur before the
                # probe value is overwritten on the next timestep. however,
                # just blocking on the sliced value (probe_tensor) doesn't
                # work, because slices of variables don't perform a
                # copy, so the slice can be "executed" and then the value
                # overwritten before the tensorarray write occurs. what we
                # really want to do is block until the probe_arrays.write
                # happens, but you can't block on probe_arrays (and blocking on
                # probe_array.flow doesn't work, although I think it should).
                # so by adding the copy here and then blocking on the copy, we
                # make sure that the probe value is read before it can be
                # overwritten.
                probe_tensors.append(
                    self.signals.gather(self.signals[probe_sig], force_copy=True)
                )
            else:
                # if a probe signal isn't in sig_map, that means that it isn't
                # involved in any simulator ops.  so we know its value never
                # changes, and we'll just return a constant containing the
                # initial value.
                if probe_sig.minibatched:
                    init_val = np.tile(
                        probe_sig.initial_value[None, :], (self.minibatch_size, 1)
                    )
                else:
                    init_val = probe_sig.initial_value
                probe_tensors.append(tf.constant(init_val, dtype=self.dtype))

        logger.debug("=" * 30)
        logger.debug("build_step complete")
        logger.debug("probe_tensors %s", [str(x) for x in probe_tensors])
        logger.debug("side_effects %s", [str(x) for x in side_effects])

        return probe_tensors, side_effects

    def _build_loop(self, progress):
        """
        Build simulation loop.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for loop construction
        """

        def loop_condition(loop_i, n_steps, *_):
            return loop_i < n_steps

        def loop_body(loop_i, n_steps, probe_arrays, base_tensors):
            # fill in signals.bases (note: we need to do this here because we
            # need to use the tensors from inside the
            # loop, not the variables in signals.saved_state)
            # TODO: test that rebuilding multiple times works properly
            assert len(self.signals.bases) == 0
            for i, key in enumerate(self.signals.saved_state):
                self.signals.bases[key] = base_tensors[i]
            # for the parameter variables we can just use the base variable
            # (since we'll only be reading inside the loop, not updating
            # the variables)
            for key in self.signals.base_params:
                self.signals.bases[key] = self.signals.base_params[key]

            for iter in range(self.unroll):
                logger.debug("BUILDING ITERATION %d", iter)
                with tf.name_scope("iteration_%d" % iter):
                    # fill in invariant input data
                    for n in self.input_phs:
                        if self.model.sig[n]["out"] in self.signals:
                            # if the out signal doesn't exist then that means that
                            # the node output isn't actually used anywhere, so we can
                            # ignore it

                            self.signals.scatter(
                                self.signals[self.model.sig[n]["out"]],
                                self.input_phs[n][:, loop_i],
                            )

                    # build the operators for a single step
                    # note: we tie things to the `loop_i` variable so that we
                    # can be sure the other things we're tying to the
                    # simulation step (side effects and probes) from the
                    # previous timestep are executed before the next step
                    # starts
                    with tf.control_dependencies([loop_i]):
                        probe_tensors, side_effects = self._build_step(progress)

                    # copy probe data to array
                    for i, p in enumerate(probe_tensors):
                        if config.get_setting(
                            self.model,
                            "keep_history",
                            default=True,
                            obj=self.model.probes[i],
                        ):
                            probe_arrays[i] = probe_arrays[i].write(loop_i, p)
                        else:
                            probe_arrays[i] = tf.cond(
                                pred=tf.equal(loop_i + 1, n_steps),
                                true_fn=lambda p=p: probe_arrays[i].write(0, p),
                                false_fn=lambda: probe_arrays[i],
                            )

                    # need to make sure that any operators that could have side
                    # effects run each timestep, so we tie them to the loop
                    # increment. we also need to make sure that all the probe
                    # reads happen before those values get overwritten on the
                    # next timestep
                    with tf.control_dependencies(side_effects + probe_tensors):
                        loop_i += 1

            state_arrays = tuple(
                self.signals.bases[key] for key in self.signals.saved_state
            )

            return loop_i, n_steps, probe_arrays, state_arrays

        loop_i = tf.constant(0)

        probe_arrays = [
            tf.TensorArray(self.dtype, clear_after_read=True, size=0, dynamic_size=True)
            for _ in self.model.probes
        ]

        # build simulation loop
        loop_vars = (
            loop_i,
            self.steps_to_run,
            probe_arrays,
            tuple(self.signals.saved_state.values()),
        )

        # TODO: try out parallel_iterations again with tensors?
        loop_vars = tf.while_loop(
            cond=loop_condition,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,
            back_prop=not self.inference_only,
        )

        # change to shape (minibatch_size,) (required by keras) instead of a scalar
        self.steps_run = tf.tile(
            tf.expand_dims(loop_vars[0], 0), (self.minibatch_size,)
        )

        self.probe_arrays = OrderedDict()
        for p, a in zip(self.model.probes, loop_vars[2]):
            x = a.stack()

            if self.model.sig[p]["in"].minibatched:
                # change from tensorarray's (steps, batch, d) to (batch, steps, d)
                perm = np.arange(x.shape.ndims)
                perm[[0, 1]] = perm[[1, 0]]
                x = tf.transpose(a=x, perm=perm)
            else:
                # add minibatch dimension for consistency
                x = tf.expand_dims(x, 0)

            self.probe_arrays[p] = x

        self.final_internal_state = loop_vars[3]

    def build_optimizer_func(self, optimizer, loss, direct_grads=None):
        """
        Adds elements into the graph to execute the given optimizer.

        Parameters
        ----------
        optimizer : ``tf.train.Optimizer``
            Instance of a TensorFlow optimizer class
        objective : dict of {`~nengo.Probe`: callable or ``None``}
            The objective to be minimized. This is a dictionary mapping Probes
            to functions
            ``f(output, target) -> loss`` that consume the actual output and
            target output for the given probe(s) and return a ``tf.Tensor``
            representing a scalar loss value.  The function may also accept a
            single argument ``f(output) -> loss`` if targets are not required.
            Some common objective functions can be found in
            `nengo_dl.objectives`.

            Passing ``None`` as the probe value (instead of a callable)
            indicates that the error is being computed outside the simulation,
            and the value passed for that probe in ``data`` directly specifies
            the output error gradient.

            If multiple probes are specified as the key, then the corresponding
            output/target values will be passed as a list to the objective
            function.

            The overall loss value being minimized will be the sum across all
            the objectives specified.

        Returns
        -------
        apply_optimizer : callable
            A function that builds the operators required to implement the
            given optimizer update.  Generally this function will then be
            passed to `~.build_outputs`.

        Notes
        -----
        This function caches its outputs, so if it is called again with the
        same arguments then it will return the previous function.  This avoids
        building duplicates of the same operations over and over.  This can
        also be important functionally, e.g. if the optimizer has internal
        state like momentum.  By caching the output we ensure that subsequent
        calls share the same internal state.
        """

        if direct_grads is None:
            direct_grads = []

        key = (optimizer, frozenset(loss.items()), frozenset(direct_grads))

        try:
            # return the cached optimizer function if it exists
            return self.optimizers[key]
        except KeyError:
            pass

        # note: the standard workflow is that sim.train calls
        # build_optimizer_func to get this function. it then passes the
        # function to run_batch, which calls build_outputs to actually
        # build these operations into the graph. we do this somewhat
        # indirect method so that everything passes through build_output,
        # allowing us to consolidate certain logic there (like capturing
        # new variables)
        def apply_optimizer(outputs, targets):
            # note: we don't actually use outputs/targets, because the loss
            # has already been computed outside this function
            nonlocal loss

            agg_method = tf.AggregationMethod.DEFAULT
            grads = []
            vars = [v for v in self.signals.all_variables if v.trainable]

            # compute gradients wrt loss
            if len(loss) > 0:
                # reduce loss to a scalar
                loss = tf.reduce_sum(
                    input_tensor=[tf.reduce_sum(input_tensor=v) for v in loss.values()]
                )

                grads.append(
                    tf.gradients(ys=loss, xs=vars, aggregation_method=agg_method)
                )

            # add in any gradients where the user directly specified the output
            # error grad
            for p in direct_grads:
                grads.append(
                    tf.gradients(
                        ys=self.probe_arrays[p],
                        xs=vars,
                        grad_ys=self.target_phs[p],
                        aggregation_method=agg_method,
                    )
                )

            # combine gradients for each variable
            if len(grads) == 1:
                grads = grads[0]
            else:
                grads = [tf.reduce_sum(input_tensor=gs, axis=0) for gs in zip(*grads)]

            opt_op = optimizer.apply_gradients(zip(grads, vars))

            with tf.control_dependencies([opt_op]):
                new_step = tf_compat.assign_add(
                    self.training_step, tf.constant(1, dtype=tf.int64)
                )

            return new_step, loss

        self.optimizers[key] = apply_optimizer

        return self.optimizers[key]

    # @with_self
    def build_outputs(self, outputs):
        """
        Adds elements into the graph to compute the given outputs.

        Parameters
        ----------
        outputs : dict of {(tuple of) `~nengo.Probe`: callable or None}
            The output function to be applied to each probe or group of probes.
            The function can accept one argument (the output of that probe) or
            two (output and target values for that probe).  If a tuple of
            Probes are given as the key, then those output/target parameters
            will be the corresponding tuple of probe/target values.  The
            function should return a ``tf.Tensor`` or tuple of Tensors
            representing the output we want from those probes.  If ``None`` is
            given instead of a function then the output will simply be the
            output value from the corresponding probes.

        Returns
        -------
        output_vals : dict of {(tuple of) `~nengo.Probe`: \
                               (tuple of) ``tf.Tensor``}
            Tensors representing the result of applying the output functions
            to the probes.
        new_vars_init : ``tf.Tensor`` or None
            Initialization op for any new variables created when building
            the outputs.

        Notes
        -----
        This function caches its outputs, so if it is called again with the
        same arguments then it will return the previous Tensors.  This avoids
        building duplicates of the same operations over and over.  This can
        also be important functionally, e.g. if the outputs have internal
        state.  By caching the output we ensure that subsequent
        calls share the same internal state.
        """

        key = frozenset(outputs.items())

        try:
            # return the cached outputs if they exist
            return self.outputs[key], None
        except KeyError:
            pass

        output_vals = {}
        new_vars = []
        # certain output functions may not return variables in a pre_build
        # function, but do add them to the global_variables collection
        # (e.g., tf.train.Optimizers). so we can compare the items
        # in that collection before and after building the function to
        # try to capture those variables as well.
        # TODO: remove this if we switch completely to keras optimizers
        pre_vars = set(
            tf_compat.get_default_graph().get_collection(
                tf_compat.GraphKeys.GLOBAL_VARIABLES
            )
        )
        for probes, out in outputs.items():
            is_tuple = isinstance(probes, tuple)
            probe_arrays = (
                tuple(self.probe_arrays[p] for p in probes)
                if is_tuple
                else self.probe_arrays[probes]
            )

            if out is None:
                # return probe output value
                output_vals[probes] = probe_arrays
            elif callable(out):
                # look up number of arguments for function
                spec = inspect.getfullargspec(out)
                nargs = len(spec.args)

                # don't count keyword arguments
                if spec.defaults is not None:
                    nargs -= len(spec.defaults)

                # don't count self argument for methods or callable classes
                out_func = out.func if isinstance(out, functools.partial) else out
                if inspect.ismethod(out_func) or not inspect.isroutine(out_func):
                    nargs -= 1

                # build function arguments
                if nargs == 1:
                    args = [probe_arrays]
                elif nargs == 2:
                    target_phs = (
                        tuple(self.target_phs[p] for p in probes)
                        if is_tuple
                        else self.target_phs[probes]
                    )
                    args = [probe_arrays, target_phs]
                else:
                    raise ValidationError(
                        "Output functions must accept 1 or 2 arguments; '%s' "
                        "takes %s arguments"
                        % (utils.function_name(out, sanitize=False), nargs),
                        "outputs",
                    )

                # call output pre_build function (if any)
                if hasattr(out, "pre_build"):
                    vars = out.pre_build(
                        *[
                            (
                                [x.shape.as_list() for x in arg]
                                if is_tuple
                                else arg.shape.as_list()
                            )
                            for arg in args
                        ]
                    )
                    if isinstance(vars, (list, tuple)):
                        new_vars.extend(vars)
                    elif vars is not None:
                        new_vars.append(vars)

                # apply output function
                with tf_compat.name_scope(utils.function_name(out)):
                    output_vals[probes] = out(*args)

            else:
                raise ValidationError("Outputs must be callable or None)", "outputs")

        # collect any new variables created during build process
        self.signals.user_vars.extend(new_vars)
        new_vars.extend(
            set(
                tf_compat.get_default_graph().get_collection(
                    tf_compat.GraphKeys.GLOBAL_VARIABLES
                )
            )
            - pre_vars
        )
        new_vars_init = (
            tf_compat.variables_initializer(new_vars) if len(new_vars) > 0 else None
        )

        self.outputs[key] = output_vals
        return output_vals, new_vars_init

    # @with_self
    def build_post(self):
        """
        Executes post-build processes for operators (after the graph has
        been constructed and whenever Simulator is reset).
        """

        # build input functions (we need to do this here, because in the case
        # of processes these functions need to be be rebuilt on reset)
        self.input_funcs = {}
        for n, output in self.invariant_inputs.items():
            if isinstance(output, np.ndarray):
                self.input_funcs[n] = output
            elif isinstance(output, Process):
                state = make_process_state(output, (n.size_in,), (n.size_out,), self.dt)
                self.input_funcs[n] = [
                    make_process_step(
                        output,
                        (n.size_in,),
                        (n.size_out,),
                        self.dt,
                        output.get_rng(self.rng),
                        state,
                    )
                    for _ in range(self.minibatch_size)
                ]
            elif n.size_out > 0:
                self.input_funcs[n] = [
                    utils.align_func((n.size_out,), self.dtype)(output)
                ]
            else:
                # a node with no inputs and no outputs, but it can still
                # have side effects
                self.input_funcs[n] = [output]

        # execute build_post on all the op builders
        self.op_builder.build_post()

    # @with_self
    def build_summaries(self, summaries):
        """
        Adds ops to collect summary data for the given objects.

        Parameters
        ----------
        summaries : list of dict or \
                            `~nengo.Connection` or \
                            `~nengo.Ensemble` or \
                            `~nengo.ensemble.Neurons` or \
                            ``tf.Tensor``}
            List of objects for which we want to collect data.  Object can be a
            Connection (in which case data on weights will be collected),
            Ensemble (encoders), Neurons (biases), a dict of
            ``{probe: objective}`` that indicates a loss function that will
            be tracked, or a pre-built summary tensor.

        Returns
        -------
        op : ``tf.Tensor``
            Merged summary op for the given summaries
        """

        summary_ops = []
        inits = []
        with tf.device("/cpu:0"):
            for obj in summaries:
                if isinstance(obj, dict):
                    # overall loss
                    loss, init = self.build_outputs(obj)
                    if init is not None:
                        inits.append(init)
                    summary_ops.append(
                        tf_compat.summary.scalar(
                            "loss",
                            tf.reduce_sum(
                                input_tensor=[
                                    tf.reduce_sum(input_tensor=v) for v in loss.values()
                                ]
                            ),
                            family="loss",
                        )
                    )

                    if len(obj) > 1:
                        # get loss for each probe
                        for p, t in loss.items():
                            summary_ops.append(
                                tf_compat.summary.scalar(
                                    utils.sanitize_name("Probe_%s_loss" % p.label),
                                    tf.reduce_sum(input_tensor=t),
                                    family="loss",
                                )
                            )
                elif isinstance(obj, (Ensemble, Neurons, Connection)):
                    if isinstance(obj, Ensemble):
                        param = "encoders"
                        name = "Ensemble_%s" % obj.label
                    elif isinstance(obj, Neurons):
                        param = "bias"
                        name = "Ensemble.neurons_%s" % obj.ensemble.label
                    elif isinstance(obj, Connection):
                        param = "weights"
                        name = "Connection_%s" % obj.label

                    summary_ops.append(
                        tf_compat.summary.histogram(
                            utils.sanitize_name("%s_%s" % (name, param)),
                            self.get_tensor(self.model.sig[obj][param]),
                        )
                    )
                elif isinstance(obj, tf.Tensor):
                    # we assume that obj is a summary op
                    summary_ops.append(obj)
                else:
                    raise SimulationError("Unknown summary object: %s" % obj)

            return (
                tf_compat.summary.merge(summary_ops),
                (None if len(inits) == 0 else inits),
            )

    # def compute_output_signature(self, input_signature):
    #     inputs = self.build_inputs()
    #     for spec, ph in zip(input_signature, inputs):
    #         assert spec.shape == ph.shape
    #         assert spec.dtype == ph.dtype
    #
    #     return [tf.TensorSpec(p.shape, dtype=p.dtype) for p in self.probe_arrays]

    # @with_self
    def get_tensor(self, sig):
        """
        Returns a Tensor corresponding to the given Signal.

        Parameters
        ----------
        sig : `~nengo.builder.Signal`
            A signal in the model

        Returns
        -------
        tensor : ``tf.Tensor``
            Tensor containing the value of the given Signal
        """

        tensor_sig = self.signals[sig]

        try:
            base = self.signals.base_params[tensor_sig.key]
        except KeyError:
            base = self.signals.saved_state[tensor_sig.key]

        if "while/" in tensor_sig.tf_indices.name:
            # rebuild tf indices outside the while loop
            tensor_sig._tf_indices = None

        return tf.gather(
            base, tensor_sig.tf_indices, axis=1 if tensor_sig.minibatched else 0
        )

    def mark_signals(self):
        """
        Mark all the signals in ``self.model`` according to whether they
        represent trainable parameters of the model (parameters that can be
        optimized by deep learning methods).

        Trainable parameters include connection weights, ensemble encoders, and
        neuron biases.  Unless one of those signals is targeted by a Nengo
        learning rule (otherwise the learning rule update conflicts with the
        deep learning optimization).

        Users can manually specify whether signals are trainable or not using
        the config system (e.g.,
        ``net.config[nengo.Ensemble].trainable = False``)
        """

        def get_trainable(net_config, obj, network_trainable):
            """Looks up the current value of ``obj.trainable``."""

            if self.inference_only:
                return False

            try:
                if obj in net_config.params:
                    # priority #1: instance config
                    trainable = net_config[obj].trainable
                elif network_trainable is not 1:  # noqa: F632
                    # priority #2: network setting
                    trainable = network_trainable
                else:
                    # priority #3: class config
                    trainable = net_config[obj].trainable
            except (ConfigError, AttributeError):
                trainable = network_trainable

            # we return 1 if trainable isn't configured, since the default is
            # for everything to be trainable but we want to be able to
            # distinguish whether something was specifically set to be
            # trainable (True) or just defaulting to trainable (1)
            return 1 if trainable is None else trainable

        def mark_network(net_config, net, network_trainable):
            """Recursively marks the signals for objects within each
            subnetwork."""

            for subnet in net.networks:
                mark_network(
                    net_config,
                    subnet,
                    get_trainable(net_config, subnet, network_trainable),
                )

            # encoders and biases are trainable
            for ens in net.ensembles:
                ens_trainable = get_trainable(net_config, ens, network_trainable)

                self.model.sig[ens]["encoders"].trainable = ens_trainable
                self.model.sig[ens]["encoders"].minibatched = False

                if not isinstance(ens.neuron_type, Direct):
                    neurons_trainable = get_trainable(
                        net_config, ens.neurons, network_trainable
                    )
                    if neurons_trainable is 1:  # noqa: F632
                        neurons_trainable = ens_trainable

                    self.model.sig[ens.neurons]["bias"].trainable = neurons_trainable
                    self.model.sig[ens.neurons]["bias"].minibatched = False

            # connection weights are trainable
            for conn in net.connections:
                # note: this doesn't include probe connections, since they
                # aren't added to the network
                self.model.sig[conn]["weights"].trainable = get_trainable(
                    net_config, conn, network_trainable
                )
                self.model.sig[conn]["weights"].minibatched = False

            # parameters can't be modified by an online Nengo learning rule
            # and offline training at the same time. (it is possible in
            # theory, but it complicates things a lot and is probably not a
            # common use case). we also make those signals minibatched
            # (they wouldn't be normally), because we want to be able to
            # learn independently in each minibatch
            for conn in net.connections:
                rule = conn.learning_rule
                if rule is not None:
                    if isinstance(rule, dict):
                        rule = list(rule.values())
                    elif not isinstance(rule, list):
                        rule = [rule]

                    for r in rule:
                        if r.modifies in ("weights", "decoders"):
                            obj = conn
                            attr = "weights"
                        elif r.modifies == "encoders":
                            obj = conn.post_obj
                            attr = "encoders"
                        else:
                            raise NotImplementedError

                        if self.model.sig[obj][attr].trainable is True:
                            warnings.warn(
                                "%s has a learning rule and is also set "
                                "to be trainable; this is likely to "
                                "produce strange training behaviour." % obj
                            )
                        else:
                            self.model.sig[obj][attr].trainable = False

                        self.model.sig[obj][attr].minibatched = True

        if self.model.toplevel is None:
            warnings.warn(
                "No top-level network in model; assuming no trainable parameters",
                UserWarning,
            )
        else:
            net_config = self.model.toplevel.config
            mark_network(
                net_config,
                self.model.toplevel,
                get_trainable(net_config, self.model.toplevel, 1),
            )

            # the connections to connection probes are not trainable, but
            # also not minibatched
            probe_seeds = [self.model.seeds[p] for p in self.model.probes]
            for obj, seed in self.model.seeds.items():
                if isinstance(obj, Connection) and seed in probe_seeds:
                    self.model.sig[obj]["weights"].trainable = False
                    self.model.sig[obj]["weights"].minibatched = False

        # time/step are not minibatched and not trainable
        self.model.step.trainable = False
        self.model.step.minibatched = False
        self.model.time.trainable = False
        self.model.time.minibatched = False

        # fill in defaults for all other signals
        # signals are not trainable by default, and views take on the
        # properties of their bases
        for op in self.model.operators:
            for sig in op.all_signals:
                if not hasattr(sig.base, "trainable"):
                    sig.base.trainable = False

                if not hasattr(sig.base, "minibatched"):
                    sig.base.minibatched = not sig.base.trainable

                if not hasattr(sig, "trainable"):
                    sig.trainable = sig.base.trainable

                if not hasattr(sig, "minibatched"):
                    sig.minibatched = sig.base.minibatched

    @trackable.no_automatic_dependency_tracking
    def create_signals(self, sigs):
        """
        Groups signal data together into larger arrays, and represent each
        individual signal as a slice into that array.

        Parameters
        ----------
        sigs : list of `~nengo.builder.Signal`
            Base signals arranged into the order in which they should reside in
            memory (e.g., output from `.graph_optimizer.order_signals`)
        """

        float_type = np.dtype(self.dtype)
        base_arrays = [OrderedDict(), OrderedDict()]
        curr_keys = {}
        sig_idxs = {s: i for i, s in enumerate(sigs)}

        # find the non-overlapping partitions of the signals
        breaks = []
        diff = defaultdict(int)
        for ops in self.plan:
            # note: we don't include Resets, otherwise the big reset block
            # overrides most of the partitioning
            if not isinstance(ops[0], Reset):
                for i in range(len(ops[0].all_signals)):
                    op_sigs = [op.all_signals[i].base for op in ops]
                    idxs = [sig_idxs[s] for s in op_sigs]
                    diff[op_sigs[np.argmin(idxs)]] += 1
                    diff[op_sigs[np.argmax(idxs)]] -= 1

        # find the partition points in signal list
        open = 0
        for i, s in enumerate(sigs):
            if s in diff:
                open += diff[s]

            if open == 0:
                breaks += [i + 1]

        logging.debug("partitions")
        logging.debug(
            "\n%s", "".join("|" if i in breaks else " " for i in range(len(sigs)))
        )

        # create all the base signals
        for i, sig in enumerate(sigs):
            assert sig not in self.signals
            assert not sig.is_view

            if i in breaks:
                # start a new array for all current bases
                for k in curr_keys:
                    curr_keys[k] = object()

            # convert to appropriate dtype
            if np.issubdtype(sig.dtype, np.floating):
                dtype = float_type
            elif np.issubdtype(sig.dtype, np.integer):
                dtype = np.int32
            elif np.issubdtype(sig.dtype, np.bool_):
                dtype = sig.dtype
            else:
                raise NotImplementedError("Unsupported signal dtype")

            if is_sparse(sig):
                # for sparse tensors, what we care about is the shape of the
                # underlying data, not the full matrix
                shape = (sig.initial_value.size,)
            else:
                # resize scalars to length 1 vectors
                shape = sig.shape if sig.shape != () else (1,)

            # parameters of signal that affect the base array
            array_params = (dtype, shape[1:], sig.trainable, sig.minibatched)

            # key used to map signals to base arrays
            if array_params not in curr_keys:
                curr_keys[array_params] = object()
            key = curr_keys[array_params]

            initial_value = sig.initial_value
            if is_sparse(sig):
                if isinstance(initial_value, SparseMatrix):
                    initial_value = initial_value.data
                else:
                    initial_value = initial_value.tocoo().data

            initial_value = initial_value.astype(dtype, copy=False)

            # broadcast scalars up to full size
            if initial_value.shape == ():
                initial_value = np.resize(initial_value, shape)

            if sig.minibatched:
                # duplicate along minibatch dimension
                initial_value = np.tile(
                    initial_value[None, ...],
                    (self.minibatch_size,) + tuple(1 for _ in shape),
                )

            if key in base_arrays[sig.trainable]:
                base_arrays[sig.trainable][key][0].append(initial_value)
                base_arrays[sig.trainable][key][1] += shape[0]
            else:
                base_arrays[sig.trainable][key] = [
                    [initial_value],
                    shape[0],
                    sig.minibatched,
                ]

            n = base_arrays[sig.trainable][key][1]
            indices = np.arange(n - shape[0], n)

            tensor_sig = self.signals.get_tensor_signal(
                indices, key, dtype, shape, sig.minibatched, label=sig.name, signal=sig
            )

            logger.debug("created base signal")
            logger.debug(sig)
            logger.debug(tensor_sig)

        # concatenate all the signal initial values into full base arrays
        for trainable in (True, False):
            for key in base_arrays[trainable]:
                minibatched = base_arrays[trainable][key][2]
                base_arrays[trainable][key] = np.concatenate(
                    base_arrays[trainable][key][0], axis=1 if minibatched else 0
                )

        # add any signal views to the sig_map
        all_views = [
            sig
            for ops in self.plan
            for op in ops
            for sig in op.all_signals
            if sig.is_view
        ]
        for sig in all_views:
            if sig.size == sig.base.size:
                # reshape view
                self.signals[sig] = self.signals[sig.base].reshape(sig.shape)
            else:
                if sig.shape[1:] != sig.base.shape[1:]:
                    # TODO: support this?
                    raise NotImplementedError("Slicing on axes > 0 is not supported")

                # slice view
                assert np.all([x == 1 for x in sig.elemstrides[1:]])

                start = sig.elemoffset
                stride = sig.elemstrides[0]
                stop = start + sig.size * stride
                if stop < 0:
                    stop = None

                self.signals[sig] = self.signals[sig.base][slice(start, stop, stride)]

        # error checking
        for sig, tensor_sig in self.signals.items():
            # tensorsignal shapes should match signal shapes
            assert (
                tensor_sig.shape == (sig.size,)
                if is_sparse(sig)
                else (sig.shape if sig.shape != () else (1,))
            )

            # tensorsignal values should match signal values
            initial_value = sig.initial_value
            if is_sparse(sig):
                if isinstance(initial_value, SparseMatrix):
                    initial_value = initial_value.data
                else:
                    initial_value = initial_value.tocoo().data

            base_value = base_arrays[sig.trainable][tensor_sig.key]
            if sig.minibatched:
                initial_value = initial_value[None, ...]
                base_value = base_value[:, tensor_sig.indices]
            else:
                base_value = base_value[tensor_sig.indices]
            assert np.allclose(base_value, initial_value)

        logger.debug("base arrays")
        logger.debug(
            "\n".join(
                [
                    str((k, v.dtype, v.shape, trainable))
                    for trainable in [True, False]
                    for k, v in base_arrays[trainable].items()
                ]
            )
        )

        self.base_arrays_init = base_arrays
