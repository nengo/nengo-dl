"""
Manages all the data and build processes associated with the TensorFlow graph.

The TensorFlow graph is the symbolic description of the computations in the
network, which will be executed by the simulator.
"""

from __future__ import print_function

from collections import OrderedDict, defaultdict
import inspect
import itertools
import logging
import warnings

from nengo import Connection, Process, Ensemble
from nengo.builder.operator import TimeUpdate, SimPyFunc, Reset
from nengo.builder.processes import SimProcess
from nengo.config import ConfigError
from nengo.ensemble import Neurons
from nengo.exceptions import SimulationError, ValidationError
from nengo.neurons import Direct
from nengo.utils.magic import decorator
import numpy as np
import pkg_resources
import tensorflow as tf

from nengo_dl import (builder, graph_optimizer, signals, utils, tensor_node,
                      config)

logger = logging.getLogger(__name__)


@decorator
def with_self(wrapped, instance, args, kwargs):
    """A decorator that can be used to ensure that any ops created within the
    wrapped method will be added to the TensorGraph object's graph."""

    with instance.graph.as_default(), instance.graph.device(instance.device):
        return wrapped(*args, **kwargs)


class TensorGraph:
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
    dtype : ``tf.DType``
        Floating point precision to use for simulation
    minibatch_size : int
        The number of simultaneous inputs that will be passed through the
        network
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        Device on which to execute computations (if None then uses the
        default device as determined by TensorFlow)
    progress : `.utils.ProgressBar`
        Progress bar for optimization stage
    """

    def __init__(self, model, dt, unroll_simulation, dtype, minibatch_size,
                 device, progress):
        self.model = model
        self.dt = dt
        self.unroll = unroll_simulation
        self.dtype = dtype
        self.minibatch_size = minibatch_size
        self.device = device
        self.graph = tf.Graph()
        self.signals = signals.SignalDict(self.dtype, self.minibatch_size)
        self.inference_only = config.get_setting(model, "inference_only",
                                                 False)

        # find invariant inputs (nodes that don't receive any input other
        # than the simulation time). we'll compute these outside the simulation
        # and feed in the result.
        if self.model.toplevel is None:
            self.invariant_inputs = OrderedDict()
        else:
            self.invariant_inputs = OrderedDict(
                (n, n.output) for n in self.model.toplevel.all_nodes
                if n.size_in == 0 and
                not isinstance(n, tensor_node.TensorNode))

        # filter unused operators
        # remove TimeUpdate because it is executed as part of the simulation
        # loop, not part of the step plan. remove input nodes because they
        # are executed outside the simulation.
        node_processes = [n.output for n in self.invariant_inputs
                          if isinstance(n.output, Process)]
        operators = [
            op for op in self.model.operators if not (
                isinstance(op, TimeUpdate) or
                (isinstance(op, SimPyFunc) and op.x is None) or
                (isinstance(op, SimProcess) and op.input is None and
                 op.process in node_processes))]

        # mark trainable signals
        self.mark_signals()

        logger.info("Initial plan length: %d", len(operators))

        # apply graph simplification functions
        simplifications = config.get_setting(model, "simplifications", [
            graph_optimizer.remove_constant_copies,
            graph_optimizer.remove_unmodified_resets,
            graph_optimizer.remove_zero_incs,
            graph_optimizer.remove_identity_muls,
        ])

        with progress.sub("operator simplificaton", max_value=None):
            old_operators = []
            while len(old_operators) != len(operators) or any(
                    x is not y for x, y in zip(operators, old_operators)):
                old_operators = operators
                for simp in simplifications:
                    operators = simp(operators)

        # group mergeable operators
        planner = config.get_setting(
            model, "planner", graph_optimizer.tree_planner)

        with progress.sub("merging operators", max_value=None):
            plan = planner(operators)

        # TODO: we could also merge operators sequentially (e.g., combine
        # a copy and dotinc into one op), as long as the intermediate signal
        # is only written to by one op and read by one op

        # order signals/operators to promote contiguous reads
        sorter = config.get_setting(
            model, "sorter", graph_optimizer.order_signals)

        with progress.sub("ordering signals", max_value=None):
            sigs, self.plan = sorter(plan, n_passes=10)

        # create base arrays and map Signals to TensorSignals (views on those
        # base arrays)
        with progress.sub("creating signals", max_value=None):
            self.create_signals(sigs)

        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info("Number of base arrays: %d", len(self.base_arrays_init))

    @with_self
    def build(self, progress):
        """
        Constructs a new graph to simulate the model.

        progress : `.utils.ProgressBar`
            Progress bar for construction stage
        """

        self.target_phs = {}
        self.outputs = {}
        self.optimizers = {}

        # create these constants once here for reuse in different operators
        self.signals.dt = tf.constant(self.dt, self.dtype)
        self.signals.dt_val = self.dt  # store the actual value as well
        self.signals.zero = tf.constant(0, self.dtype)
        self.signals.one = tf.constant(1, self.dtype)

        if not self.inference_only:
            # this variable controls behaviour in the simulation that is
            # conditional on whether we are doing training or inference
            self.signals.training = tf.placeholder(tf.bool, shape=(),
                                                   name="training")

            # variable to track training step
            self.training_step = tf.train.get_or_create_global_step()
        else:
            self.training_step = None

        # create base arrays
        sub = progress.sub("creating base arrays")
        self.base_vars = OrderedDict()
        unique_ids = defaultdict(int)
        for k, (v, trainable) in sub(self.base_arrays_init.items()):
            name = "%s_%s_%s_%d" % (
                v.dtype, "_".join(str(x) for x in v.shape), trainable,
                unique_ids[(v.dtype, v.shape, trainable)])
            unique_ids[(v.dtype, v.shape, trainable)] += 1

            # we initialize all the variables from placeholders, and then
            # feed in the initial values when the init op is called. this
            # prevents TensorFlow from storing large constants in the graph
            # def, which can cause problems for large models
            ph = tf.placeholder(v.dtype, v.shape, name="%s_init" % name)

            if trainable:
                with tf.variable_scope("trainable_vars", reuse=False):
                    var = tf.get_variable(name, initializer=ph, trainable=True)
            else:
                with tf.variable_scope("local_vars", reuse=False):
                    var = tf.get_local_variable(name, initializer=ph,
                                                trainable=False)

            self.base_vars[k] = (var, ph, v)

        logger.debug("created base arrays")
        logger.debug([str(x[0]) for x in self.base_vars.values()])

        # set up invariant inputs
        sub = progress.sub("building inputs")
        self.build_inputs(sub)

        # pre-build stage
        sub = progress.sub("pre-build stage")
        self.op_builds = {}
        build_config = builder.BuildConfig(
            inference_only=self.inference_only,
            lif_smoothing=config.get_setting(self.model, "lif_smoothing"),
            cpu_only=(
                self.device == "/cpu:0" or
                len([d for d in pkg_resources.working_set if d.project_name in
                     ("tensorflow-gpu", "tf-nightly-gpu")]) == 0),
        )
        for ops in sub(self.plan):
            with self.graph.name_scope(utils.sanitize_name(
                    builder.Builder.builders[type(ops[0])].__name__)):
                builder.Builder.pre_build(ops, self.signals, self.op_builds,
                                          build_config)

        # build stage
        sub = progress.sub("unrolled step ops")
        self.build_loop(sub)

        # ops for initializing variables (will be called by simulator)
        trainable_vars = tf.trainable_variables()
        if not self.inference_only:
            trainable_vars.append(self.training_step)
        self.trainable_init_op = tf.variables_initializer(trainable_vars)
        self.local_init_op = tf.local_variables_initializer()
        self.global_init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v not in trainable_vars])
        self.constant_init_op = tf.variables_initializer(
            tf.get_collection("constants"))

        # logging
        logger.info("Number of reads: %d", sum(
            x for x in self.signals.read_types.values()))
        for x in self.signals.read_types.items():
            logger.info("    %s: %d", *x)
        logger.info("Number of writes: %d", sum(
            x for x in self.signals.write_types.values()))
        for x in self.signals.write_types.items():
            logger.info("    %s: %d", *x)

    def build_step(self):
        """
        Build the operators that execute a single simulation timestep
        into the graph.

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
        side_effects = []

        # manually build TimeUpdate. we don't include this in the plan,
        # because loop variables (`step`) are (semi?) pinned to the CPU, which
        # causes the whole variable to get pinned to the CPU if we include
        # `step` as part of the normal planning process.
        self.signals.time = tf.cast(self.signals.step,
                                    self.dtype) * self.signals.dt

        # build operators
        for ops in self.plan:
            with self.graph.name_scope(utils.sanitize_name(
                    builder.Builder.builders[type(ops[0])].__name__)):
                outputs = builder.Builder.build(ops, self.signals,
                                                self.op_builds)

            if outputs is not None:
                side_effects += outputs

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
                probe_tensors.append(self.signals.gather(
                    self.signals[probe_sig], force_copy=True))
            else:
                # if a probe signal isn't in sig_map, that means that it isn't
                # involved in any simulator ops.  so we know its value never
                # changes, and we'll just return a constant containing the
                # initial value.
                if probe_sig.minibatched:
                    init_val = np.tile(probe_sig.initial_value[..., None],
                                       (1, self.minibatch_size))
                else:
                    init_val = probe_sig.initial_value
                probe_tensors.append(tf.constant(init_val, dtype=self.dtype))

        logger.debug("=" * 30)
        logger.debug("build_step complete")
        logger.debug("probe_tensors %s", [str(x) for x in probe_tensors])
        logger.debug("side_effects %s", [str(x) for x in side_effects])

        return probe_tensors, side_effects

    def build_loop(self, progress):
        """
        Build simulation loop.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for loop construction
        """

        def loop_condition(step, stop, *_):
            return step < stop

        def loop_body(step, stop, loop_i, probe_arrays, base_vars):
            # fill in signals.bases (note: we need to do this here because we
            # need to use the versions of the base vars from inside the
            # loop, not the static variables in self.base_vars)
            assert len(self.signals.bases) == 0
            for i, key in enumerate(itertools.chain(
                    self.base_vars.keys(), self.signals.internal_vars.keys())):
                self.signals.bases[key] = base_vars[i]

            for iter in progress(range(self.unroll)):
                logger.debug("BUILDING ITERATION %d", iter)
                with self.graph.name_scope("iteration_%d" % iter):
                    # note: nengo step counter is incremented at the beginning
                    # of the timestep
                    step += 1
                    self.signals.step = step

                    # fill in invariant input data
                    for n in self.input_ph:
                        self.signals.scatter(
                            self.signals[self.model.sig[n]["out"]],
                            self.input_ph[n][loop_i])

                    # build the operators for a single step
                    # note: we tie things to the `loop_i` variable so that we
                    # can be sure the other things we're tying to the
                    # simulation step (side effects and probes) from the
                    # previous timestep are executed before the next step
                    # starts
                    with self.graph.control_dependencies([loop_i]):
                        # note: we use the variable scope to make sure that we
                        # aren't accidentally creating new variables for
                        # unrolled iterations (this is really only a concern
                        # with TensorNodes)
                        with tf.variable_scope(tf.get_variable_scope(),
                                               reuse=iter > 0):
                            probe_tensors, side_effects = self.build_step()

                    # copy probe data to array
                    for i, p in enumerate(probe_tensors):
                        if config.get_setting(
                                self.model, "keep_history",
                                default=True, obj=self.model.probes[i]):
                            probe_arrays[i] = probe_arrays[i].write(loop_i, p)
                        else:
                            probe_arrays[i] = tf.cond(
                                tf.equal(step, stop),
                                lambda p=p: probe_arrays[i].write(0, p),
                                lambda: probe_arrays[i])

                    # need to make sure that any operators that could have side
                    # effects run each timestep, so we tie them to the loop
                    # increment. we also need to make sure that all the probe
                    # reads happen before those values get overwritten on the
                    # next timestep
                    with self.graph.control_dependencies(side_effects +
                                                         probe_tensors):
                        loop_i += 1

            base_vars = tuple(self.signals.bases.values())

            return step, stop, loop_i, probe_arrays, base_vars

        self.step_var = tf.placeholder(tf.int32, shape=(), name="step")
        self.stop_var = tf.placeholder(tf.int32, shape=(), name="stop")
        loop_i = tf.constant(0)

        probe_arrays = [
            tf.TensorArray(
                self.dtype, clear_after_read=True, size=0,
                dynamic_size=True)
            for _ in self.model.probes]

        # build simulation loop
        loop_vars = (
            self.step_var, self.stop_var, loop_i, probe_arrays,
            tuple(x[0]._ref() if isinstance(x[0], tf.Variable) else x[0]
                  for x in self.base_vars.values()) +
            tuple(x._ref() for x in self.signals.internal_vars.values()))

        loop_vars = tf.while_loop(
            loop_condition, loop_body, loop_vars=loop_vars,
            parallel_iterations=1, back_prop=not self.inference_only)

        self.steps_run = loop_vars[2]
        self.probe_arrays = OrderedDict()
        for p, a in zip(self.model.probes, loop_vars[3]):
            x = a.stack()

            if self.model.sig[p]["in"].minibatched:
                x = tf.transpose(x, np.roll(np.arange(x.get_shape().ndims), 1))
            else:
                x = tf.expand_dims(x, 0)

            self.probe_arrays[p] = x

    def build_inputs(self, progress):
        """
        Sets up the inputs in the model (which will be computed outside of
        TensorFlow and fed in each simulation block).

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for input construction
        """

        self.input_ph = {}
        for n in progress(self.invariant_inputs):
            if self.model.sig[n]["out"] in self.signals:
                # set up a placeholder input for this node
                self.input_ph[n] = tf.placeholder(
                    self.dtype, (None, n.size_out, self.minibatch_size),
                    name="%s_ph" % utils.sanitize_name(n))

    def build_optimizer_func(self, optimizer, objective):
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

        key = (optimizer, frozenset(objective.items()))

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
            # note: we don't actually use outputs/targets, because the same
            # data is pulled implicitly from `objective` below.
            # we just check that outputs and targets match up with
            # objective, to make sure there's nothing weird going on.
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            if not isinstance(targets, tuple):
                targets = (targets,)
            assert set(outputs) == set(self.probe_arrays[p] for p in objective)
            assert set(targets) == set(self.target_phs[p] for p in objective)

            agg_method = tf.AggregationMethod.DEFAULT
            grads = []
            vars = tf.trainable_variables()

            # compute loss
            # note: we drop the `None` items in objective, because we
            # want to treat those as direct gradients (rather than
            # returning the probe value, which is the standard behaviour for
            # build_outputs)
            loss, _ = self.build_outputs(
                {k: v for k, v in objective.items() if v is not None})

            # compute gradients wrt loss
            if len(loss) > 0:
                # reduce loss to a scalar
                loss = tf.reduce_sum([tf.reduce_sum(v) for v in loss.values()])

                grads.append(tf.gradients(
                    loss, vars, aggregation_method=agg_method))

            # add in any gradients where the user directly specified the output
            # error grad
            for p, g in objective.items():
                if g is None:
                    grads.append(tf.gradients(
                        self.probe_arrays[p], vars, grad_ys=self.target_phs[p],
                        aggregation_method=agg_method))

            # combine gradients for each variable
            if len(grads) == 1:
                grads = grads[0]
            else:
                grads = [tf.reduce_sum(gs, axis=0) for gs in zip(*grads)]

            opt_op = optimizer.apply_gradients(
                zip(grads, tf.trainable_variables()),
                global_step=self.training_step)

            # this is the op that increments the global step. we set it to
            # be the output value of that op, rather than the op itself, so
            # that it returns the global step value.
            opt_op = opt_op.outputs[0]

            return opt_op, loss

        self.optimizers[key] = apply_optimizer

        return apply_optimizer

    @with_self
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
        for probes, out in outputs.items():
            is_tuple = isinstance(probes, tuple)
            probe_arrays = (
                tuple(self.probe_arrays[p] for p in probes) if is_tuple else
                self.probe_arrays[probes])

            if out is None:
                # return probe output value
                output_vals[probes] = probe_arrays
            elif callable(out):
                # look up number of positional arguments for function
                spec = inspect.getfullargspec(out)

                nargs = len(spec.args)
                if spec.defaults is not None:
                    # don't count keyword arguments
                    nargs -= len(spec.defaults)
                if inspect.ismethod(out) or not inspect.isroutine(out):
                    # don't count self argument for methods or callable classes
                    nargs -= 1

                # build function arguments
                if nargs == 1:
                    args = [probe_arrays]
                elif nargs == 2:
                    for p in probes if is_tuple else (probes,):
                        # create a placeholder for the target values if one
                        # hasn't been created yet
                        if p not in self.target_phs:
                            self.target_phs[p] = tf.placeholder(
                                self.dtype,
                                (self.minibatch_size, None, p.size_in),
                                name="%s_ph" % utils.sanitize_name(p))
                    target_phs = (tuple(self.target_phs[p] for p in probes)
                                  if is_tuple else self.target_phs[probes])
                    args = [probe_arrays, target_phs]
                else:
                    raise ValidationError(
                        "Output functions must accept 1 or 2 arguments; '%s' "
                        "takes %s arguments" % (
                            utils.function_name(out, sanitize=False), nargs),
                        "outputs")

                # apply output function
                with tf.variable_scope(utils.function_name(out)) as scope:
                    output_vals[probes] = out(*args)

                # collect any new variables from building the outputs
                for collection in [tf.GraphKeys.GLOBAL_VARIABLES,
                                   tf.GraphKeys.LOCAL_VARIABLES,
                                   "gradient_vars"]:
                    new_vars.extend(scope.get_collection(collection))
            else:
                raise ValidationError("Outputs must be callable or None)",
                                      "outputs")

        new_vars_init = (tf.variables_initializer(new_vars)
                         if len(new_vars) > 0 else None)

        self.outputs[key] = output_vals

        return output_vals, new_vars_init

    @with_self
    def build_post(self, sess, rng):
        """
        Executes post-build processes for operators (after the graph has
        been constructed and session/variables initialized).

        Note that unlike other build functions, this is called every time
        the simulator is reset.

        Parameters
        ----------
        sess : ``tf.Session``
            The TensorFlow session for the simulator
        rng : `~numpy.random.RandomState`
            Seeded random number generator
        """

        # build input functions (we need to do this here, because in the case
        # of processes these functions depend on the rng, and need to be be
        # rebuilt on reset)
        self.input_funcs = {}
        for n, output in self.invariant_inputs.items():
            if isinstance(output, np.ndarray):
                self.input_funcs[n] = output
            elif isinstance(output, Process):
                self.input_funcs[n] = [
                    output.make_step(
                        (n.size_in,), (n.size_out,), self.dt,
                        output.get_rng(rng))
                    for _ in range(self.minibatch_size)]
            elif n.size_out > 0:
                self.input_funcs[n] = [
                    utils.align_func((n.size_out,), self.dtype)(output)]
            else:
                # a node with no inputs and no outputs, but it can still
                # have side effects
                self.input_funcs[n] = [output]

        # call build_post on all the op builders
        for ops, built_ops in self.op_builds.items():
            built_ops.build_post(ops, self.signals, sess, rng)

    @with_self
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
                    summary_ops.append(tf.summary.scalar(
                        "loss", tf.reduce_sum([tf.reduce_sum(v)
                                               for v in loss.values()]),
                        family="loss"))

                    if len(obj) > 1:
                        # get loss for each probe
                        for p, t in loss.items():
                            summary_ops.append(tf.summary.scalar(
                                utils.sanitize_name("Probe_%s_loss" % p.label),
                                tf.reduce_sum(t), family="loss"))
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

                    summary_ops.append(tf.summary.histogram(
                        utils.sanitize_name("%s_%s" % (name, param)),
                        self.get_tensor(self.model.sig[obj][param])))
                elif isinstance(obj, tf.Tensor):
                    # we assume that obj is a summary op
                    summary_ops.append(obj)
                else:
                    raise SimulationError(
                        "Unknown summary object: %s" % obj)

            return tf.summary.merge(summary_ops), (None if len(inits) == 0 else
                                                   inits)

    @with_self
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

        base = self.base_vars[tensor_sig.key][0]

        if "while/" in tensor_sig.tf_indices.name:
            # rebuild tf indices outside the while loop
            tensor_sig._tf_indices = None

        return tf.gather(base, tensor_sig.tf_indices)

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
                elif network_trainable is not 1:
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
                mark_network(net_config, subnet,
                             get_trainable(net_config, subnet,
                                           network_trainable))

            # encoders and biases are trainable
            for ens in net.ensembles:
                ens_trainable = get_trainable(net_config, ens,
                                              network_trainable)

                self.model.sig[ens]["encoders"].trainable = ens_trainable
                self.model.sig[ens]["encoders"].minibatched = False

                if not isinstance(ens.neuron_type, Direct):
                    neurons_trainable = get_trainable(net_config, ens.neurons,
                                                      network_trainable)
                    if neurons_trainable is 1:
                        neurons_trainable = ens_trainable

                    self.model.sig[ens.neurons]["bias"].trainable = (
                        neurons_trainable)
                    self.model.sig[ens.neurons]["bias"].minibatched = False

            # connection weights are trainable
            for conn in net.connections:
                # note: this doesn't include probe connections, since they
                # aren't added to the network
                self.model.sig[conn]["weights"].trainable = get_trainable(
                    net_config, conn, network_trainable)
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
                                "produce strange training behaviour." %
                                obj)
                        else:
                            self.model.sig[obj][attr].trainable = False

                        self.model.sig[obj][attr].minibatched = True

        if self.model.toplevel is None:
            warnings.warn(
                "No top-level network in model; assuming no trainable "
                "parameters", UserWarning)
        else:
            net_config = self.model.toplevel.config
            mark_network(net_config, self.model.toplevel,
                         get_trainable(net_config, self.model.toplevel, 1))

            # the connections to connection probes are not trainable, but
            # also not minibatched
            probe_seeds = [self.model.seeds[p] for p in self.model.probes]
            for obj, seed in self.model.seeds.items():
                if isinstance(obj, Connection) and seed in probe_seeds:
                    self.model.sig[obj]["weights"].trainable = False
                    self.model.sig[obj]["weights"].minibatched = False

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

        float_type = self.dtype.as_numpy_dtype
        base_arrays = OrderedDict()
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
        logging.debug("\n%s", "".join("|" if i in breaks else " "
                                      for i in range(len(sigs))))

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

            # resize scalars to length 1 vectors
            shape = sig.shape if sig.shape != () else (1,)

            # parameters of signal that affect the base array
            array_params = (dtype, shape[1:], sig.trainable, sig.minibatched)

            # key used to map signals to base arrays
            if array_params not in curr_keys:
                curr_keys[array_params] = object()
            key = curr_keys[array_params]

            initial_value = sig.initial_value.astype(dtype, copy=False)

            # broadcast scalars up to full size
            if initial_value.shape != shape:
                initial_value = np.resize(initial_value, shape)

            if sig.minibatched:
                # duplicate along minibatch dimension
                initial_value = np.tile(
                    initial_value[..., None],
                    tuple(1 for _ in shape) + (self.minibatch_size,))

            if key in base_arrays:
                base_arrays[key][0].append(initial_value)
                base_arrays[key][2] += shape[0]
            else:
                base_arrays[key] = [[initial_value], sig.trainable, shape[0]]

            n = base_arrays[key][-1]
            indices = np.arange(n - shape[0], n)

            tensor_sig = self.signals.get_tensor_signal(
                indices, key, dtype, shape, sig.minibatched, label=sig.name,
                signal=sig)

            logger.debug("created base signal")
            logger.debug(sig)
            logger.debug(tensor_sig)

        for key in base_arrays:
            arrs, t, _ = base_arrays[key]
            base_arrays[key] = (np.concatenate(arrs, axis=0), t)

        # add any signal views to the sig_map
        all_views = [sig for ops in self.plan for op in ops for sig in
                     op.all_signals if sig.is_view]
        for sig in all_views:
            if sig.size == sig.base.size:
                # reshape view
                self.signals[sig] = self.signals[sig.base].reshape(sig.shape)
            else:
                if sig.shape[1:] != sig.base.shape[1:]:
                    # TODO: support this?
                    raise NotImplementedError(
                        "Slicing on axes > 0 is not supported")

                # slice view
                assert np.all([x == 1 for x in sig.elemstrides[1:]])

                start = sig.elemoffset
                stride = sig.elemstrides[0]
                stop = start + sig.size * stride
                if stop < 0:
                    stop = None

                self.signals[sig] = self.signals[sig.base][slice(start, stop,
                                                                 stride)]

        # error checking
        for sig, tensor_sig in self.signals.items():
            # tensorsignal shapes should match signal shapes
            assert tensor_sig.shape == (sig.shape if sig.shape != () else (1,))

            # tensorsignal values should match signal values
            initial_value = sig.initial_value
            if sig.minibatched:
                initial_value = initial_value[..., None]

            assert np.allclose(
                base_arrays[tensor_sig.key][0][tensor_sig.indices],
                initial_value.astype(dtype))

        logger.debug("base arrays")
        logger.debug("\n".join([str((k, v.dtype, v.shape, trainable))
                                for k, (v, trainable) in base_arrays.items()]))

        self.base_arrays_init = base_arrays
