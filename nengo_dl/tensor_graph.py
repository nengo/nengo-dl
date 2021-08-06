"""
Manages the data and build processes associated with implementing a Nengo simulation
in TensorFlow.
"""

import logging
import warnings
from collections import defaultdict

import numpy as np
import tensorflow as tf
from nengo import Connection, Process
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import Reset, SimPyFunc, TimeUpdate
from nengo.builder.processes import SimProcess
from nengo.config import ConfigError
from nengo.exceptions import BuildError
from nengo.neurons import Direct
from nengo.synapses import Lowpass
from nengo.transforms import SparseMatrix
from tensorflow.python.eager import context
from tensorflow.python.training.tracking import base as trackable

from nengo_dl import (
    builder,
    compat,
    config,
    graph_optimizer,
    signals,
    tensor_node,
    utils,
)

logger = logging.getLogger(__name__)


class TensorGraph(tf.keras.layers.Layer):
    """
    Implement the Nengo simulation as a Keras Layer.

    Parameters
    ----------
    model : `~nengo.builder.Model`
        Pre-built Nengo model describing the network to be simulated.
    dt : float
        Length of a simulator timestep, in seconds.
    unroll_simulation : int
        Unroll simulation loop by explicitly building ``unroll_simulation``
        iterations into the computation graph.
    minibatch_size : int
        The number of simultaneous inputs that will be passed through the
        network.
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        Device on which to execute computations (if None then uses the
        default device as determined by TensorFlow).
    progress : `.utils.ProgressBar`
        Progress bar for optimization stage.
    seed : int
        Seed for random number generation.
    """

    @trackable.no_automatic_dependency_tracking
    def __init__(
        self, model, dt, unroll_simulation, minibatch_size, device, progress, seed
    ):
        super().__init__(
            name="TensorGraph",
            dynamic=False,
            trainable=not config.get_setting(model, "inference_only", False),
            dtype=config.get_setting(model, "dtype", "float32"),
            batch_size=minibatch_size,
        )

        self.model = model
        self.dt = dt
        self.unroll = unroll_simulation
        self.use_loop = config.get_setting(model, "use_loop", True)
        self.minibatch_size = minibatch_size
        self.device = device
        self.seed = seed
        self.inference_only = not self.trainable
        self.signals = signals.SignalDict(self.dtype, self.minibatch_size)

        # find invariant inputs (nodes that don't receive any input other
        # than the simulation time). we'll compute these outside the simulation
        # and feed in the result.
        if self.model.toplevel is None:
            self.invariant_inputs = {}
        else:
            self.invariant_inputs = {
                n: n.output
                for n in self.model.toplevel.all_nodes
                if n.size_in == 0 and not isinstance(n, tensor_node.TensorNode)
            }

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

        # check for unsupported operators
        for op in operators:
            if type(op) not in builder.Builder.builders:
                raise BuildError(
                    "No registered builder for operators of type %s; "
                    "consider registering a custom builder" % type(op)
                )

        # mark trainable signals
        self.mark_signals()

        logger.info("Initial plan length: %d", len(operators))

        # apply graph simplification functions
        simplifications = config.get_setting(
            model, "simplifications", graph_optimizer.default_simplifications
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

        # generate unique names for layer inputs/outputs
        # this follows the TensorFlow unique naming scheme, so if multiple objects are
        # created with the same name, they will be named like name, NAME_1, name_2
        # (note: case insensitive)
        self.io_names = {}
        name_count = defaultdict(int)
        for obj in list(self.invariant_inputs.keys()) + self.model.probes:
            name = (
                type(obj).__name__.lower()
                if obj.label is None
                else utils.sanitize_name(obj.label)
            )

            key = name.lower()

            if name_count[key] > 0:
                name += f"_{name_count[key]}"

            self.io_names[obj] = name
            name_count[key] += 1

        # set up op builder
        self.op_builder = builder.Builder(self.plan)

        # logging
        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info(
            "Number of base arrays: (%s, %d), (%s, %d), (%s, %d)",
            *sum(((k, len(x)) for k, x in self.base_arrays_init.items()), ()),
        )

    def build_inputs(self):
        """
        Generates a set of Input layers that can be used as inputs to a
        TensorGraph layer.

        Returns
        -------
        n_steps : ``tf.keras.layers.Input``
            Input layer for specifying the number of simulation timesteps.
        inputs : dict of {`nengo.Node`: ``tf.keras.layers.Input``}
            Input layers for each of the Nodes in the network.
        """

        # input placeholders
        inputs = {}
        for n in self.invariant_inputs:
            inputs[n] = tf.keras.layers.Input(
                shape=(None, n.size_out),
                batch_size=self.minibatch_size,
                dtype=self.dtype,
                name=self.io_names[n],
            )

        # number of steps to run
        n_steps = tf.keras.layers.Input(
            shape=(1,), batch_size=self.minibatch_size, dtype="int32", name="n_steps"
        )

        return inputs, n_steps

    def build(self, input_shape=None):
        """
        Create any Variables used in the model.

        Parameters
        ----------
        input_shape : list of tuple of int
            Shapes of all the inputs to this layer.
        """

        super().build(input_shape)

        tf.random.set_seed(self.seed)

        def get_initializer(init_vals):
            """Use more efficient initializers if possible to save memory."""

            values, shapes, dtype, minibatched = init_vals

            # initial value of None means that the initial value isn't used, so we
            # can use anything for the initial value
            if all(v is None for v in values):
                initializer = None
            elif all(v is None or np.all(v == 0) for v in values):
                initializer = tf.initializers.zeros()
            elif all(v is None or np.all(v == 1) for v in values):
                initializer = tf.initializers.ones()
            else:
                val = tf.constant(
                    np.concatenate(
                        [
                            np.zeros(s, dtype)
                            if v is None
                            else np.broadcast_to(np.asarray(v, dtype=dtype), s)
                            for v, s in zip(values, shapes)
                        ],
                        axis=1 if minibatched else 0,
                    ),
                    dtype=dtype,
                )
                initializer = lambda shape=None, dtype=None: val

            # figure out shape of full concatenated initial value
            shape = list(shapes[0])
            shape[minibatched] = sum(x[minibatched] for x in shapes)

            return initializer, tuple(shape), dtype

        # save initializers so that we can reset the model later
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.initial_values = {}

        # variables for model parameters
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.base_params = {}
        assert len(self.base_params) == 0
        for sig_type in ("trainable", "non_trainable"):
            for k, v in self.base_arrays_init[sig_type].items():
                initializer, shape, dtype = get_initializer(v)
                assert initializer is not None  # params should never be set
                self.base_params[k] = self.add_weight(
                    initializer=initializer,
                    shape=shape,
                    dtype=dtype,
                    trainable=sig_type == "trainable",
                    name=f"base_params/{sig_type}_{dtype}_"
                    f"{'_'.join(str(x) for x in shape)}",
                )

                self.initial_values[k] = initializer

        logger.debug("created base param variables")
        logger.debug([str(x) for x in self.base_params.values()])

        # variables to save the internal state of simulation between runs
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.saved_state = {}
        for k, v in self.base_arrays_init["state"].items():
            initializer, shape, dtype = get_initializer(v)
            if initializer is not None:
                # don't need to save the state for signals where the initial value
                # doesn't matter
                self.saved_state[k] = tf.Variable(
                    initial_value=lambda: initializer(shape=shape, dtype=dtype),
                    shape=shape,
                    dtype=dtype,
                    trainable=False,
                    name=f"saved_state/{dtype}_{'_'.join(str(x) for x in shape)}",
                )

                self.initial_values[k] = initializer

        logger.debug("created saved state variables")
        logger.debug([str(x) for x in self.saved_state.values()])

        # call build on any TensorNode Layers

        def unbuild(layer):
            assert layer.built

            # clear any losses attached to layer (they will be recreated in the
            # build step, so we don't want to keep around any losses
            # associated with the previous build)
            # note: not clearing layer._losses, because those are manually added
            # by the user (not created during the build process)
            layer._eager_losses = []
            layer._callable_losses = []

            layer.built = False

            for sub in compat.sub_layers(layer):
                if isinstance(sub, tf.keras.layers.Layer):
                    unbuild(sub)

        layer_ops = [
            op
            for ops in self.plan
            if isinstance(ops[0], tensor_node.SimTensorNode)
            for op in ops
            if isinstance(op.func, tf.keras.layers.Layer)
        ]
        weight_gets = []
        weight_sets = []
        for op in layer_ops:
            if op.func in compat.sub_layers(self):
                # already built this layer
                continue

            if op.time is None:
                shape_in = []
            else:
                shape_in = [()]
            if op.input is not None:
                shape_in += [(self.minibatch_size,) + op.shape_in]
            if len(shape_in) == 1:
                shape_in = shape_in[0]

            if op.func.built:
                # we rebuild the layer (even if it is already built),
                # because we need to build the weights within the TensorGraph
                # context

                # save the weight values so they can be restored
                # exactly inside the tensornode
                weights = op.func.weights
                weight_gets.extend(weights)

                # clear the results of previous build
                unbuild(op.func)
            else:
                weights = None

            with tf.name_scope(op.func.name):
                op.func.build(shape_in)

            if weights is not None:
                weight_sets.extend(op.func.weights)

            # add op func to _layers so that any weights are collected
            compat.sub_layers(self).append(op.func)

        if len(weight_gets) > 0:
            # do all the weight getting/setting in one go, for efficiency reasons

            # match the fetch context to the context in which the weights were created
            ctx = (
                weight_gets[0].graph.as_default()
                if hasattr(weight_gets[0], "graph")
                else context.eager_mode()
            )
            with ctx:
                weight_vals = tf.keras.backend.batch_get_value(weight_gets)

            tf.keras.backend.batch_set_value(zip(weight_sets, weight_vals))

        if not compat.eager_enabled():
            # initialize state variables (need to do this manually because we're not
            # adding them to self.weights)
            tf.keras.backend.batch_get_value(
                [var.initializer for var in self.saved_state.values()]
            )

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, progress=None, stateful=False):
        """
        Constructs the graph elements to simulate the model.

        Parameters
        ----------
        inputs : list of ``tf.Tensor``
            Input layers/tensors for the network (must match the structure defined in
            `.build_inputs`).
        training : bool
            Whether the network is being run in training or inference mode.  If None,
            uses the symbolic Keras learning phase variable.
        progress : `.utils.ProgressBar`
            Progress bar for construction stage.
        stateful : bool
            Whether or not to build the model to support preserving the internal state
            between executions.

        Returns
        -------
        probe_arrays : list of ``tf.Tensor``
            Tensors representing the output of all the Probes in the network (order
            corresponding to ``self.model.probes``, which is the order the Probes were
            instantiated).
        """

        override_training = config.get_setting(self.model, "learning_phase", None)
        training = training if override_training is None else override_training

        super().call(inputs, training=training)

        if training is True and self.inference_only:
            raise BuildError(
                f"TensorGraph was created with inference_only=True; cannot be called "
                f"with training={training}"
            )

        tf.random.set_seed(self.seed)

        if progress is None:
            progress = utils.NullProgressBar()

        # reset signaldict
        self.signals.reset()

        # create these constants once here for reuse in different operators
        self.signals.dt = tf.constant(self.dt, self.dtype)
        self.signals.dt_val = self.dt  # store the actual value as well
        self.signals.zero = tf.constant(0, self.dtype)
        self.signals.one = tf.constant(1, self.dtype)

        # set up invariant inputs
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.node_inputs = {}
        for n, inp in zip(self.invariant_inputs, inputs):
            # specify shape of inputs (keras sometimes loses this shape information)
            inp.set_shape([self.minibatch_size, inp.shape[1], n.size_out])

            self.node_inputs[n] = inp

        self.steps_to_run = inputs[-1][0, 0]

        # set up build config
        # TODO: it would be nicer if buildconfig was static (i.e. find a separate
        #  way to pass around `training`)
        build_config = builder.BuildConfig(
            inference_only=self.inference_only,
            lif_smoothing=config.get_setting(self.model, "lif_smoothing"),
            cpu_only=(self.device is not None and "cpu" in self.device.lower())
            or not utils.tf_gpu_installed,
            rng=np.random.RandomState(self.seed),
            training=(
                tf.keras.backend.learning_phase() if training is None else training
            ),
        )

        # pre-build stage
        with progress.sub("pre-build stage", max_value=len(self.plan)) as sub:
            self.op_builder.build_pre(self.signals, build_config, sub)

        # build stage
        with progress.sub("build stage", max_value=len(self.plan) * self.unroll) as sub:
            steps_run, probe_arrays, final_internal_state, final_base_params = (
                self._build_loop(sub) if self.use_loop else self._build_no_loop(sub)
            )

        # store these so that they can be accessed after the initial build
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.steps_run = steps_run
            self.probe_arrays = probe_arrays
            self.final_internal_state = final_internal_state
            self.final_base_params = final_base_params

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
        outputs = list(probe_arrays.values()) + [steps_run]

        updates = []
        if stateful:
            # update saved state
            for var, val in zip(self.saved_state.values(), final_internal_state):
                updates.append(var.assign(val))

        # if any of the base params have changed (due to online learning rules) then we
        # also need to assign those back to the original variable (so that their
        # values will persist). any parameters targeted by online learning rules
        # will be minibatched, so we only need to update the minibatched params.
        for (key, var), val in zip(self.base_params.items(), final_base_params):
            try:
                minibatched = self.base_arrays_init["non_trainable"][key][-1]
            except KeyError:
                minibatched = self.base_arrays_init["trainable"][key][-1]

            if minibatched:
                updates.append(var.assign(val))

        logger.info("Number of state updates: %d", len(updates))

        if not compat.eager_enabled() and len(updates) > 0:
            with tf.control_dependencies(updates):
                outputs = [tf.identity(x) for x in outputs]

        return outputs

    def _fill_bases(self, saved_state, base_params):
        """
        Initialize signals.bases from TensorGraph params.

        Parameters
        ----------
        saved_state : dict
            Mapping from base keys to initial values
        base_params : dict
            Mapping from base keys to initial values
        """

        for key, val in saved_state.items():
            # we add the tf.identity so that when we write we're not updating
            # the base variable
            self.signals.bases[key] = tf.identity(val)
        for key, val in base_params.items():
            self.signals.bases[key] = tf.identity(val)
        for key, (_, shapes, _, minibatched) in self.base_arrays_init["state"].items():
            if key not in self.signals.bases:
                # no saved state for this base, so we just temporarily insert
                # the shape information so that future scatters will know
                # what the base shape is
                shape = list(shapes[0])
                shape[minibatched] = sum(x[minibatched] for x in shapes)
                self.signals.bases[key] = tuple(shape)

    def _build_loop(self, progress):
        """
        Build simulation loop using symbolic while loop.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for loop construction

        Returns
        -------
        steps_run : ``tf.Tensor``
            The number of simulation steps that were executed.
        probe_arrays : dict of {`nengo.Probe`: ``tf.Tensor``}
            Arrays containing the output values for each Probe.
        final_internal_state: list of ``tf.Tensor``
            Tensors representing the value of all internal state at the end of the run.
        """

        def loop_condition(loop_i, n_steps, *_):
            return loop_i < n_steps

        def loop_body(loop_i, n_steps, probe_arrays, saved_state, base_params):
            # fill in signals.bases
            # note: we need to do this here because we
            # need to use the tensors from inside the loop, not the source variables)
            self._fill_bases(
                dict(zip(self.saved_state, saved_state)),
                dict(zip(self.base_params, base_params)),
            )

            def update_probes(probe_tensors, loop_i):
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
                            true_fn=lambda p=p, i=i: probe_arrays[i].write(0, p),
                            false_fn=lambda i=i: probe_arrays[i],
                        )

            loop_i = self._build_inner_loop(loop_i, update_probes, progress)

            state_arrays = tuple(self.signals.bases[key] for key in self.saved_state)
            base_arrays = tuple(self.signals.bases[key] for key in self.base_params)

            return loop_i, n_steps, probe_arrays, state_arrays, base_arrays

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
            tuple(self.saved_state.values()),
            tuple(self.base_params.values()),
        )

        loop_vars = tf.while_loop(
            cond=loop_condition,
            body=loop_body,
            loop_vars=loop_vars,
            parallel_iterations=1,  # TODO: check performance impact
        )

        # change to shape (minibatch_size,) (required by keras) instead of a scalar
        steps_run = tf.tile(tf.expand_dims(loop_vars[0], 0), (self.minibatch_size,))

        probe_arrays = {}
        for p, a in zip(self.model.probes, loop_vars[2]):
            x = a.stack()

            if self.model.sig[p]["in"].minibatched:
                # change from tensorarray's (steps, batch, d) to (batch, steps, d)
                perm = np.arange(x.shape.ndims)
                perm[[0, 1]] = perm[[1, 0]]
                x = tf.transpose(x, perm=perm)
            else:
                # add minibatch dimension for consistency
                x = tf.expand_dims(x, 0)

            probe_arrays[p] = x

        final_internal_state = loop_vars[3]
        final_base_params = loop_vars[4]

        return steps_run, probe_arrays, final_internal_state, final_base_params

    def _build_no_loop(self, progress):
        """
        Build simulation loop through explicit unrolling.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for loop construction

        Returns
        -------
        steps_run : ``tf.Tensor``
            The number of simulation steps that were executed.
        probe_arrays : dict of {`nengo.Probe`: ``tf.Tensor``}
            Arrays containing the output values for each Probe.
        final_internal_state: list of ``tf.Tensor``
            Tensors representing the value of all internal state at the end of the run.
        """

        self._fill_bases(self.saved_state, self.base_params)

        loop_i = tf.constant(0)  # symbolic loop variable
        loop_iter = 0  # non-symbolic loop variable
        probe_data = [[] for _ in self.model.probes]

        def update_probes(probe_tensors, _):
            nonlocal loop_iter

            for i, p in enumerate(probe_tensors):
                if config.get_setting(
                    self.model, "keep_history", default=True, obj=self.model.probes[i]
                ):
                    probe_data[i].append(p)
                elif loop_iter == self.unroll - 1:
                    probe_data[i].append(p)

            loop_iter += 1

        loop_i = self._build_inner_loop(loop_i, update_probes, progress)

        # change to shape (minibatch_size,) (required by keras) instead of a scalar
        steps_run = tf.tile(tf.expand_dims(loop_i, 0), (self.minibatch_size,))

        probe_arrays = {}
        for p, a in zip(self.model.probes, probe_data):
            if self.model.sig[p]["in"].minibatched:
                x = tf.stack(a, axis=1)
            else:
                x = tf.stack(a, axis=0)

                # add minibatch dimension for consistency
                x = tf.expand_dims(x, 0)

            probe_arrays[p] = x

        final_internal_state = tuple(
            self.signals.bases[key] for key in self.saved_state
        )
        final_base_params = tuple(self.signals.bases[key] for key in self.base_params)

        return steps_run, probe_arrays, final_internal_state, final_base_params

    def _build_inner_loop(self, loop_i, update_probes, progress):
        """

        Parameters
        ----------
        loop_i : ``tf.Tensor``
            Loop iteration variable.
        update_probes : callable
            Function that will update some stored probe data in each iteration.
        progress
            Progress bar for loop construction.

        Returns
        -------
        loop_i : ``tf.Tensor``
            Updated loop iteration variable.
        """

        for unroll_iter in range(self.unroll):
            logger.debug("BUILDING ITERATION %d", unroll_iter)
            with tf.name_scope(f"iteration_{unroll_iter}"):
                # fill in invariant input data
                for node, vals in self.node_inputs.items():
                    if self.model.sig[node]["out"] in self.signals:
                        # if the out signal doesn't exist then that means that
                        # the node output isn't actually used anywhere, so we can
                        # ignore it
                        self.signals.scatter(
                            self.signals[self.model.sig[node]["out"]],
                            vals[:, loop_i],
                        )

                # build the operators for a single step
                # note: we tie things to the `loop_i` variable so that we
                # can be sure the other things we're tying to the
                # simulation step (side effects and probes) from the
                # previous timestep are executed before the next step
                # starts
                with tf.control_dependencies([loop_i]):
                    # build operators
                    side_effects = self.op_builder.build_step(self.signals, progress)

                    logger.debug("collecting probe tensors")
                    probe_tensors = []
                    for p in self.model.probes:
                        probe_tensors.append(
                            self.signals.gather(self.signals[self.model.sig[p]["in"]])
                        )

                    logger.debug("=" * 30)
                    logger.debug("build_step complete")
                    logger.debug("probe_tensors %s", [str(x) for x in probe_tensors])
                    logger.debug("side_effects %s", [str(x) for x in side_effects])

                # update probe data
                update_probes(probe_tensors, loop_i)

                # need to make sure that any operators that could have side
                # effects run each timestep, so we tie them to the loop
                # increment. we also need to make sure that all the probe
                # reads happen before those values get overwritten on the
                # next timestep
                with tf.control_dependencies(side_effects + probe_tensors):
                    loop_i += 1

        return loop_i

    @trackable.no_automatic_dependency_tracking
    def build_post(self):
        """
        Executes post-build processes for operators (after the graph has
        been constructed and whenever Simulator is reset).
        """

        rng = np.random.RandomState(self.seed)

        # build input functions (we need to do this here, because in the case
        # of processes these functions need to be be rebuilt on reset)
        self.input_funcs = {}
        for n, output in self.invariant_inputs.items():
            if isinstance(output, np.ndarray):
                self.input_funcs[n] = output
            elif isinstance(output, Process):
                state = output.make_state((n.size_in,), (n.size_out,), self.dt)
                self.input_funcs[n] = [
                    output.make_step(
                        (n.size_in,),
                        (n.size_out,),
                        self.dt,
                        output.get_rng(rng),
                        state,
                    )
                    for _ in range(self.minibatch_size)
                ]
            elif n.size_out > 0:
                self.input_funcs[n] = [utils.align_func(self.dtype)(output)]
            else:
                # a node with no inputs and no outputs, but it can still
                # have side effects
                self.input_funcs[n] = [output]

        # execute build_post on all the op builders
        self.op_builder.build_post(self.signals)

    def get_tensor(self, sig):
        """
        Returns a Tensor corresponding to the given Signal.

        Parameters
        ----------
        sig : `~nengo.builder.Signal`
            A signal in the Nengo model.

        Returns
        -------
        tensor : ``tf.Tensor``
            Tensor containing the value of the given Signal.
        """

        tensor_sig = self.signals[sig]

        try:
            base = self.base_params[tensor_sig.key]
        except KeyError:
            base = self.saved_state[tensor_sig.key]

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
        ``net.config[nengo.Ensemble].trainable = False``).

        The trainable attribute will be set to one of three values:

        - ``True``: Signal is trainable
        - ``False``: Signal could be trainable, but has been set to non-trainable
          (e.g., because the user manually configured that object not to be trainable).
        - ``None``: Signal is never trainable (e.g., simulator state)
        """

        def get_trainable(parent_configs, obj):
            """Looks up the current value of ``obj.trainable``."""

            if self.inference_only:
                return False

            # default to 1 (so that we can distinguish between an object being
            # set to trainable vs defaulting to trainable)
            trainable = 1

            # we go from top down (so lower level settings will override)
            for cfg in parent_configs:
                try:
                    cfg_trainable = getattr(cfg[obj], "trainable", None)
                except ConfigError:
                    # object not configured in this network config
                    cfg_trainable = None

                if cfg_trainable is not None:
                    trainable = cfg_trainable

            return trainable

        def mark_network(parent_configs, net):
            """Recursively marks the signals for objects within each subnetwork."""

            parent_configs = parent_configs + [net.config]

            for subnet in net.networks:
                mark_network(parent_configs, subnet)

            # encoders and biases are trainable
            for ens in net.ensembles:
                ens_trainable = get_trainable(parent_configs, ens)

                self.model.sig[ens]["encoders"].trainable = ens_trainable
                self.model.sig[ens]["encoders"].minibatched = False

                if not isinstance(ens.neuron_type, Direct):
                    neurons_trainable = get_trainable(parent_configs, ens.neurons)
                    if neurons_trainable and type(neurons_trainable) == int:
                        # neurons_trainable is 1, so default to trainability of parent
                        neurons_trainable = ens_trainable

                    self.model.sig[ens.neurons]["bias"].trainable = neurons_trainable
                    self.model.sig[ens.neurons]["bias"].minibatched = False

            # connection weights are trainable
            for conn in net.connections:
                # note: this doesn't include probe connections, since they
                # aren't added to the network
                if compat.conn_has_weights(conn):
                    self.model.sig[conn]["weights"].trainable = get_trainable(
                        parent_configs, conn
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
                                f"{obj} has a learning rule and is also set to be "
                                f"trainable; this is likely to produce strange "
                                f"training behaviour."
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
            mark_network([], self.model.toplevel)

            # the connections to connection probes are not trainable, but
            # also not minibatched
            probe_seeds = [self.model.seeds[p] for p in self.model.probes]
            for obj, seed in self.model.seeds.items():
                if isinstance(obj, Connection) and seed in probe_seeds:
                    if compat.conn_has_weights(obj):
                        self.model.sig[obj]["weights"].trainable = None
                        self.model.sig[obj]["weights"].minibatched = False

        # time/step are not minibatched and not trainable
        self.model.step.trainable = None
        self.model.step.minibatched = False
        self.model.time.trainable = None
        self.model.time.minibatched = False

        # fill in defaults for all other signals
        # signals are not trainable by default, and views take on the
        # properties of their bases
        all_sigs = [sig for op in self.model.operators for sig in op.all_signals]
        for sig in all_sigs:
            if not hasattr(sig.base, "trainable"):
                sig.base.trainable = None

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

        base_arrays = {"trainable": {}, "non_trainable": {}, "state": {}}
        curr_keys = {}

        sig_idxs = {s: i for i, s in enumerate(sigs)}

        # find the non-overlapping partitions of the signals
        breaks = []
        diff = defaultdict(int)
        for ops in self.plan:
            if isinstance(ops[0], Reset):
                # don't include Resets, otherwise the big reset block
                # overrides most of the partitioning
                partition_sigs = []
            else:
                partition_sigs = range(len(ops[0].all_signals))

            for i in partition_sigs:
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

        # find all the signals that have a set operation associated with them

        def special_set(s, op):
            return (
                # we don't include Lowpass ops, because for efficiency reasons in the
                # nengo-dl Lowpass implementation we reuse the output signal (which is
                # set) as the state signal (so we need to include that signal in the
                # state)
                (isinstance(op, SimProcess) and isinstance(op.process, Lowpass))
                # nengo marks the time step as a set, but really it's an inc (since
                # it's incrementing the simulation step)
                or (isinstance(op, TimeUpdate) and s is op.step)
                # nengo marks neuron state as a set, but really it's more like an
                # inc/update (since the neuron calculation may depend on the state)
                or (
                    isinstance(op, SimNeurons) and s in compat.neuron_state(op).values()
                )
            )

        set_sigs = {
            s.base
            for ops in self.plan
            for op in ops
            for s in op.sets
            if not special_set(s, op)
        }

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
                dtype = self.dtype
            elif np.issubdtype(sig.dtype, np.integer):
                dtype = "int32"
            elif np.issubdtype(sig.dtype, np.bool_):
                dtype = "bool"
            else:
                raise NotImplementedError("Unsupported signal dtype")

            if sig.sparse:
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

            if sig in set_sigs:
                # signals with a set operation associated with them don't need an
                # initial value (since the value will just be immediately overridden
                # by the set operation)
                initial_value = None
            else:
                initial_value = sig.initial_value
                if sig.sparse:
                    if isinstance(initial_value, SparseMatrix):
                        initial_value = initial_value.data
                    else:
                        initial_value = initial_value.tocoo().data

            if sig.minibatched:
                shape = (self.minibatch_size,) + shape

            if sig.trainable is None:
                sig_type = "state"
            elif sig.trainable:
                sig_type = "trainable"
            else:
                sig_type = "non_trainable"

            if key in base_arrays[sig_type]:
                base_arrays[sig_type][key][0].append(initial_value)
                base_arrays[sig_type][key][1].append(shape)
            else:
                logger.debug("starting new base signal %s", key)
                base_arrays[sig_type][key] = [
                    [initial_value],
                    [shape],
                    dtype,
                    sig.minibatched,
                ]

            n = sum(x[sig.minibatched] for x in base_arrays[sig_type][key][1])
            slices = [(n - shape[sig.minibatched], n)]

            tensor_sig = self.signals.get_tensor_signal(
                slices,
                key,
                dtype,
                shape[sig.minibatched :],
                sig.minibatched,
                label=sig.name,
                signal=sig,
            )

            logger.debug(sig)
            logger.debug(tensor_sig)

        # add any signal views to the sig_map
        all_views = compat.FrozenOrderedSet(
            sig
            for ops in self.plan
            for op in ops
            for sig in op.all_signals
            if sig.is_view
        )

        for sig in all_views:
            if sig.size == sig.base.size:
                # reshape view
                self.signals[sig] = self.signals[sig.base].reshape(sig.shape)
            else:
                # slice view

                if sig.shape[1:] != sig.base.shape[1:]:
                    # TODO: support this?
                    raise NotImplementedError("Slicing on axes > 0 is not supported")

                # compute slice along first dimension (dividing the element-wise strides
                # by the size of each row)
                row_size = np.prod(sig.shape[1:], dtype=np.int32)
                start = sig.elemoffset // row_size
                stride = sig.elemstrides[0] // row_size
                stop = start + sig.shape[0] * stride
                if stop < 0:
                    stop = None

                self.signals[sig] = self.signals[sig.base][slice(start, stop, stride)]

        self.base_arrays_init = base_arrays
