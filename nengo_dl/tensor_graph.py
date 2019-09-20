"""
Manages all the data and build processes associated with the TensorFlow graph.

The TensorFlow graph is the symbolic description of the computations in the
network, which will be executed by the simulator.
"""

from collections import OrderedDict, defaultdict
import logging
import warnings

from nengo import Connection, Process
from nengo.builder.operator import SimPyFunc, Reset
from nengo.builder.processes import SimProcess
from nengo.config import ConfigError
from nengo.neurons import Direct
import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable

from nengo_dl import builder, graph_optimizer, signals, utils, tensor_node, config
from nengo_dl.compat import (
    SparseMatrix,
    is_sparse,
    make_process_state,
    make_process_step,
)

logger = logging.getLogger(__name__)


class TensorGraph(tf.keras.layers.Layer):
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
    seed : int
        Seed for random number generation.
    """

    @trackable.no_automatic_dependency_tracking
    def __init__(
        self,
        model,
        dt,
        unroll_simulation,
        dtype,
        minibatch_size,
        device,
        progress,
        seed,
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
        self.seed = seed
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
                name += "_%d" % name_count[key]

            self.io_names[obj] = name
            name_count[key] += 1

        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info(
            "Number of base arrays: %d, %d",
            *tuple(len(x) for x in self.base_arrays_init),
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
        inputs = OrderedDict()
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
        """

        super().build(input_shape)

        # variables for model parameters
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.base_params = OrderedDict()
        assert len(self.base_params) == 0
        for k, v in self.base_arrays_init[True].items():
            self.base_params[k] = self.add_weight(
                initializer=tf.initializers.constant(v),
                shape=v.shape,
                dtype=v.dtype,
                trainable=True,
                name="base_params/%s_%s" % (v.dtype, "_".join(str(x) for x in v.shape)),
            )

        logger.debug("created base param variables")
        logger.debug([str(x) for x in self.base_params.values()])

        # variables to save the internal state of simulation between runs
        # note: place these on CPU because they'll only be accessed once at the
        # beginning of the simulation loop, and they can be quite large
        with trackable.no_automatic_dependency_tracking_scope(self):
            self.saved_state = OrderedDict()
        with tf.device("/cpu:0"):
            for k, v in self.base_arrays_init[False].items():
                self.saved_state[k] = self.add_weight(
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
        logger.debug([str(x) for x in self.saved_state.values()])

        # call build on any TensorNode Layers
        for ops in self.plan:
            if isinstance(ops[0], tensor_node.SimTensorNode):
                for op in ops:
                    if isinstance(op.func, tf.keras.layers.Layer):
                        if op.time is None:
                            shape_in = []
                        else:
                            shape_in = [()]
                        if op.input is not None:
                            shape_in += [(self.minibatch_size,) + op.shape_in]
                        if len(shape_in) == 1:
                            shape_in = shape_in[0]

                        op.func.build(shape_in)

                        # add op func to _layers so that any weights are collected
                        self._layers.append(op.func)

    # @tf.function  # TODO: get this working? does this help?
    @tf.autograph.experimental.do_not_convert  # TODO: enable autograph?
    @trackable.no_automatic_dependency_tracking
    def call(self, inputs, training=None, progress=None, stateful=False):
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
        self.node_inputs = {}
        for n, inp in zip(self.invariant_inputs, inputs):
            # specify shape of inputs (keras sometimes loses this shape information)
            inp.set_shape([self.minibatch_size, inp.shape[1], n.size_out])

            self.node_inputs[n] = inp

        self.steps_to_run = inputs[-1][0, 0]

        # initialize op builder
        build_config = builder.BuildConfig(
            inference_only=self.inference_only,
            lif_smoothing=config.get_setting(self.model, "lif_smoothing"),
            cpu_only=self.device == "/cpu:0" or not utils.tf_gpu_installed,
            rng=np.random.RandomState(self.seed),
            training=(
                tf.keras.backend.learning_phase() if training is None else training
            ),
        )
        self.op_builder = builder.Builder(self.plan, self.signals, build_config)

        # pre-build stage
        with progress.sub("pre-build stage", max_value=len(self.plan)) as sub:
            self.op_builder.build_pre(sub)

        # build stage
        with progress.sub("build stage", max_value=len(self.plan) * self.unroll) as sub:
            self._build_loop(sub)

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
        outputs = list(self.probe_arrays.values()) + [self.steps_run]

        if stateful:
            # update saved state
            state_updates = [
                var.assign(val)
                for var, val in zip(
                    self.saved_state.values(), self.final_internal_state
                )
            ]
            with tf.control_dependencies(state_updates), tf.name_scope("save"):
                outputs = [tf.identity(x) for x in outputs]

        return outputs

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

        def loop_body(loop_i, n_steps, probe_arrays, base_params, saved_state):
            # fill in signals.bases (note: we need to do this here because we
            # need to use the tensors from inside the loop, not the source variables)
            # TODO: test that rebuilding multiple times works properly
            for key, val in zip(self.saved_state.keys(), saved_state):
                # eager while loops pass in the variable directly,
                # so we add the tf.identity so that when we write we're not updating
                # the saved state
                self.signals.bases[key] = tf.identity(val)
            for key, val in zip(self.base_params.keys(), base_params):
                # we never write to base_params, so don't need identity
                self.signals.bases[key] = val

            for iter in range(self.unroll):
                logger.debug("BUILDING ITERATION %d", iter)
                with tf.name_scope("iteration_%d" % iter):
                    # fill in invariant input data
                    for n in self.node_inputs:
                        if self.model.sig[n]["out"] in self.signals:
                            # if the out signal doesn't exist then that means that
                            # the node output isn't actually used anywhere, so we can
                            # ignore it

                            self.signals.scatter(
                                self.signals[self.model.sig[n]["out"]],
                                self.node_inputs[n][:, loop_i],
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
                                true_fn=lambda p=p, i=i: probe_arrays[i].write(0, p),
                                false_fn=lambda i=i: probe_arrays[i],
                            )

                    # need to make sure that any operators that could have side
                    # effects run each timestep, so we tie them to the loop
                    # increment. we also need to make sure that all the probe
                    # reads happen before those values get overwritten on the
                    # next timestep
                    with tf.control_dependencies(side_effects + probe_tensors):
                        loop_i += 1

            base_arrays = tuple(self.signals.bases[key] for key in self.base_params)
            state_arrays = tuple(self.signals.bases[key] for key in self.saved_state)

            return loop_i, n_steps, probe_arrays, base_arrays, state_arrays

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
            tuple(self.base_params.values()),
            tuple(self.saved_state.values()),
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
                x = tf.transpose(x, perm=perm)
            else:
                # add minibatch dimension for consistency
                x = tf.expand_dims(x, 0)

            self.probe_arrays[p] = x

        self.final_internal_state = loop_vars[4]

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
                state = make_process_state(output, (n.size_in,), (n.size_out,), self.dt)
                self.input_funcs[n] = [
                    make_process_step(
                        output,
                        (n.size_in,),
                        (n.size_out,),
                        self.dt,
                        output.get_rng(rng),
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
            base = self.base_params[tensor_sig.key]
        except KeyError:
            base = self.saved_state[tensor_sig.key]

        return tf.gather(
            base,
            tf.constant(tensor_sig.indices),
            axis=1 if tensor_sig.minibatched else 0,
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
