from __future__ import print_function, division

from collections import OrderedDict
import logging
import warnings

from nengo import Connection, Process, Ensemble
from nengo.builder.operator import TimeUpdate, SimPyFunc
from nengo.builder.processes import SimProcess
from nengo.config import ConfigError
from nengo.ensemble import Neurons
from nengo.exceptions import SimulationError
from nengo.neurons import Direct
import numpy as np
import tensorflow as tf

from nengo_dl import builder, graph_optimizer, signals, utils, tensor_node

if tf.__version__ < "1.4.0":
    from tensorflow.contrib.data import Dataset, Iterator
else:
    from tensorflow.python.data import Dataset, Iterator

logger = logging.getLogger(__name__)


def with_self(func):
    """A decorator that can be used to ensure that any ops created within the
    wrapped method will be added to the TensorGraph object's graph."""

    def func_with_self(self, *args, **kwargs):
        with self.graph.as_default(), tf.device(self.device):
            return func(self, *args, **kwargs)

    return func_with_self


class TensorGraph(object):
    """Manages the construction of the TensorFlow symbolic computation graph.

    Parameters
    ----------
    model : :class:`~nengo:nengo.builder.Model`
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
    """

    def __init__(self, model, dt, unroll_simulation, dtype,
                 minibatch_size, device):
        self.model = model
        self.dt = dt
        self.unroll = unroll_simulation
        self.dtype = dtype
        self.minibatch_size = minibatch_size
        self.device = device
        self.graph = tf.Graph()

        # find invariant inputs (nodes that don't receive any input other
        # than the simulation time). we'll compute these outside the simulation
        # and feed in the result.
        if self.model.toplevel is None:
            self.invariant_inputs = []
        else:
            self.invariant_inputs = [n for n in self.model.toplevel.all_nodes
                                     if n.size_in == 0 and
                                     not isinstance(n, tensor_node.TensorNode)]

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
        old_operators = []
        while len(old_operators) != len(operators):
            old_operators = operators
            operators = graph_optimizer.remove_constant_copies(operators)
            operators = graph_optimizer.remove_unmodified_resets(operators)
            operators = graph_optimizer.remove_zero_incs(operators)
            operators = graph_optimizer.remove_identity_muls(operators)

        # group mergeable operators
        try:
            planner = model.toplevel.config[model.toplevel].planner
        except (ConfigError, AttributeError):
            planner = graph_optimizer.tree_planner
        plan = planner(operators)

        # TODO: we could also merge operators sequentially (e.g., combine
        # a copy and dotinc into one op), as long as the intermediate signal
        # is only written to by one op and read by one op

        # order signals/operators to promote contiguous reads
        sigs, self.plan = graph_optimizer.order_signals(plan, n_passes=10)

        # create base arrays and map Signals to TensorSignals (views on those
        # base arrays)
        self.base_arrays_init, self.sig_map = graph_optimizer.create_signals(
            sigs, self.plan, float_type=dtype.as_numpy_dtype,
            minibatch_size=self.minibatch_size)

        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info("Number of base arrays: %d", len(self.base_arrays_init))

    @with_self
    def build(self):
        """Constructs a new graph to simulate the model."""

        self.signals = signals.SignalDict(self.sig_map, self.dtype,
                                          self.minibatch_size)
        self.target_phs = {}
        self.losses = {}
        self.optimizers = {}

        # make sure indices are loaded for all probe signals (they won't
        # have been loaded if this signal is only accessed as part of a
        # larger block during the simulation)
        for p in self.model.probes:
            probe_sig = self.model.sig[p]["in"]
            if probe_sig in self.sig_map:
                self.sig_map[probe_sig].load_indices()

        # create this constant once here so we don't end up creating a new
        # dt constant in each operator
        self.signals.dt = tf.constant(self.dt, self.dtype)
        self.signals.dt_val = self.dt  # store the actual value as well

        # variable to track training step
        with tf.device("/cpu:0"):
            with tf.variable_scope("misc_vars", reuse=False):
                self.training_step = tf.get_variable(
                    "training_step", initializer=tf.constant_initializer(0),
                    dtype=tf.int64, shape=(), trainable=False)
            self.training_step_inc = tf.assign_add(self.training_step, 1)

        # create base arrays
        self.base_vars = []
        for k, (v, trainable) in self.base_arrays_init.items():
            unique_idx = 0
            duplicate = True
            while duplicate:
                name = "%s_%s_%s_%s" % (
                    v.dtype, "_".join(str(x) for x in v.shape), trainable,
                    unique_idx)

                if any([name in x.name for x in (
                        tf.trainable_variables() if trainable else
                        tf.local_variables())]):
                    unique_idx += 1
                else:
                    duplicate = False

            if trainable:
                with tf.variable_scope("trainable_vars", reuse=False):
                    var = tf.get_variable(
                        name, initializer=tf.constant_initializer(v),
                        dtype=v.dtype, shape=v.shape, trainable=True)
            else:
                with tf.variable_scope("local_vars", reuse=False):
                    var = tf.get_local_variable(
                        name, initializer=tf.constant_initializer(v),
                        dtype=v.dtype, shape=v.shape, trainable=False)

            self.base_vars += [var]

        logger.debug("created base arrays")
        logger.debug([str(x) for x in self.base_vars])

        # set up inputs
        # TODO: check if this cpu assignment hurts performance
        with tf.device("/cpu:0"):
            self.build_inputs()

        # pre-build stage
        self.op_builds = {}
        for ops in self.plan:
            with self.graph.name_scope(utils.sanitize_name(
                    builder.Builder.builders[type(ops[0])].__name__)):
                builder.Builder.pre_build(ops, self.signals,
                                          self.op_builds)

        # build stage
        self.build_loop()

        # ops for initializing variables (will be called by simulator)
        trainable_vars = tf.trainable_variables() + [self.training_step]
        self.trainable_init_op = tf.variables_initializer(trainable_vars)
        self.local_init_op = tf.local_variables_initializer()
        self.global_init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v not in trainable_vars])

    def build_step(self):
        """Build the operators that execute a single simulation timestep
        into the graph.

        Returns
        -------
        probe_tensors : list of ``tf.Tensor``
            The Tensor objects representing the data required for each model
            Probe
        side_effects : list of ``tf.Tensor``
            The output Tensors of computations that may have side-effects
            (e.g., :class:`~nengo:nengo.Node` functions), meaning that they
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
            if probe_sig in self.sig_map:
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
                    self.sig_map[probe_sig], force_copy=True))
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

    def build_loop(self):
        """Build simulation loop.

        Loop can be constructed using the ``tf.while_loop`` architecture, or
        explicitly unrolled.  Unrolling increases graph construction time
        and memory usage, but increases simulation speed.
        """

        def loop_condition(step, stop, *_):
            return step < stop

        def loop_body(step, stop, loop_i, probe_arrays, base_vars):
            self.signals.bases = OrderedDict(
                [(k, v) for k, v in zip(self.base_arrays_init.keys(),
                                        base_vars)])

            for iter in range(self.unroll):
                logger.debug("BUILDING ITERATION %d", iter)
                with self.graph.name_scope("iteration_%d" % iter):
                    # note: nengo step counter is incremented at the beginning
                    # of the timestep
                    step += 1
                    self.signals.step = step

                    # fill in invariant input data
                    if self.data_phs is not None:
                        for n in self.invariant_inputs:
                            data = self.data_iter_next[n][loop_i]
                            data.set_shape((n.size_out, self.minibatch_size))
                            self.signals.scatter(
                                self.sig_map[self.model.sig[n]["out"]],
                                data)

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
                        with tf.variable_scope("", reuse=iter > 0):
                            probe_tensors, side_effects = self.build_step()

                    # copy probe data to array
                    for i, p in enumerate(probe_tensors):
                        probe_arrays[i] = probe_arrays[i].write(loop_i, p)

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
                self.signals.dtype, clear_after_read=True, size=0,
                dynamic_size=True)
            for _ in self.model.probes]

        # build simulation loop
        loop_vars = (
            self.step_var, self.stop_var, loop_i, probe_arrays,
            tuple(x._ref() if isinstance(x, tf.Variable) else x
                  for x in self.base_vars))

        # TODO: add option to disable backprop through loop, for when users
        # want to train a network running over time, but optimize on a
        # timestep-by-timestep basis
        loop_vars = tf.while_loop(
            loop_condition, loop_body, loop_vars=loop_vars,
            parallel_iterations=1, back_prop=True)

        self.steps_run = loop_vars[2]
        self.probe_arrays = []
        for p in loop_vars[3]:
            x = p.stack()
            self.probe_arrays += [x]

    def build_inputs(self):
        """Sets up the inputs in the model (which will be computed outside of
        TensorFlow and fed in each simulation block).
        """

        data_phs = {}
        for n in self.invariant_inputs:
            if self.model.sig[n]["out"] in self.sig_map:
                # make sure the indices for this input are loaded into
                # TensorFlow (they may not be, if the output of this node is
                # only read as part of a larger block during the simulation)
                self.sig_map[self.model.sig[n]["out"]].load_indices()

                # set up a placeholder input for this node
                data_phs[n] = tf.placeholder(
                    self.dtype, (None, None, n.size_out))

        if len(data_phs) > 0:
            tmp = tf.shape(next(iter(data_phs.values())))
            batch_size = tmp[0]
            sig_len = tmp[1]

            for p in self.model.probes:
                data_phs[p] = tf.placeholder_with_default(
                    tf.zeros((batch_size, sig_len, p.size_in),
                             dtype=self.dtype),
                    (None, None, p.size_in),
                    name=utils.sanitize_name(str(p)) + "_ph")

            # build output iterators
            self.datasets = {}

            # datasets can only work with string keys
            id_map = {str(id(x)): x for x in data_phs}
            data_ph_ids = {str(id(k)): v for k, v in data_phs.items()}

            # normal sim.run inputs
            data = Dataset.from_tensors(data_ph_ids)
            pad_len = tf.cast(tf.ceil(sig_len / self.unroll),
                              tf.int32) * self.unroll - sig_len
            data = data.map(
                lambda x: {k: tf.pad(v, ((0, 0), (0, pad_len), (0, 0)))
                           for k, v in x.items()})
            data = data.repeat(1)
            data = data.map(lambda x: {k: tf.transpose(v, (1, 2, 0))
                                       for k, v in x.items()})
            self.datasets["run"] = data

            # sim.train inputs
            # TODO: do we need the sig_len padding for these?
            self.epoch_ph = tf.placeholder(tf.int64, ())
            data = Dataset.from_tensor_slices(data_ph_ids)
            data = data.shuffle(tf.cast(tf.where(batch_size < 32, batch_size,
                                                 batch_size // 4),
                                        tf.int64))
            data = data.repeat(self.epoch_ph)
            data = data.batch(self.minibatch_size)
            data = data.filter(lambda x: any(
                tf.shape(v)[0] != self.minibatch_size for v in x.values()))
            data = data.map(lambda x: {k: tf.transpose(v, (1, 2, 0))
                                       for k, v in x.items()})
            self.datasets["train"] = data

            data = Dataset.from_tensor_slices(data_ph_ids)
            data = data.repeat(self.epoch_ph)
            data = data.batch(self.minibatch_size)
            data = data.filter(lambda x: any(
                tf.shape(v)[0] != self.minibatch_size for v in x.values()))
            data = data.map(lambda x: {k: tf.transpose(v, (1, 2, 0))
                                       for k, v in x.items()})
            self.datasets["train_no_shuffle"] = data

            # create iterators
            self.data_phs = data_phs
            self.data_iter = Iterator.from_structure(
                self.datasets["run"].output_types,
                self.datasets["run"].output_shapes)
            self.data_iter_next = {id_map[k]: v for k, v in
                                   self.data_iter.get_next().items()}
        else:
            self.data_phs = None

    @with_self
    def build_optimizer(self, optimizer, objective):
        """Adds elements into the graph to execute the given optimizer.

        Parameters
        ----------
        optimizer : ``tf.train.Optimizer``
            Instance of a TensorFlow optimizer class
        objective : dict of {:class:`~nengo:nengo.Probe`: ``"mse"`` or \
                                                          callable}
            The objective to be minimized for each probe. Passing
            ``"mse"`` will train with mean squared error. A custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be averaged across Probes).

        Returns
        -------
        ``tf.Tensor``
            Operator implementing the given optimizer update
        """

        loss = self.build_loss(objective)

        key = (optimizer, frozenset(objective.items()))

        try:
            # return the cached optimizer if it exists
            return self.optimizers[key]
        except KeyError:
            pass

        with tf.variable_scope(optimizer.get_name()) as scope:
            # create optimizer operator
            opt_op = optimizer.minimize(
                loss, var_list=tf.trainable_variables())

            # get any new variables created by the optimizer (so they
            # can be initialized)
            opt_slots_init = tf.variables_initializer(
                scope.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self.optimizers[key] = (opt_op, opt_slots_init)

        return self.optimizers[key]

    @with_self
    def build_loss(self, objective):
        """Adds elements into the graph to compute the given objective.

        Parameters
        ----------
        objective : dict of {:class:`~nengo:nengo.Probe`: ``"mse"`` or \
                                                          callable}
            The objective used to compute loss for each probe. Passing
            ``"mse"`` will use mean squared error. A custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be summed across Probes).

        Returns
        -------
        ``tf.Tensor``
            Tensor representing the sum of the given objectives applied to
            target probes
        """

        key = frozenset(objective.items())

        try:
            # return the cached loss tensor if it exists
            return self.losses[key]
        except KeyError:
            pass

        loss = []
        for p, obj in objective.items():
            probe_index = self.model.probes.index(p)
            target = self.data_iter_next[p]

            # compute loss
            if obj == "mse":
                # note: nan targets converted to zero error
                target = tf.where(tf.is_nan(target),
                                  self.probe_arrays[probe_index],
                                  target)

                loss += [tf.reduce_mean(
                    tf.square(target - self.probe_arrays[probe_index]))]
            elif callable(obj):
                # move minibatch dimension back to the front
                x = tf.transpose(self.probe_arrays[probe_index], (2, 0, 1))
                t = tf.transpose(target, (2, 0, 1))

                # note: control dependencies is here so that the target
                # iterator always gets incremented even if it isn't used in
                # the loss function (so that it terminates)
                with tf.control_dependencies([t]):
                    loss += [obj(x, t)]
            else:
                raise NotImplementedError

        # sum loss across probes (note: this will also sum across
        # the output of `objective` if it doesn't return a scalar)
        loss = tf.reduce_sum(loss)

        self.losses[key] = loss

        return loss

    @with_self
    def build_post(self, sess, rng):
        """Executes post-build processes for operators (after the graph has
        been constructed and session/variables initialized).

        Note that unlike other build functions, this is called every time
        the simulator is reset.

        Parameters
        ----------
        sess : ``tf.Session``
            The TensorFlow session for the simulator
        rng : :class:`~numpy:numpy.random.RandomState`
            Seeded random number generator
        """

        for ops, built_ops in self.op_builds.items():
            built_ops.build_post(ops, self.signals, sess, rng)

    @with_self
    def build_summaries(self, summaries):
        """Adds ops to collect summary data for the given objects.

        Parameters
        ----------
        summaries : list of tuple or \
                            :class:`~nengo:nengo.Connection` or \
                            :class:`~nengo:nengo.Ensemble` or \
                            :class:`~nengo:nengo.ensemble.Neurons` or \
                            ``tf.Tensor``}
            List of objects for which we want to collect data.  Object can be a
            Connection (in which case data on weights will be collected),
            Ensemble (encoders), Neurons (biases), a tuple of
            ``(objective, probes)`` that indicates a loss function that will
            be tracked, or a pre-built summary tensor.

        Returns
        -------
        ``tf.Tensor``
            Merged summary op for the given summaries
        """

        summary_ops = []
        with tf.device("/cpu:0"):
            for obj in summaries:
                if isinstance(obj, dict):
                    # overall loss
                    loss = self.build_loss(obj)
                    summary_ops.append(tf.summary.scalar(
                        "loss", loss, family="loss"))

                    if len(obj) > 1:
                        # get loss for each probe
                        inputs = tf.unstack(loss.op.inputs[0])
                        for p, t in zip(obj, inputs):
                            summary_ops.append(tf.summary.scalar(
                                utils.sanitize_name("Probe_%s_loss" % p.label),
                                t, family="loss"))
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

            return tf.summary.merge(summary_ops)

    @with_self
    def get_tensor(self, sig):
        """Returns a Tensor corresponding to the given Signal.

        Parameters
        ----------
        sig : :class:`~nengo:nengo.builder.Signal`
            A signal in the model

        Returns
        -------
        ``tf.Tensor``
            Tensor containing the value of the given Signal
        """

        tensor_sig = self.sig_map[sig]
        keys = list(self.signals.bases.keys())

        if tensor_sig.tf_indices is None:
            tensor_sig.load_indices()

        base = self.base_vars[keys.index(tensor_sig.key)]
        return tf.gather(base, tensor_sig.tf_indices)

    def mark_signals(self):
        """Mark all the signals in ``self.model`` according to whether they
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

        def get_trainable(config, obj, network_trainable):
            """Looks up the current value of ``obj.trainable``."""

            try:
                if obj in config.params:
                    # priority #1: instance config
                    trainable = config[obj].trainable
                elif network_trainable is not 1:
                    # priority #2: network setting
                    trainable = network_trainable
                else:
                    # priority #3: class config
                    trainable = config[obj].trainable
            except (ConfigError, AttributeError):
                trainable = network_trainable

            # we return 1 if trainable isn't configured, since the default is
            # for everything to be trainable but we want to be able to
            # distinguish whether something was specifically set to be
            # trainable (True) or just defaulting to trainable (1)
            return 1 if trainable is None else trainable

        def mark_network(config, net, network_trainable):
            """Recursively marks the signals for objects within each
            subnetwork."""

            for subnet in net.networks:
                mark_network(config, subnet,
                             get_trainable(config, subnet, network_trainable))

            # encoders and biases are trainable
            for ens in net.ensembles:
                ens_trainable = get_trainable(config, ens, network_trainable)

                self.model.sig[ens]["encoders"].trainable = ens_trainable
                self.model.sig[ens]["encoders"].minibatched = False

                if not isinstance(ens.neuron_type, Direct):
                    neurons_trainable = get_trainable(config, ens.neurons,
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
                    config, conn, network_trainable)
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
            config = self.model.toplevel.config
            mark_network(config, self.model.toplevel,
                         get_trainable(config, self.model.toplevel, 1))

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
