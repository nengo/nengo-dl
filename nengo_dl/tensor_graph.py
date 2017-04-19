from collections import OrderedDict
import datetime
import logging
import time

from nengo import Process
from nengo.builder.operator import TimeUpdate, SimPyFunc
from nengo.builder.processes import SimProcess
from nengo.exceptions import SimulationError
import tensorflow as tf

from nengo_dl import builder, graph_optimizer, signals, utils, tensor_node

logger = logging.getLogger(__name__)


class TensorGraph(object):
    """Manages the construction of the TensorFlow symbolic computation graph.

    Parameters
    ----------
    model : :class:`~nengo:nengo.builder.Model`
        pre-built Nengo model describing the network to be simulated
    dt : float
        length of a simulator timestep, in seconds
    step_blocks : int
        controls how many simulation steps run each time the graph is
        executed (affects memory usage and graph construction time)
    unroll_simulation : bool
        if True, unroll simulation loop by explicitly building each iteration
        (up to ``step_blocks``) into the computation graph. if False, use a
        symbolic loop, which is more general and produces a simpler graph, but
        is likely to be slower to simulate
    dtype : ``tf.DType``
        floating point precision to use for simulation
    minibatch_size : int
        the number of simultaneous inputs that will be passed through the
        network
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        device on which to execute computations (if None then uses the
        default device as determined by Tensorflow)
    """

    def __init__(self, model, dt, step_blocks, unroll_simulation, dtype,
                 minibatch_size, device):
        self.model = model
        self.dt = dt
        self.step_blocks = step_blocks
        self.unroll_simulation = unroll_simulation
        self.dtype = dtype
        self.minibatch_size = minibatch_size
        self.device = device

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

        logger.info("Initial plan length: %d", len(operators))

        utils.print_and_flush("Optimizing graph", end="")
        start = time.time()

        # group mergeable operators
        plan = graph_optimizer.greedy_planner(operators)
        # plan = graph_optimizer.tree_planner(operators)
        # plan = graph_optimizer.noop_planner(operators)

        # order signals/operators to promote contiguous reads
        sigs, self.plan = graph_optimizer.order_signals(plan, n_passes=10)
        # sigs, self.plan = graph_optimizer.noop_order_signals(plan)

        # create base arrays and map Signals to TensorSignals (views on those
        # base arrays)
        self.base_arrays_init, self.sig_map = graph_optimizer.create_signals(
            sigs, self.plan, float_type=dtype.as_numpy_dtype,
            minibatch_size=self.minibatch_size)

        print("\rOptimization completed in %s " %
              datetime.timedelta(seconds=int(time.time() - start)))

        logger.info("Optimized plan length: %d", len(self.plan))
        logger.info("Number of base arrays: %d", len(self.base_arrays_init))

    def build(self, rng):
        """Constructs a new graph to simulate the model.

        Parameters
        ----------
        rng : :class:`~numpy:numpy.random.RandomState`
            the Simulator's random number generator
        """

        self.graph = tf.Graph()
        self.signals = signals.SignalDict(self.sig_map, self.dtype,
                                          self.minibatch_size)
        self.target_phs = {}
        self.losses = {}
        self.optimizers = {}

        with self.graph.as_default(), tf.device(self.device):
            # make sure indices are loaded for all probe signals (they won't
            # have been loaded if this signal is only accessed as part of a
            # larger block during the simulation)
            for p in self.model.probes:
                self.sig_map[self.model.sig[p]["in"]].load_indices()

            # create this constant once here so we don't end up creating a new
            # dt constant in each operator
            self.signals.dt = tf.constant(self.dt, self.dtype)
            self.signals.dt_val = self.dt  # store the actual value as well

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
                            tf.global_variables() if trainable else
                            tf.local_variables())]):
                        unique_idx += 1
                    else:
                        try:
                            self.graph.get_tensor_by_name(name + ":0")
                            unique_idx += 1
                        except KeyError:
                            duplicate = False

                # if trainable:
                #     # trainable signal, so create Variable
                #     with tf.variable_scope("base_vars", reuse=False):
                #         var = tf.get_variable(
                #             name, initializer=tf.constant_initializer(v),
                #             dtype=v.dtype, shape=v.shape, trainable=True)
                # else:
                #     var = tf.placeholder(tf.as_dtype(v.dtype), shape=v.shape,
                #                          name=name)
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

            # set up invariant inputs
            self.build_inputs(rng)

            # pre-build stage
            for ops in self.plan:
                with self.graph.name_scope(utils.sanitize_name(
                        builder.Builder.builders[type(ops[0])].__name__)):
                    builder.Builder.pre_build(ops, self.signals, rng)

            # build stage
            self.build_loop()

            # ops for initializing variables (will be called by simulator)
            self.trainable_init_op = tf.variables_initializer(
                tf.trainable_variables())
            self.local_init_op = tf.local_variables_initializer()
            # note: the only non-trainable global variables should be those
            # created inside TensorNodes
            self.global_init_op = tf.variables_initializer(
                [v for v in tf.global_variables()
                 if v not in tf.trainable_variables()])

    def build_step(self):
        """Build the operators that execute a single simulation timestep
        into the graph.

        Returns
        -------
        probe_tensors : list of ``tf.Tensor``
            the Tensor objects representing the data required for each model
            Probe
        side_effects : list of ``tf.Tensor``
            the output Tensors of computations that may have side-effects
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
                outputs = builder.Builder.build(ops, self.signals)

            if outputs is not None:
                side_effects += outputs

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
        # so by adding the copy here and then blocking on the copy, we make
        # sure that the probe value is read before it can be overwritten.
        logger.debug("collecting probe tensors")
        probe_tensors = [
            self.signals.gather(self.sig_map[self.model.sig[p]["in"]],
                                force_copy=True)
            for p in self.model.probes]

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

            # note: nengo step counter is incremented at the beginning of
            # the timestep
            step += 1
            self.signals.step = step

            # fill in invariant input data
            for n in self.invariant_ph:
                self.signals.scatter(
                    self.sig_map[self.model.sig[n]["out"]],
                    self.invariant_ph[n][loop_i])

            # build the operators for a single step
            # note: we tie things to the `loop_i` variable so that we can be
            # sure the other things we're tying to the simulation step (side
            # effects and probes) from the previous timestep are executed
            # before the next step starts
            with self.graph.control_dependencies([loop_i]):
                probe_tensors, side_effects = self.build_step()

            # copy probe data to array
            for i, p in enumerate(probe_tensors):
                probe_arrays[i] = probe_arrays[i].write(loop_i, p)

            # need to make sure that any operators that could have side
            # effects run each timestep, so we tie them to the loop increment.
            # we also need to make sure that all the probe reads happen before
            # those values get overwritten on the next timestep
            with self.graph.control_dependencies(side_effects + probe_tensors):
                loop_i += 1

            base_vars = tuple(self.signals.bases.values())

            return step, stop, loop_i, probe_arrays, base_vars

        self.step_var = tf.placeholder(tf.int32, shape=(), name="step")
        self.stop_var = tf.placeholder(tf.int32, shape=(), name="stop")
        loop_i = tf.constant(0)

        probe_arrays = [
            tf.TensorArray(
                self.signals.dtype, clear_after_read=False,
                size=0 if self.step_blocks is None else self.step_blocks,
                dynamic_size=self.step_blocks is None)
            for _ in self.model.probes]

        # build simulation loop
        loop_vars = (
            self.step_var, self.stop_var, loop_i, probe_arrays,
            tuple(x._ref() if isinstance(x, tf.Variable) else x
                  for x in self.base_vars))

        if self.unroll_simulation:
            for n in range(self.step_blocks):
                logger.debug("BUILDING ITERATION %d", n)
                with self.graph.name_scope("iteration_%d" % n):
                    loop_vars = loop_body(*loop_vars)
        else:
            # TODO: get parallel iterations working? nengo simulations are
            # pretty serial though, so I'm not sure how much benefit we would
            # get (and it seems non-trivial to get working correctly)
            loop_vars = tf.while_loop(
                loop_condition, loop_body, loop_vars=loop_vars,
                parallel_iterations=1, back_prop=True)

        self.end_base_arrays = loop_vars[4]
        self.probe_arrays = []
        for p in loop_vars[3]:
            x = p.stack()
            if self.step_blocks is not None:
                x.set_shape([self.step_blocks] +
                            x.get_shape().as_list()[1:])
            self.probe_arrays += [x]

        # note: we need to make sure the final base array updates get computed,
        # even if they aren't being read by anything, because they may be
        # being read on the next `_run_steps` call. the `tf.while_loop`
        # enter/exit logic takes care of that on its own, so we only need to
        # do this for the unrolled case
        with tf.control_dependencies(self.end_base_arrays if
                                     self.unroll_simulation else []):
            self.steps_run = tf.identity(loop_vars[2])

    def build_inputs(self, rng):
        """Sets up the inputs in the model (which will be computed outside of
        Tensorflow and fed in each simulation block).

        Parameters
        ----------
        rng : :class:`~numpy:numpy.random.RandomState`
            the Simulator's random number generator
        """

        self.invariant_funcs = {}
        self.invariant_ph = {}
        for n in self.invariant_inputs:
            if self.model.sig[n]["out"] in self.sig_map:
                # make sure the indices for this input are loaded into
                # TensorFlow (they may not be, if the output of this node is
                # only read as part of a larger block during the simulation)
                self.sig_map[self.model.sig[n]["out"]].load_indices()

                # set up a placeholder input for this node
                self.invariant_ph[n] = tf.placeholder(
                    self.dtype, (self.step_blocks, n.size_out,
                                 self.minibatch_size))

            # build the node functions, which will be called offline to
            # generate the input values
            if isinstance(n.output, Process):
                self.invariant_funcs[n] = n.output.make_step(
                    (n.size_in,), (n.size_out,), self.dt,
                    n.output.get_rng(rng))
            elif n.size_out > 0:
                self.invariant_funcs[n] = utils.align_func(
                    (n.size_out,), self.dtype)(n.output)
            else:
                self.invariant_funcs[n] = n.output

    def build_optimizer(self, optimizer, targets, objective):
        """Adds elements into the graph to execute the given optimizer.

        Parameters
        ----------
        optimizer : ``tf.train.Optimizer``
            instance of a Tensorflow optimizer class
        targets : tuple of :class:`~nengo:nengo.Probe`
            the Probes corresponding to the output signals being optimized
        objective : ``"mse"`` or callable
            the objective to be minimized. passing ``"mse"`` will train with
            mean squared error. a custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be averaged across Probes).
        """

        with self.graph.as_default(), tf.device(self.device):
            loss = self.build_loss(objective, targets)

            key = (optimizer, targets, objective)
            if key not in self.optimizers:
                # create optimizer operator
                try:
                    opt_op = optimizer.minimize(
                        loss, var_list=tf.trainable_variables())
                except ValueError:
                    raise SimulationError(
                        "Network graph contains non-differentiable elements")

                # get any new variables created by optimizer (so they can be
                # initialized)
                opt_slots_init = tf.variables_initializer(
                    [optimizer.get_slot(v, name)
                     for v in tf.trainable_variables()
                     for name in optimizer.get_slot_names()])

                self.optimizers[key] = (opt_op, opt_slots_init)

            return self.optimizers[key]

    def build_loss(self, objective, targets):
        """Adds elements into the graph to compute the given objective.

        Parameters
        ----------
        objective : ``"mse"`` or callable
            the objective used to compute loss. passing ``"mse"`` will use
            mean squared error. a custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be averaged across Probes).
        targets : tuple of :class:`~nengo:nengo.Probe`
            the Probes corresponding to target values in objective
        """

        if isinstance(targets, list):
            targets = tuple(targets)

        if (objective, targets) in self.losses:
            return self.losses[(objective, targets)]

        with self.graph.as_default(), tf.device(self.device):
            loss = []
            for p in targets:
                probe_index = self.model.probes.index(p)

                # create a placeholder for the target values
                if p not in self.target_phs:
                    self.target_phs[p] = tf.placeholder(
                        self.dtype, (self.step_blocks, p.size_in,
                                     self.minibatch_size), name="targets")

                # compute loss
                if objective == "mse":
                    loss += [tf.reduce_mean(tf.square(
                        self.target_phs[p] - self.probe_arrays[probe_index]))]
                elif callable(objective):
                    # move minibatch dimension back to the front
                    x = tf.transpose(self.probe_arrays[probe_index], (2, 0, 1))
                    t = tf.transpose(self.target_phs[p], (2, 0, 1))
                    loss += [objective(x, t)]
                else:
                    raise NotImplementedError

        # average loss across probes (note: this will also average across
        # the output of `objective` if it doesn't return a scalar)
        loss = tf.reduce_mean(loss)

        self.losses[(objective, targets)] = loss

        return loss
