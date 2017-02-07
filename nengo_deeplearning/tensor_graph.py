from collections import defaultdict, OrderedDict

from nengo import Process
from nengo.builder.operator import TimeUpdate, SimPyFunc
from nengo.builder.processes import SimProcess
import tensorflow as tf

from nengo_deeplearning import (builder, graph_optimizer, signals, utils,
                                DEBUG)


class TensorGraph(object):
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
                                     if n.size_in == 0]

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

    def build(self, rng):
        self.graph = tf.Graph()
        self.signals = signals.SignalDict(self.sig_map, self.dtype,
                                          self.minibatch_size)

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

            # create base variables
            self.base_vars = []
            for k, (v, trainable) in self.base_arrays_init.items():
                name = "%s_%s_%s_%s" % (
                    v.dtype, "_".join(str(x) for x in v.shape), trainable,
                    str(k)[-9:-1])
                if trainable:
                    # trainable signal, so create Variable
                    with tf.variable_scope("base_vars", reuse=False):
                        var = tf.get_variable(
                            name, initializer=tf.constant_initializer(v),
                            dtype=v.dtype, shape=v.shape, trainable=True)
                else:
                    var = tf.placeholder(tf.as_dtype(v.dtype), shape=v.shape,
                                         name=name)

                # pre-compute indices for the full range (used in scatter_f2)
                self.signals.base_ranges[k] = tf.range(v.shape[0])

                self.base_vars += [var]

            if DEBUG:
                print("created variables")
                print([str(x) for x in self.base_vars])
            self.init_op = tf.global_variables_initializer()

            # set up invariant inputs
            self.build_inputs(rng)

            # pre-build stage
            for ops in self.plan:
                with self.graph.name_scope(utils.sanitize_name(
                        builder.Builder.builders[type(ops[0])].__name__)):
                    builder.Builder.pre_build(ops, self.signals, rng)

            # build stage
            self.build_loop()

    def build_step(self):
        """Build the operators that execute a single simulation timestep
        into the graph.

        Returns
        -------
        probe_tensors : list of `tf.Tensor`
            the Tensor objects representing the data required for each model
            Probe
        side_effects : list of `tf.Tensor`
            the output Tensors of computations that may have side-effects
            (e.g., `Node` functions), meaning that they must be executed each
            time step even if their output doesn't appear to be used
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

        if DEBUG:
            print("=" * 30)
            print("collecting probe tensors")

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
        probe_tensors = [
            self.signals.gather(self.sig_map[self.model.sig[p]["in"]],
                                force_copy=True)
            for p in self.model.probes]

        if DEBUG:
            print("build_step complete")
            print("probe_tensors", [str(x) for x in probe_tensors])
            print("side_effects", [str(x) for x in side_effects])

        return probe_tensors, side_effects

    def build_loop(self):
        """Build simulation loop.

        Loop can be constructed using the `tf.while_loop` architecture, or
        explicitly unrolled.  Unrolling increases graph construction time
        and memory usage, but increases simulation speed.
        """

        def loop_condition(step, stop, *_):
            return step < stop

        def loop_body(step, stop, loop_i, probe_arrays, base_vars):
            self.signals.bases = OrderedDict(
                [(k, v) for k, v in zip(self.base_arrays_init.keys(),
                                        base_vars)])

            # note: nengo step counter is incremented at the beginning of the
            # timestep
            step += 1
            self.signals.step = step

            # build the operators for a single step
            # note: we tie things to the `step` variable so that we can be
            # sure the other things we're tying to the step (side effects and
            # probes) from the previous timestep are executed before the
            # next step starts
            with self.graph.control_dependencies([loop_i]):
                # fill in invariant input data
                # if self.invariant_ph is not None:
                #     self.signals.scatter(self.invariant_ph[0],
                #                          self.invariant_ph[1][loop_i])
                for n in self.invariant_ph:
                    self.signals.scatter(
                        self.sig_map[self.model.sig[n]["out"]],
                        self.invariant_ph[n][loop_i])

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

            # set up tensorboard output
            # if self.tensorboard:
            # TODO: get this part to work again
            # names = []
            # for i, probe in enumerate(self.model.probes):
            #     # add probes to tensorboard summary
            #     if self.tensorboard:
            #         # cut out the memory address so tensorboard doesn't
            #         # display them as separate graphs for each run
            #         name = utils.sanitize_name(probe)
            #         name = name[:name.index("_at_0x")]
            #         count = len(
            #             [x for x in names if x.startswith(name)])
            #         name += "_%d" % count
            #         names += [name]
            #
            #         for j in range(probe.size_in):
            #             tf.summary.scalar("%s.dim%d" % (name, j),
            #                               probe_tensors[i][j])
            #
            # summary_op = tf.summary.merge_all()

            return (step, stop, loop_i, probe_arrays,
                    tuple(self.signals.bases.values()))

        self.step_var = tf.placeholder(tf.int32, shape=(), name="step")
        self.stop_var = tf.placeholder(tf.int32, shape=(), name="stop")
        loop_i = tf.constant(0)
        self.signals.reads_by_base = defaultdict(list)

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
                with self.graph.name_scope("iteration_%d" % n):
                    loop_vars = loop_body(*loop_vars)
        else:
            loop_vars = tf.while_loop(
                loop_condition, loop_body, loop_vars=loop_vars,
                parallel_iterations=1,  # TODO: more parallel iterations
                back_prop=False)

        self.end_step = loop_vars[0]
        self.probe_arrays = [p.pack() for p in loop_vars[3]]
        self.end_base_arrays = loop_vars[4]

    def build_inputs(self, rng):
        # invariant_data = [self.sig_map[self.model.sig[n]["out"]]
        #                   for n in self.invariant_inputs
        #                   if self.model.sig[n]["out"] in self.sig_map]
        self.invariant_funcs = {}
        self.invariant_ph = {}
        for n in self.invariant_inputs:
            if self.model.sig[n]["out"] in self.sig_map:
                self.sig_map[self.model.sig[n]["out"]].load_indices()
                self.invariant_ph[n] = tf.placeholder(
                    self.dtype, (self.step_blocks, n.size_out,
                                 self.minibatch_size))

            if isinstance(n.output, Process):
                self.invariant_funcs[n] = n.output.make_step(
                    (n.size_in,), (n.size_out,), self.dt,
                    n.output.get_rng(rng))
            elif n.size_out > 0:
                self.invariant_funcs[n] = utils.align_func(
                    (n.size_out,), self.dtype)(n.output)
            else:
                self.invariant_funcs[n] = n.output
        # self.invariant_ph = (
        #     None if invariant_data == [] else
        #     (invariant_data, tf.placeholder(
        #         self.dtype, (self.step_blocks, invariant_data.shape[0],
        #                      self.minibatch_size), name="input_data")))

    def build_optimizer(self, optimizer, targets, objective):
        self.target_phs = {}
        with self.graph.as_default(), tf.device(self.device):
            # compute loss
            loss = []
            for p in targets:
                probe_index = self.model.probes.index(p)
                self.target_phs[p] = tf.placeholder(
                    self.dtype, (self.step_blocks, p.size_in,
                                 self.minibatch_size), name="targets")

                if objective == "mse":
                    loss += [tf.square(self.target_phs[p] -
                                       self.probe_arrays[probe_index])]
                elif callable(objective):
                    loss += objective(self.probe_arrays[probe_index],
                                      self.target_phs[p])
                else:
                    raise NotImplementedError

            self.loss = tf.reduce_mean(loss)

            # create optimizer operator
            self.opt_op = optimizer.minimize(
                self.loss, var_list=tf.trainable_variables())

            # get any new variables created by optimizer (so they can be
            # initialized)
            self.opt_slots_init = tf.variables_initializer(
                [optimizer.get_slot(v, name) for v in tf.trainable_variables()
                 for name in optimizer.get_slot_names()])
