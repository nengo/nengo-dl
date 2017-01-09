from collections import Mapping, OrderedDict, defaultdict
import copy
import datetime
import logging
import os
import time
import warnings

from nengo.builder import Model
from nengo.builder.neurons import SimNeurons
from nengo.exceptions import (ReadonlyError, SimulatorClosed, NengoWarning,
                              SimulationError)
import numpy as np
import tensorflow as tf
from tensorflow.python.client.timeline import Timeline

from nengo_deeplearning import (signals, utils, graph_optimizer, DEBUG,
                                DATA_DIR)
from nengo_deeplearning.builder import Builder

logger = logging.getLogger(__name__)

try:
    from nengo.builder.operator import PreserveValue
except:
    PreserveValue = None


class Simulator(object):
    # unsupported unit tests
    unsupported = [
        ("nengo/tests/test_simulator.py:test_warn_on_opensim_del",
         "nengo_deeplearning raises a different (more visible) warning (see "
         "tests/test_simulator.py:test_warn_on_opensim_del"),

        ("nengo/tests/test_simulator.py:test_signal_init_values",
         "different method required to manually step simulator (see "
         "tests/test_simulator.py:test_signal_init_values"),

        ("nengo/tests/test_simulator.py:test_entry_point",
         "overridden so we can pass custom test simulators (see "
         "tests/test_simulator.py:test_entry_point"),

        ("nengo/tests/test_builder.py:test_signal_init_values",
         "duplicate of test_simulator.py:test_signal_init_values"),

        ("nengo/tests/test_node.py:test_args",
         "time is passed as np.float32, not a float (see "
         "tests/test_simulator.py:test_args")
    ]

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 progress_bar=True, tensorboard=False, dtype=tf.float32,
                 step_blocks=50, device=None, unroll_simulation=True):
        self.closed = None
        self.sess = None
        self.progress_bar = progress_bar
        self.tensorboard = tensorboard
        self.dtype = dtype
        self.step_blocks = step_blocks
        self.device = device
        self.unroll_simulation = unroll_simulation

        # build model (uses default nengo builder)
        if model is None:
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            if dt != model.dt:
                warnings.warn("Model dt (%g) does not match Simulator "
                              "dt (%g)" % (model.dt, dt), NengoWarning)
            self.model = model

        if network is not None:
            self.model.build(network, progress_bar=self.progress_bar)

        # group mergeable operators
        plan = graph_optimizer.greedy_planner(self.model.operators)

        # order signals/operators to promote contiguous reads
        signals, self.plan = graph_optimizer.order_signals(plan, n_passes=10)
        # signals, self.plan = graph_optimizer.noop_order_signals(plan)

        # create base arrays and map Signals to TensorSignals (views on those
        # base arrays)
        self.base_arrays_init, self.sig_map = graph_optimizer.create_signals(
            signals, self.plan, float_type=dtype.as_numpy_dtype)

        self.data = ProbeDict(self.model.params)

        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        # close old session
        if self.sess is not None:
            self.close()

        if seed is not None:
            self.seed = seed

        self.rng = np.random.RandomState(self.seed)

        # (re)build graph
        self.graph = tf.Graph()

        print("Constructing graph")
        start = time.time()
        with self.graph.as_default(), tf.device(self.device):
            # clear probe data
            for p in self.model.probes:
                self.sig_map[self.model.sig[p]["in"]].load_indices()
                self.model.params[p] = []

            # create signal dict
            self.signals = signals.SignalDict(self.sig_map, self.dtype,
                                              self.dt)

            # create base variables
            base_vars = self.get_base_variables()
            if DEBUG:
                print("created variables")
                print([(k, v.name) for k, v in base_vars.items()])
            init_op = tf.global_variables_initializer()

            # pre-build stage
            for op_type, ops in self.plan:
                with self.graph.name_scope(
                        utils.sanitize_name(op_type.__name__)):
                    Builder.pre_build(op_type, ops, self.signals, self.rng)

            # build stage
            if self.unroll_simulation:
                self.build_loop_unrolled()
            else:
                self.build_loop()

        print("Construction completed in %s " %
              datetime.timedelta(seconds=int(time.time() - start)))

        # start session
        config = tf.ConfigProto(
            allow_soft_placement=False,
            # log_device_placement=True
        )
        print("Initializing session")
        self.sess = tf.Session(graph=self.graph, config=config)
        self.closed = False

        # initialize variables
        self.sess.run(init_op)

        self.n_steps = 0
        self.time = 0.0
        self.summary = None

    def build_step(self):
        # load base variables
        self.signals.bases = self.get_base_variables(reuse=True)

        # build operators
        side_effects = []
        self.signals.reads_by_base = defaultdict(list)

        # manually build TimeUpdate. we don't include this in the plan,
        # because loop variables (`step`) are (semi?) pinned to the CPU, which
        # causes the whole variable to get pinned to the CPU if we include
        # `step` as part of the normal planning process.
        self.signals.time = tf.cast(self.signals.step,
                                    self.signals.dtype) * self.dt

        for op_type, ops in self.plan:
            with self.graph.name_scope(utils.sanitize_name(op_type.__name__)):
                outputs = Builder.build(ops, self.signals)

            if outputs is not None and len(outputs) > 0:
                side_effects += outputs

        probe_tensors = [self.signals[self.model.sig[p]["in"]]
                         for p in self.model.probes]

        # TODO: figure out why this is necessary, then a more graceful solution
        # something to do with this telling tensorflow that the
        # probe read needs to happen each iteration
        probe_tensors = [p + 0 for p in probe_tensors]

        if DEBUG:
            print("build_step complete")
            print("probe_tensors", [str(x) for x in probe_tensors])
            print("side_effects", [str(x) for x in side_effects])

        return probe_tensors, side_effects

    def build_loop_unrolled(self):
        probe_arrays = [
            tf.TensorArray(
                utils.cast_dtype(self.model.sig[p]["in"].dtype,
                                 tf.float32),
                size=(0 if self.step_blocks is None else
                      self.step_blocks),
                dynamic_size=self.step_blocks is None,
                clear_after_read=True)
            for i, p in enumerate(self.model.probes)]
        # self.probe_arrays_ph = [tf.placeholder(
        #     tf.float32, shape=(None, p.size_in)) for p in self.model.probes]

        self.step_var = tf.placeholder(tf.int32, shape=())
        step = tf.identity(self.step_var)
        loop_i = tf.constant(0)

        # note: these aren't used, they are just for compatibility with
        # build_loop()
        self.start_var = tf.placeholder(tf.int32)
        self.stop_var = tf.placeholder(tf.int32)

        probe_reads = []
        for n in range(self.step_blocks):
            self.signals.step = step + 1

            # build the operators for a single step
            # note: we have to make sure that all the probe values are read
            # from the previous step before being overwritten in the next
            with self.graph.name_scope("iteration_%d" % n), \
                 self.graph.control_dependencies(probe_reads):
                probe_tensors, side_effects = self.build_step()

            # copy probe data to array
            probe_reads = [tf.identity(p) for p in probe_tensors]
            for i, p in enumerate(probe_reads):
                period = (
                    1 if self.model.probes[i].sample_every is None else
                    self.model.probes[i].sample_every / self.dt)

                if p.dtype != tf.float32:
                    p = tf.cast(p, tf.float32)

                if period == 1:
                    probe_arrays[i] = probe_arrays[i].write(loop_i, p)
                else:
                    index = tf.cast(
                        tf.cast(loop_i, tf.float32) / period,
                        tf.int32)
                    condition = tf.cast(step + 1, self.dtype) % period < 1
                    probe_arrays[i] = tf.case(
                        [(condition,
                          lambda: probe_arrays[i].write(index, p))],
                        default=lambda: probe_arrays[i])

            # need to make sure that any operators that could have side
            # effects run each timestep, so we tie them to the step increment
            with self.graph.control_dependencies(side_effects):
                step += 1

            loop_i += 1

        self.end_step = step
        self.probe_arrays = [p.pack() for p in probe_arrays]
        self.end_base_arrays = [b for b in self.signals.bases.values()]

        if self.tensorboard:
            directory = "%s/%s" % (DATA_DIR, self.model.toplevel.label)
            if os.path.isdir(directory):
                run_number = max(
                    [int(x[4:]) for x in os.listdir(directory)
                     if x.startswith("run")]) + 1
            else:
                run_number = 0
            self.summary = tf.summary.FileWriter(
                "%s/run_%d" % (directory, run_number),
                graph=self.graph)

    def build_loop(self):
        def loop_condition(step, start, stop, *_):
            return step < stop

        def loop_body(step, start, stop, probe_arrays):
            # note: nengo step counter is incremented at the beginning of the
            # timestep. we don't want to increment it yet, because we need
            # to figure out the side effects first, so we feed in step+1
            # here and then increment it later
            self.signals.step = step + 1

            # build the operators for a single step
            probe_tensors, side_effects = self.build_step()

            # copy probe data to array
            for i, p in enumerate(probe_tensors):
                period = (1 if self.model.probes[i].sample_every is None else
                          self.model.probes[i].sample_every / self.dt)

                p = tf.cast(p, tf.float32)

                if period == 1:
                    probe_arrays[i] = probe_arrays[i].write(step - start, p)
                else:
                    index = tf.cast(tf.cast(step - start, tf.float32) / period,
                                    tf.int32)
                    condition = tf.cast(step + 1, self.dtype) % period < 1
                    probe_arrays[i] = tf.case(
                        [(condition, lambda: probe_arrays[i].write(index, p))],
                        default=lambda: probe_arrays[i])

            # need to make sure that any operators that could have side
            # effects run each timestep, so we tie them to the step increment
            with self.graph.control_dependencies(
                            side_effects + list(self.signals.bases.values())):
                step += 1

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

            return step, start, stop, probe_arrays

        self.step_var = tf.placeholder(tf.int32)
        self.start_var = tf.placeholder(tf.int32)
        self.stop_var = tf.placeholder(tf.int32)

        # note: probe_arrays need to be float32 because in the tensorflow
        # case logic they end up comparing the tensorarray dtype to the
        # tensorarray.flow dtype (which is always float32). could submit
        # a patch to tensorflow if this is a significant issue, but
        # it's probably a good idea to have the probe arrays be float32
        # anyway
        # for future reference, the patch would be in
        # tensorflow/python/ops/control_flow_ops.py:2956
        # def _correct_empty(v):
        #     ...
        #     else:
        #         dtype = (v.flow.dtype if
        #                  isinstance(v, tensor_array_ops.TensorArray)
        #                  else v.dtype)
        #         return array_ops.constant(dtype.as_numpy_dtype())
        probe_arrays = []
        for i, p in enumerate(self.model.probes):
            probe_period = (1 if p.sample_every is None else
                            int(p.sample_every / self.dt))
            size = (0 if self.step_blocks is None else
                    self.step_blocks // probe_period)
            probe_arrays += [
                tf.TensorArray(
                    tf.float32, size=size,
                    dynamic_size=self.step_blocks is None,
                    clear_after_read=True)
            ]

        loop_output = tf.while_loop(
            loop_condition, loop_body,
            loop_vars=(self.step_var, self.start_var, self.stop_var,
                       probe_arrays),
            parallel_iterations=1,  # TODO: more parallel iterations
            back_prop=False)

        self.end_step = loop_output[0]
        self.probe_arrays = [p.pack() for p in loop_output[3]]
        self.end_base_arrays = list(
            self.get_base_variables(reuse=True).values())

        if self.tensorboard:
            directory = "%s/%s" % (DATA_DIR, self.model.toplevel.label)
            if os.path.isdir(directory):
                run_number = max(
                    [int(x[4:]) for x in os.listdir(directory)
                     if x.startswith("run")]) + 1
            else:
                run_number = 0
            self.summary = tf.summary.FileWriter(
                "%s/run_%d" % (directory, run_number),
                graph=self.graph)

    def step(self):
        self.run_steps(1)

    def run(self, time_in_seconds):
        """Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for.
        """
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.info("Running %s for %f seconds, or %d steps",
                    self.model.label, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, n_steps, profile=False):
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        if self.step_blocks is not None and n_steps % self.step_blocks != 0:
            raise SimulationError(
                "Number of steps (%d) must be an even multiple of "
                "`step_blocks` (%d) " % (n_steps, self.step_blocks))

        print("Simulation started")
        start = time.time()

        if self.step_blocks is None:
            self._run_steps(n_steps, profile=profile)
        else:
            # break the run up into `step_blocks` sized chunks
            remaining_steps = n_steps
            while remaining_steps > 0:
                self._run_steps(self.step_blocks, profile=profile)
                remaining_steps -= self.step_blocks

        print("Simulation finished in %s" %
              datetime.timedelta(seconds=int(time.time() - start)))

    def _run_steps(self, n_steps, profile=False):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to run the simulation for.
        """

        if profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # execute the loop
        try:
            final_step, probe_data, final_bases = self.sess.run(
                [self.end_step, self.probe_arrays, self.end_base_arrays],
                feed_dict={
                    self.step_var: self.n_steps,
                    self.start_var: self.n_steps,
                    self.stop_var: self.n_steps + n_steps,
                },
                options=run_options, run_metadata=run_metadata)
        except tf.errors.InternalError as e:
            if e.op.type == "PyFunc":
                raise SimulationError(
                    "Function '%s' caused an error "
                    "(see error log above)" % e.op.name) from None

            raise e

        # update n_steps
        assert final_step - self.n_steps == n_steps
        self.n_steps = final_step
        self.time = self.n_steps * self.dt

        # update probe data
        for i, p in enumerate(self.model.probes):
            self.model.params[p] += [probe_data[i]]

        if profile:
            timeline = Timeline(run_metadata.step_stats)
            with open("timeline.json", "w") as f:
                f.write(timeline.generate_chrome_trace_format())

    def get_base_variables(self, reuse=False):
        with tf.variable_scope("base_vars", reuse=reuse):
            bases = OrderedDict(
                [(k, tf.get_variable(
                    "%s_%s" % (k[0].__name__,
                               "_".join(str(x) for x in k[1])),
                    initializer=tf.constant_initializer(v),
                    dtype=v.dtype, shape=v.shape))
                 for k, v in self.base_arrays_init.items()])

        return bases

    def close(self):
        if not self.closed:
            self.sess.close()
            self.closed = True
            self.sess = None

            # note: we use getattr in case it crashes before the summary
            # object is created
            if getattr(self, "summary", None) is not None:
                self.summary.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def dt(self):
        """(float) The time step of the simulator."""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise ReadonlyError(attr='dt', obj=self)

    def __del__(self):
        """Raise a ResourceWarning if we are deallocated while open."""
        if self.closed is not None and not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. "
                "Simulators should be closed manually to ensure resources "
                "are properly freed." % self.model, RuntimeWarning)
            self.close()

    def trange(self, dt=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        dt : float, optional (Default: None)
            The sampling period of the probe to create a range for.
            If None, the simulator's ``dt`` will be used.
        """
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)


class ProbeDict(Mapping):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw):
        self.raw = raw
        self._cache = {}

    def __getitem__(self, key):
        cache_miss = (key not in self._cache or
                      len(self._cache[key]) != len(self.raw[key]))
        if cache_miss:
            rval = self.raw[key]
            if isinstance(rval, list):
                rval = np.concatenate(rval, axis=0)
                rval.setflags(write=False)
            self._cache[key] = rval
        return self._cache[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def __str__(self):
        return str(self.raw)
