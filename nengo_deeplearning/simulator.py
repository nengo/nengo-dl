from collections import Mapping, OrderedDict, defaultdict
import datetime
import logging
import os
import time
import warnings

from nengo.builder import Model
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
         "tests/test_nengo_tests.py:test_warn_on_opensim_del"),

        ("nengo/tests/test_simulator.py:test_signal_init_values",
         "different method required to manually step simulator (see "
         "tests/test_nengo_tests.py:test_signal_init_values"),

        ("nengo/tests/test_simulator.py:test_entry_point",
         "overridden so we can pass custom test simulators (see "
         "tests/test_nengo_tests.py:test_entry_point"),

        ("nengo/tests/test_builder.py:test_signal_init_values",
         "duplicate of test_nengo_tests.py:test_signal_init_values"),

        ("nengo/tests/test_node.py:test_args",
         "time is passed as np.float32, not a float (see "
         "tests/test_nengo_tests.py:test_args"),

        ("nengo/tests/test_node.py:test_unconnected_node",
         "need to set `step_blocks` to ensure node runs the correct number "
         "of times (see tests/test_nengo_tests.py:test_unconnected_node"),
    ]

    """Simulate network using the `nengo_deeplearning` backend.

    Parameters
    ----------
    network : `nengo.Network` or None
        a network object to be built and then simulated. If None,
        then a `nengo.builder.Model` with the build model must be provided
        instead
    dt : float, optional
        length of a simulator timestep, in seconds
    seed : int, optional
        seed for all stochastic operators used in this simulator
    model : `nengo.builder.Model`, optional
        pre-built model object
    tensorboard : bool, optional
        if True, save network output in the tensorflow summary format, which
        can be loaded into Tensorboard
    dtype : tf.DType, optional
        floating point precision to use for simulation
    step_blocks : int, optional
        controls how many simulation steps run each time the graph is executed
        (affects memory usage and graph construction time)
    device : None or "/cpu:0" or "/gpu:[0-n]", optional
        device on which to execute computations (if None then uses the default
        device as determined by tensorflow)
    unroll_simulation : bool, optional
        if True, unroll simulation loop by explicitly building each iteration
        (up to `step_blocks`) into the computation graph. if False, use a
        symbolic loop, which is more general and produces a simpler graph, but
        is likely to be slower to simulate
    """

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 tensorboard=False, dtype=tf.float32, step_blocks=50,
                 device=None, unroll_simulation=True):
        self.closed = None
        self.sess = None
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
            self.model.build(network, progress_bar=False)

        # group mergeable operators
        plan = graph_optimizer.greedy_planner(self.model.operators)
        # plan = graph_optimizer.noop_planner(self.model.operators)

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
        """Resets the simulator to initial conditions.

        Parameters
        ----------
        seed : int, optional
            if not None, overwrite the default simulator seed with this value
            (note: this becomes the new default simulator seed)
        """

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
            self.build_loop()

        print("Construction completed in %s " %
              datetime.timedelta(seconds=int(time.time() - start)))

        # start session
        config = tf.ConfigProto(
            allow_soft_placement=False,
            # log_device_placement=True
        )

        self.sess = tf.Session(graph=self.graph, config=config)
        self.closed = False

        print("Session initialized")

        # initialize variables
        self.sess.run(init_op)

        self.n_steps = 0
        self.time = 0.0
        self.summary = None

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
            # timestep. we don't want to increment it yet, because we need
            # to figure out the side effects first, so we feed in step+1
            # here and then increment it later
            self.signals.step = step + 1

            # build the operators for a single step
            # note: we tie things to the `step` variable so that we can be
            # sure the other things we're tying to the step (side effects and
            # probes) from the previous timestep are executed before the
            # next step starts
            with self.graph.control_dependencies([self.signals.step]):
                probe_tensors, side_effects = self.build_step()

            # copy probe data to array
            for i, p in enumerate(probe_tensors):
                probe_arrays[i] = probe_arrays[i].write(loop_i, p)

            # need to make sure that any operators that could have side
            # effects run each timestep, so we tie them to the step increment.
            # we also need to make sure that all the probe reads happen before
            # those values get overwritten on the next timestep
            with self.graph.control_dependencies(side_effects + probe_tensors):
                step += 1

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

        self.step_var = tf.placeholder(tf.int32)
        self.stop_var = tf.placeholder(tf.int32)
        loop_i = tf.constant(0)

        probe_arrays = [
            tf.TensorArray(
                self.dtype, clear_after_read=False,
                size=0 if self.step_blocks is None else self.step_blocks,
                dynamic_size=self.step_blocks is None)
            for _ in self.model.probes]

        # build simulation loop
        loop_vars = (
            self.step_var, self.stop_var, loop_i, probe_arrays,
            tuple(x._ref() for x in
                  self.get_base_variables(reuse=True).values()))

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
        self.end_base_arrays = loop_vars[4:]

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
        """Run the simulation for one time step."""

        self.run_steps(1)

    def run(self, time_in_seconds):
        """Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for.
        """

        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def run_steps(self, n_steps, profile=False):
        """Simulate for the given number of steps.

        Parameters
        ----------
        n_steps : int
            the number of simulation steps to be executed
        profile : bool, optional
            if True, collect tensorflow profiling information while the
            simulation is running (this will slow down the simulation)

        Notes
        -----
        If `step_blocks` is specified, and `n_steps > step_blocks`, this will
        repeatedly execute `step_blocks` timesteps until the the number of
        steps executed is >= `n_steps`.
        """

        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        if self.step_blocks is not None and n_steps % self.step_blocks != 0:
            warnings.warn(
                "Number of steps (%d) is not an even multiple of `step_blocks`"
                " (%d).  Simulation will run for %d steps, which may have "
                "unintended side effects." %
                (n_steps, self.step_blocks,
                 self.step_blocks * (n_steps // self.step_blocks + 1)))

        print("Simulation started")
        start = time.time()

        if self.step_blocks is None:
            probe_data = self._run_steps(n_steps, profile=profile)

            self.update_probe_data(probe_data, self.n_steps - n_steps, n_steps)
        else:
            # break the run up into `step_blocks` sized chunks
            remaining_steps = n_steps
            while remaining_steps > 0:
                probe_data = self._run_steps(self.step_blocks, profile=profile)
                remaining_steps -= self.step_blocks

                self.update_probe_data(
                    probe_data, self.n_steps - self.step_blocks,
                    self.step_blocks + min(remaining_steps, 0))

            # update n_steps/time
            self.n_steps += remaining_steps
            self.time = self.n_steps * self.dt

        print("Simulation finished in %s" %
              datetime.timedelta(seconds=int(time.time() - start)))

    def _run_steps(self, n_steps, profile=False):
        """Execute `step_blocks` sized segments of the simulation.

        Parameters
        ----------
        n_steps : int
            the number of simulation steps to be executed
        profile : bool, optional
            if True, collect tensorflow profiling information while the
            simulation is running (this will slow down the simulation)
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
                    self.stop_var: self.n_steps + n_steps,
                },
                options=run_options, run_metadata=run_metadata)
        except tf.errors.InternalError as e:
            if e.op.type == "PyFunc":
                raise SimulationError(
                    "Function '%s' caused an error "
                    "(see error log above)" % e.op.name) from None
            else:
                raise e

        # update n_steps
        assert final_step - self.n_steps == n_steps
        self.n_steps = final_step
        self.time = self.n_steps * self.dt

        if profile:
            timeline = Timeline(run_metadata.step_stats)
            with open("nengo_dl_profile.json", "w") as f:
                f.write(timeline.generate_chrome_trace_format())

        return probe_data

    def get_base_variables(self, reuse=False):
        """Loads the base variables used to store all simulation data.

        Parameters
        ----------
        reuse : bool, optional
            if False, create new Variables, otherwise look up the previously
            created Variables

        Returns
        -------
        dict of {(dtype, tuple of int): `tf.Variable`}
            base variables, keyed by the dtype and shape of data stored in
            this variable
        """

        with tf.variable_scope("base_vars", reuse=reuse):
            bases = OrderedDict(
                [(k, tf.get_variable(
                    "%s_%s" % (k[0].__name__, "_".join(str(x) for x in k[1])),
                    initializer=tf.constant_initializer(v),
                    dtype=v.dtype, shape=v.shape))
                 for k, v in self.base_arrays_init.items()])

        return bases

    def close(self):
        """Close the simulation, freeing resources.

        Notes
        -----
        The simulation cannot be restarted after it is closed.  This is not a
        technical limitation, just a design decision made for all Nengo
        simulators.
        """

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
        """Raise a RuntimeWarning if we are deallocated while open."""

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
        dt : float, optional
            the sampling period of the probe to create a range for;
            if None, the simulator's ``dt`` will be used.
        """

        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def update_probe_data(self, probe_data, start, n_steps):
        """Updates the stored probe data (since the last reset) with the data
        from the latest run.

        Downsamples the probe data returned from tensorflow (from every
        simulation timestep) according to probe `sample_every` and the number
        of steps run.

        Parameters
        ----------
        probe_data : list of `np.ndarray`
            probe data from every timestep
        start : int
            the simulation timestep at which probe data starts
        n_steps : int
            the number of timesteps over which we want to collect data
        """

        # first, remove any extra timesteps (due to `step_blocks` mismatch)
        probe_data = [p[:n_steps] for p in probe_data]

        for i, p in enumerate(self.model.probes):
            if p.sample_every is not None:
                # downsample probe according to `sample_every`
                period = p.sample_every / self.dt
                steps = np.arange(start, start + n_steps)
                probe_data[i] = probe_data[i][(steps + 1) % period < 1]

            # update stored probe data
            self.model.params[p] += [probe_data[i]]


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
