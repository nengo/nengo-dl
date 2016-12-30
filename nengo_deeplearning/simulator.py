from collections import Mapping
import datetime
import logging
import os
import time
import warnings

from nengo.builder import Model
from nengo.builder.neurons import SimNeurons
from nengo.exceptions import (ReadonlyError, SimulatorClosed, NengoWarning,
                              SimulationError)
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph
import numpy as np
import tensorflow as tf
from tensorflow.python.client.timeline import Timeline

from nengo_deeplearning import signals, utils, Builder, DATA_DIR

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
                 max_run_steps=None):
        self.closed = None
        self.sess = None
        self.progress_bar = progress_bar
        self.tensorboard = tensorboard
        self.dtype = dtype
        self.max_run_steps = max_run_steps

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

        # figure out order of operations based on signal dependencies
        self.op_order = toposort(operator_depencency_graph(
            self.model.operators))

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
        self.signals = signals.SignalDict(self.dtype)
        with self.graph.as_default():
            state_signals = []
            for op in self.op_order:
                state_signals += op.updates

                if isinstance(op, SimNeurons):
                    state_signals += op.states

            self.build_loop(state_signals)

            init_op = tf.global_variables_initializer()

        # clear probe data
        for p in self.model.probes:
            self.model.params[p] = np.zeros(
                (0,) + self.model.sig[p]["in"].shape,
                dtype=self.dtype.as_numpy_dtype)

        # start session
        self.sess = tf.Session(graph=self.graph)
        self.closed = False

        # initialize variables
        self.sess.run(init_op)

        self.n_steps = 0
        self.time = 0.0
        self.summary = None

    def build_step(self):
        # build operators
        side_effects = []
        stateful = []
        for op in self.op_order:
            if PreserveValue is not None and isinstance(op, PreserveValue):
                continue

            with self.graph.name_scope(utils.function_name(op)):
                outputs = Builder.build(op, self.signals, self.dt,
                                        self.rng)

            if len(op.updates) > 0 or isinstance(op, SimNeurons):
                stateful += outputs
            elif outputs is not None:
                side_effects += [x for x in outputs
                                 if x.op.type == "PyFunc"]

        probe_tensors = [self.signals[self.model.sig[p]["in"]]
                         for p in self.model.probes]

        # TODO: figure out why this is necessary, then a more graceful solution
        # something to do with this telling tensorflow that the
        # probe read needs to happen each iteration
        probe_tensors = [p + 0 if p.dtype._is_ref_dtype else p
                         for p in probe_tensors]

        return probe_tensors, stateful, side_effects

    def build_loop(self, state_sigs):
        def loop_condition(step, start, stop, *_):
            return step < stop

        def loop_body(step, start, stop, probe_arrays, state):
            # note: this function is only actually called once (to determine
            # the graph structure of the loop body), it isn't called each
            # iteration

            # note: nengo step counter is incremented at the beginning of the
            # timestep. we don't want to increment it yet, because we need
            # to figure out the side effects first, so we feed in step+1
            # here and then increment it later
            self.signals[self.model.step] = step + 1
            for i, sig in enumerate(state_sigs):
                self.signals[sig] = state[i]

            # build the operators for a single step
            probe_tensors, state, side_effects = self.build_step()

            # copy probe data to array
            # TODO: maybe do this off the GPU (to save memory)?
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
            with self.graph.control_dependencies(side_effects):
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

            return step, start, stop, probe_arrays, state

        with self.graph.as_default():
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
            probe_arrays = [
                tf.TensorArray(
                    utils.cast_dtype(self.model.sig[p]["in"].dtype,
                                     tf.float32),
                    size=(0 if self.max_run_steps is None else
                          self.max_run_steps),
                    dynamic_size=self.max_run_steps is None,
                    clear_after_read=True)
                for i, p in enumerate(self.model.probes)]

            # TODO: fill in state values from previous run_steps call
            state = [tf.constant(np.asarray(u.initial_value,
                                            dtype=self.dtype.as_numpy_dtype))
                     for u in state_sigs]

            self.end_step, _, _, probe_arrays, _ = tf.while_loop(
                loop_condition, loop_body,
                loop_vars=(self.step_var, self.start_var, self.stop_var,
                           probe_arrays, state),
                parallel_iterations=1)

            self.probe_arrays = [p.pack() for p in probe_arrays]

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

    # def step(self, profile=False):
    #     if self.closed:
    #         raise SimulatorClosed("Simulator cannot run because it is closed.")
    #
    #     # we need to explicitly fetch the node_outputs and updates (even though
    #     # we don't use those values) to make sure those ops run
    #     # note: using a fetches dict has the effect of removing any duplicates
    #     # (so e.g. we don't double-fetch something if it is in both
    #     # probe_tensors and updates)
    #     step_tensor = self.signals[self.model.step]
    #     time_tensor = self.signals[self.model.time]
    #     fetches = {x: x for x in
    #                [step_tensor, time_tensor] + self.probe_tensors +
    #                self.node_outputs + self.updates}
    #
    #     if self.tensorboard and self.summary_op is not None:
    #         fetches[self.summary_op] = self.summary_op
    #
    #     if profile:
    #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #         run_metadata = tf.RunMetadata()
    #     else:
    #         run_options = None
    #         run_metadata = None
    #
    #     try:
    #         output = self.sess.run(fetches, options=run_options,
    #                                run_metadata=run_metadata)
    #     except tf.errors.InternalError as e:
    #         utils.handle_internal_error(e)
    #
    #     self.n_steps = output[step_tensor]
    #     self.time = output[time_tensor]
    #
    #     for i, p in enumerate(self.model.probes):
    #         period = (1 if p.sample_every is None else
    #                   p.sample_every / self.dt)
    #
    #         if self.n_steps % period < 1:
    #             self.model.params[p].append(output[self.probe_tensors[i]])
    #
    #     if self.tensorboard and self.summary_op is not None:
    #         self.summary.add_summary(output[self.summary_op], self.n_steps)
    #
    #     if profile:
    #         timeline = Timeline(run_metadata.step_stats)
    #         with open("timeline.json", "w") as f:
    #             f.write(timeline.generate_chrome_trace_format())

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

        print("Starting simulation")
        start = time.time()

        if self.max_run_steps is None:
            self._run_steps(n_steps, profile=profile)
        else:
            # break the run up into `max_run_steps` chunks
            remaining_steps = n_steps
            while remaining_steps > 0:
                self._run_steps(min(self.max_run_steps, remaining_steps),
                                profile=profile)
                remaining_steps -= self.max_run_steps

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
            final_step, probe_data = self.sess.run(
                [self.end_step, self.probe_arrays],
                feed_dict={self.step_var: self.n_steps,
                           self.start_var: self.n_steps,
                           self.stop_var: self.n_steps + n_steps},
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
            self.model.params[p] = np.concatenate(
                (self.model.params[p], probe_data[i]), axis=0)

        if profile:
            timeline = Timeline(run_metadata.step_stats)
            with open("timeline.json", "w") as f:
                f.write(timeline.generate_chrome_trace_format())

    def close(self):
        if not self.closed:
            self.sess.close()
            self.closed = True
            self.sess = None

            if self.summary is not None:
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
                rval = np.asarray(rval)
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
