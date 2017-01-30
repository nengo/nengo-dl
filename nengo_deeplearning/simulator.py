from collections import Mapping
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

from nengo_deeplearning import signals, DATA_DIR
from nengo_deeplearning.tensor_graph import TensorGraph

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
    minibatch_size : int, optional
        the number of simultaneous inputs that will be passed through the
        network
    """

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 tensorboard=False, dtype=tf.float32, step_blocks=50,
                 device=None, unroll_simulation=True, minibatch_size=None):
        self.closed = None
        self.sess = None
        self.tensorboard = tensorboard
        self.step_blocks = step_blocks
        self.minibatch_size = 1 if minibatch_size is None else minibatch_size

        # build model (uses default nengo builder)
        if model is None:
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            if dt != model.dt:
                warnings.warn("Model dt (%g) does not match Simulator "
                              "dt (%g)" % (model.dt, dt), NengoWarning)
            self.model = model

        if network is not None:
            print("Building network", end="", flush=True)
            start = time.time()
            self.model.build(network, progress_bar=False)
            print("\rBuilding completed in %s " %
                  datetime.timedelta(seconds=int(time.time() - start)))

        # mark trainable signals
        signals.mark_signals(self.model)

        # set up tensorflow graph plan
        self.tensor_graph = TensorGraph(
            self.model, self.dt, step_blocks, unroll_simulation, dtype,
            self.minibatch_size, device)

        self.data = ProbeDict(
            self.model.params,
            {p: (minibatch_size if self.model.sig[p]["in"].minibatched
                 else -1) for p in self.model.probes})

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

        rng = np.random.RandomState(self.seed)

        # clear probe data
        for p in self.model.probes:
            self.model.params[p] = []

        # (re)build graph
        print("Constructing graph", end="", flush=True)
        start = time.time()
        self.tensor_graph.build(rng)
        print("\rConstruction completed in %s " %
              datetime.timedelta(seconds=int(time.time() - start)))

        # output graph description to tensorboard summary
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
                graph=self.tensor_graph.graph)

        # start session
        config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
        )

        self.sess = tf.Session(graph=self.tensor_graph.graph, config=config)
        self.closed = False

        # initialize variables
        self.sess.run(self.tensor_graph.init_op)

        self.n_steps = 0
        self.time = 0.0

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

    def run_steps(self, n_steps, **kwargs):
        """Simulate for the given number of steps.

        Parameters
        ----------
        n_steps : int
            the number of simulation steps to be executed
        kwargs : dict
            see :func:`._run_steps`

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

        print("Simulation started", end="", flush=True)
        start = time.time()

        if self.step_blocks is None:
            probe_data = self._run_steps(n_steps, **kwargs)

            self.update_probe_data(probe_data, self.n_steps - n_steps, n_steps)
        else:
            # break the run up into `step_blocks` sized chunks
            remaining_steps = n_steps
            while remaining_steps > 0:
                probe_data = self._run_steps(self.step_blocks, **kwargs)
                remaining_steps -= self.step_blocks

                self.update_probe_data(
                    probe_data, self.n_steps - self.step_blocks,
                    self.step_blocks + min(remaining_steps, 0))

            # update n_steps/time
            self.n_steps += remaining_steps
            self.time = self.n_steps * self.dt

        print("\rSimulation completed in %s" %
              datetime.timedelta(seconds=int(time.time() - start)))

    def _run_steps(self, n_steps, profile=False, input_feeds=None):
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

        # generate node feeds
        if input_feeds is None:
            input_feeds = {}
        feed_vals = []
        for n in self.tensor_graph.invariant_inputs:
            if n in input_feeds:
                # move minibatch dimension to the end
                feed_val = np.moveaxis(input_feeds[n], 0, -1)
            elif isinstance(n.output, np.ndarray):
                feed_val = np.tile(n.output[None, :, None],
                                   (n_steps, 1, self.minibatch_size))
            else:
                func = self.tensor_graph.invariant_funcs[n]

                feed_val = []
                for i in range(self.n_steps + 1, self.n_steps + n_steps + 1):
                    func_out = func(i * self.dt)
                    if func_out is not None:
                        # note: need to copy the output of func
                        feed_val += [np.array(func_out)]

                if self.model.sig[n]["out"] in self.tensor_graph.sig_map:
                    feed_val = np.stack(feed_val, axis=0)
                    feed_val = np.tile(feed_val[..., None],
                                       (1, 1, self.minibatch_size))

            if self.model.sig[n]["out"] in self.tensor_graph.sig_map:
                feed_vals += [feed_val]

        # execute the loop
        feed_dict = {
            self.tensor_graph.step_var: self.n_steps,
            self.tensor_graph.stop_var: self.n_steps + n_steps,
        }
        if self.tensor_graph.invariant_ph is not None:
            feed_dict[self.tensor_graph.invariant_ph[1]] = np.concatenate(
                feed_vals, axis=1)

        try:
            final_step, probe_data, final_bases = self.sess.run(
                [self.tensor_graph.end_step, self.tensor_graph.probe_arrays,
                 self.tensor_graph.end_base_arrays],
                feed_dict=feed_dict,
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

    def __init__(self, raw, minibatches):
        self.raw = raw
        self.minibatches = minibatches
        self._cache = {}

    def __getitem__(self, key):
        cache_miss = (key not in self._cache or
                      len(self._cache[key]) != len(self.raw[key]))
        if cache_miss:
            rval = self.raw[key]
            if isinstance(rval, list):
                # combine data from _run_steps iterations
                rval = np.concatenate(rval, axis=0)

                if self.minibatches[key] != -1:
                    if self.minibatches[key] is None:
                        # get rid of batch dimension
                        rval = rval[..., 0]
                    else:
                        # move batch dimension to front
                        rval = np.moveaxis(rval, -1, 0)

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
