from collections import Mapping, defaultdict
import logging
import os
import warnings

from nengo.builder import Model
from nengo.builder.operator import SimPyFunc
from nengo.exceptions import (ReadonlyError, SimulatorClosed, NengoWarning)
from nengo.utils.graphs import toposort
from nengo.utils.progress import ProgressTracker
from nengo.utils.simulator import operator_depencency_graph
import numpy as np
import tensorflow as tf

from nengo_deeplearning import signals, utils, Builder, DATA_DIR

logger = logging.getLogger(__name__)


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

        ("nengo/tests/test_node.py:test_args",
         "time is passed as np.float32, not a float (see "
         "tests/test_simulator.py:test_args")
    ]

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 progress_bar=True, tensorboard=False, dtype=tf.float32):
        self.closed = None
        self.sess = None
        self.progress_bar = progress_bar
        self.tensorboard = tensorboard
        self.dtype = dtype

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

        # convert operator graph to tensorflow graph
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
        # TODO: just rebuild the sim_process nodes
        self.build_graph()

        # set up probes
        with self.graph.as_default():
            names = []
            for i, probe in enumerate(self.model.probes):
                # clear probe data
                self.model.params[probe] = []

                # add probes to tensorboard summary
                if self.tensorboard:
                    # cut out the memory address so tensorboard doesn't display
                    # them as separate graphs for each run
                    name = utils.sanitize_name(probe)
                    name = name[:name.index("_at_0x")]
                    count = len([x for x in names if x.startswith(name)])
                    name += "_%d" % count
                    names += [name]

                    for j in range(probe.size_in):
                        tf.summary.scalar(
                            "%s.dim%d" % (name, j), self.probe_tensors[i][j])

            self.summary_op = tf.summary.merge_all()

        # start session
        self.sess = tf.Session(graph=self.graph)
        print("Session initialized")
        self.closed = False

        # initialize signals
        self.sess.run(self.init_op)

        self.n_steps = 0
        self.time = 0.0

        if self.tensorboard:
            directory = "%s/%s" % (DATA_DIR, self.model.toplevel.label)
            if os.path.isdir(directory):
                run_number = max([int(x[4:]) for x in os.listdir(directory)
                                  if x.startswith("run")]) + 1
            else:
                run_number = 0
            self.summary = tf.summary.FileWriter(
                "%s/run_%d" % (directory, run_number), graph=self.graph)
        else:
            self.summary = None

    def build_graph(self):
        with tf.Graph().as_default() as self.graph:
            self.signals = signals.SignalDict(
                self.dtype,
                {self.model.step: tf.Variable(0, name="step"),
                 self.model.time: tf.Variable(0.0, dtype=self.dtype,
                                              name="time")})

            # build all the non-update operators
            self.node_outputs = []
            self.updates = []
            self.reads = defaultdict(list)
            for op in self.op_order:
                # note: the only thing we need to explicitly sequence is that
                # updates happen after reads. the other requirements (sets ->
                # incs -> reads) are implicitly enforced because assign ops
                # produce a new tensor that future ops will operate on
                dependencies = [x for sig in op.updates
                                for x in self.reads[self.signals[sig]]]
                with self.graph.control_dependencies(dependencies):
                    with self.graph.name_scope(utils.function_name(op)):
                        outputs = Builder.build(op, self.signals, self.dt,
                                                self.rng)

                for r in op.reads:
                    self.reads[self.signals[r]] += outputs

                if isinstance(op, SimPyFunc):
                    self.node_outputs += outputs
                if len(op.updates) > 0:
                    self.updates += [self.signals[x] for x in op.updates]

            self.probe_tensors = [self.signals[self.model.sig[p]["in"]]
                                  for p in self.model.probes]

            self.init_op = tf.global_variables_initializer()

    def step(self):
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        # we need to explicitly fetch the node_outputs and updates (even though
        # we don't use those values) to make sure those ops run
        # note: using a fetches dict has the effect of removing any duplicates
        # (so e.g. we don't double-fetch something if it is in both
        # probe_tensors and updates)
        step_tensor = self.signals[self.model.step]
        time_tensor = self.signals[self.model.time]
        fetches = {x: x for x in
                   [step_tensor, time_tensor] + self.probe_tensors +
                   self.node_outputs + self.updates}

        if self.tensorboard:
            fetches[self.summary_op] = self.summary_op

        try:
            output = self.sess.run(fetches)
        except tf.errors.InternalError as e:
            utils.handle_internal_error(e)

        self.n_steps = output[step_tensor]
        self.time = output[time_tensor]

        for i, p in enumerate(self.model.probes):
            period = (1 if p.sample_every is None else
                      p.sample_every / self.dt)

            if self.n_steps % period < 1:
                self.model.params[p].append(output[self.probe_tensors[i]])

        if self.tensorboard:
            self.summary.add_summary(output[self.summary_op], self.n_steps)

    def run(self, time_in_seconds, progress_bar=None):
        """Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for.
        progress_bar : bool or `.ProgressBar` or `.ProgressUpdater`, optional \
                       (Default: True)
            Progress bar for displaying the progress of the simulation run.

            If True, the default progress bar will be used.
            If False, the progress bar will be disabled.
            For more control over the progress bar, pass in a `.ProgressBar`
            or `.ProgressUpdater` instance.
        """
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.info("Running %s for %f seconds, or %d steps",
                    self.model.label, time_in_seconds, steps)
        self.run_steps(steps, progress_bar=progress_bar)

    def run_steps(self, steps, progress_bar=None):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        progress_bar : bool or `.ProgressBar` or `.ProgressUpdater`, optional \
                       (Default: True)
            Progress bar for displaying the progress of the simulation run.

            If True, the default progress bar will be used.
            If False, the progress bar will be disabled.
            For more control over the progress bar, pass in a `.ProgressBar`
            or `.ProgressUpdater` instance.
        """

        if progress_bar is None:
            progress_bar = self.progress_bar
        with ProgressTracker(steps, progress_bar, "Simulation") as progress:
            for _ in range(steps):
                self.step()
                progress.step()

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
