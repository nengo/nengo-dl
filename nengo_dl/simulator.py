from __future__ import print_function

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

from nengo_dl import signals, utils, DATA_DIR
from nengo_dl.tensor_graph import TensorGraph
from nengo_dl.utils import print_and_flush

logger = logging.getLogger(__name__)


class Simulator(object):
    """Simulate network using the ``nengo_dl`` backend.

    Parameters
    ----------
    network : :class:`~nengo:nengo.Network` or None
        a network object to be built and then simulated. If None,
        then a built model must be passed to ``model`` instead
    dt : float, optional
        length of a simulator timestep, in seconds
    seed : int, optional
        seed for all stochastic operators used in this simulator
    model : :class:`~nengo:nengo.builder.Model`, optional
        pre-built model object
    tensorboard : bool, optional
        if True, save network output in the Tensorflow summary format,
        which can be loaded into Tensorboard
    dtype : ``tf.DType``, optional
        floating point precision to use for simulation
    step_blocks : int, optional
        controls how many simulation steps run each time the graph is
        executed (affects memory usage and graph construction time)
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``, optional
        device on which to execute computations (if None then uses the
        default device as determined by Tensorflow)
    unroll_simulation : bool, optional
        if True, unroll simulation loop by explicitly building each iteration
        (up to ``step_blocks``) into the computation graph. if False, use a
        symbolic loop, which is more general and produces a simpler graph, but
        is likely to be slower to simulate
    minibatch_size : int, optional
        the number of simultaneous inputs that will be passed through the
        network
    """

    # unsupported unit tests
    unsupported = [
        ("nengo/tests/test_simulator.py:test_warn_on_opensim_del",
         "nengo_dl raises a different (more visible) warning (see "
         "tests/test_nengo_tests.py:test_warn_on_opensim_del"),

        ("nengo/tests/test_simulator.py:test_signal_init_values",
         "different method required to manually step simulator (see "
         "tests/test_nengo_tests.py:test_signal_init_values"),

        ("nengo/tests/test_simulator.py:test_entry_point",
         "overridden so we can pass custom test simulators (see "
         "tests/test_nengo_tests.py:test_entry_point"),

        ("nengo/tests/test_node.py:test_args",
         "time is passed as np.float32, not a float (see "
         "tests/test_nengo_tests.py:test_args"),

        ("nengo/tests/test_node.py:test_unconnected_node",
         "need to set `step_blocks` to ensure node runs the correct number "
         "of times (see tests/test_nengo_tests.py:test_unconnected_node"),
    ]

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 tensorboard=False, dtype=tf.float32, step_blocks=50,
                 device=None, unroll_simulation=True, minibatch_size=None):
        self.closed = None
        self.sess = None
        self.tensorboard = tensorboard
        self.step_blocks = step_blocks
        self.minibatch_size = 1 if minibatch_size is None else minibatch_size

        # TODO: allow the simulator to be called flexibly with/without
        # minibatching

        # TODO: there's some possible bugs with device=None and
        # unroll_simulation=False

        # build model (uses default nengo builder)
        if model is None:
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt))
        else:
            if dt != model.dt:
                warnings.warn("Model dt (%g) does not match Simulator "
                              "dt (%g)" % (model.dt, dt), NengoWarning)
            self.model = model

        if network is not None:
            print_and_flush("Building network", end="")
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

        self.rng = np.random.RandomState(self.seed)
        # TODO: why is setting the tensorflow seed necessary to make
        # gradient descent training deterministic?
        tf.set_random_seed(self.seed)

        # clear probe data
        for p in self.model.probes:
            self.model.params[p] = []

        # (re)build graph
        print_and_flush("Constructing graph", end="")
        start = time.time()
        self.tensor_graph.build(self.rng)
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

        self.sess = tf.InteractiveSession(graph=self.tensor_graph.graph,
                                          config=config)
        self.closed = False

        # initialize variables
        self.sess.run(self.tensor_graph.init_op)

        self.n_steps = 0
        self.time = 0.0
        self.final_bases = [
            x[0] for x in self.tensor_graph.base_arrays_init.values()]

    def step(self, **kwargs):
        """Run the simulation for one time step.

        Parameters
        ----------
        kwargs : dict
            see :meth:`._run_steps`
        """

        self.run_steps(1, **kwargs)

    def run(self, time_in_seconds, **kwargs):
        """Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            amount of time to run the simulation for
        kwargs : dict
            see :meth:`._run_steps`
        """

        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps, **kwargs)

    def run_steps(self, n_steps, **kwargs):
        """Simulate for the given number of steps.

        Parameters
        ----------
        n_steps : int
            the number of simulation steps to be executed
        kwargs : dict
            see :meth:`._run_steps`

        Notes
        -----
        If ``step_blocks`` is specified, and ``n_steps > step_blocks``, this
        will repeatedly execute ``step_blocks`` timesteps until the the number
        of steps executed is >= ``n_steps``.
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

        print_and_flush("Simulation started", end="")
        start = time.time()

        if self.step_blocks is None:
            probe_data = self._run_steps(n_steps, **kwargs)

            self._update_probe_data(probe_data, self.n_steps - n_steps,
                                    n_steps)
        else:
            # break the run up into `step_blocks` sized chunks
            remaining_steps = n_steps
            while remaining_steps > 0:
                probe_data = self._run_steps(self.step_blocks, **kwargs)
                remaining_steps -= self.step_blocks

                self._update_probe_data(
                    probe_data, self.n_steps - self.step_blocks,
                    self.step_blocks + min(remaining_steps, 0))

            # update n_steps/time
            self.n_steps += remaining_steps
            self.time = self.n_steps * self.dt

        print("\rSimulation completed in %s" %
              datetime.timedelta(seconds=int(time.time() - start)))

    def _run_steps(self, n_steps, profile=False, input_feeds=None):
        """Execute ``step_blocks`` sized segments of the simulation.

        Parameters
        ----------
        n_steps : int
            the number of simulation steps to be executed
        profile : bool, optional
            if True, collect Tensorflow profiling information while the
            simulation is running (this will slow down the simulation)
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            override the values of input Nodes with the given data.  arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.

        Notes
        -----
        - This function should not be called directly; run the simulator
          through :meth:`.Simulator.step`, :meth:`.Simulator.run_steps`, or
          :meth:`.Simulator.run`.

        - The ``input_feeds`` argument allows the user to pass several
          simultaneous input sequences through the model.  That is, instead of
          running the model ``n`` times with 1 input at a time, the model
          can be run once with ``n`` inputs at a time.  Only the values of
          input nodes (nodes with no incoming Connections) can be overwritten
          in this way.
        """

        if profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # execute the simulation loop
        try:
            final_step, probe_data, self.final_bases = self.sess.run(
                [self.tensor_graph.end_step, self.tensor_graph.probe_arrays,
                 self.tensor_graph.end_base_arrays],
                feed_dict=self._fill_feed(n_steps, input_feeds,
                                          start=self.n_steps),
                options=run_options, run_metadata=run_metadata)
        except tf.errors.InternalError as e:
            if e.op.type == "PyFunc":
                raise SimulationError(
                    "Function '%s' caused an error "
                    "(see error log above)" % e.op.name)
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

    def train(self, inputs, targets, optimizer, n_epochs=1, objective="mse",
              shuffle=True):
        """Optimize the trainable parameters of the network using the given
        optimization method, minimizing the objective value over the given
        inputs and targets.

        Parameters
        ----------
        inputs : dict of {:class:`~nengo:nengo.Node`: \
                          :class:`~numpy:numpy.ndarray`}
            input values for Nodes in the network; arrays should have shape
            ``(batch_size, sim.step_blocks, node.size_out)``
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}
            desired output value at Probes, corresponding to each value in
            ``inputs``; arrays should have shape
            ``(batch_size, sim.step_blocks, probe.size_in)``
        optimizer : ``tf.train.Optimizer``
            Tensorflow optimizer, e.g.
            ``tf.train.GradientDescentOptimizer(learning_rate=0.1)``
        n_epochs : int, optional
            run training for the given number of epochs (complete passes
            through ``inputs``)
        objective : ``"mse"`` or callable, optional
            the objective to be minimized. passing ``"mse"`` will train with
            mean squared error. a custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be averaged across Probes).
        shuffle : bool, optional
            if True, randomize the data into different minibatches each epoch

        Notes
        -----
        - Deep learning methods require the network to be differentiable, which
          means that trying to train a network with non-differentiable elements
          will result in an error.  Examples of common non-differentiable
          elements include :class:`~nengo:nengo.LIF`,
          :class:`~nengo:nengo.Direct`, or processes/neurons that don't have a
          custom Tensorflow implementation (see
          :class:`.processes.SimProcessBuilder`/
          :class:`.neurons.SimNeuronsBuilder`)

        - Most Tensorflow optimizers do not have GPU support for networks with
          sparse reads, which are a common element in Nengo models.  If your
          network contains sparse reads then training will have to be
          executed on the CPU (by creating the simulator via
          ``nengo_dl.Simulator(..., device="/cpu:0")``), or is limited to
          optimizers with GPU support (currently this is only
          ``tf.train.GradientDescentOptimizer``). Follow `this issue
          <https://github.com/tensorflow/tensorflow/issues/2314>`_ for updates
          on Tensorflow GPU support.
        """

        if self.closed:
            raise SimulatorClosed("Simulator cannot be trained because it is "
                                  "closed.")

        if not self.tensor_graph.unroll_simulation:
            raise SimulationError(
                "Simulation must be unrolled for training "
                "(`Simulator(..., unroll_simulation=True)`)")

        for n, x in inputs.items():
            if x.shape[1] != self.step_blocks:
                raise SimulationError(
                    "Length of input sequence (%s) does not match "
                    "`step_blocks` (%s)" % (x.shape[1], self.step_blocks))
            if x.shape[2] != n.size_out:
                raise SimulationError(
                    "Dimensionality of input sequence (%d) does not match "
                    "node.size_out (%d)" % (x.shape[2], n.size_out))

        for p, x in targets.items():
            if x.shape[1] != self.step_blocks:
                raise SimulationError(
                    "Length of target sequence (%s) does not match "
                    "`step_blocks` (%s)" % (x.shape[1], self.step_blocks))
            if x.shape[2] != p.size_in:
                raise SimulationError(
                    "Dimensionality of target sequence (%d) does not match "
                    "probe.size_in (%d)" % (x.shape[2], p.size_in))

        # check for non-differentiable elements in graph
        # utils.find_non_differentiable(
        #     [self.tensor_graph.invariant_ph[1]],
        #     [self.tensor_graph.probe_arrays[self.model.probes.index(p)]
        #      for p in targets])

        # build optimizer op
        self.tensor_graph.build_optimizer(optimizer, targets, objective)

        # initialize any variables that were created by the optimizer
        try:
            self.sess.run(self.tensor_graph.opt_slots_init)
        except tf.errors.InvalidArgumentError:
            raise SimulationError(
                "Tensorflow does not yet support this optimizer on the "
                "GPU; try `Simulator(..., device='/cpu:0')`")

        progress = utils.ProgressBar(n_epochs, "Training")

        for n in range(n_epochs):
            for inp, tar in utils.minibatch_generator(
                    inputs, targets, self.minibatch_size, rng=self.rng,
                    shuffle=shuffle):
                # TODO: set up queue to feed in data more efficiently
                self.sess.run(
                    [self.tensor_graph.opt_op],
                    feed_dict=self._fill_feed(self.step_blocks, inp, tar))
            progress.step()

    def loss(self, inputs, targets, objective=None):
        """Compute the loss value for the given objective and inputs/targets.

        Parameters
        ----------
        inputs : dict of {:class:`~nengo:nengo.Node`: \
                          :class:`~numpy:numpy.ndarray`}
            input values for Nodes in the network; arrays should have shape
            ``(batch_size, sim.step_blocks, node.size_out)``
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}
            desired output value at Probes, corresponding to each value in
            ``inputs``; arrays should have shape
            ``(batch_size, sim.step_blocks, probe.size_in)``
        objective : ``"mse"`` or callable, optional
            the objective used to compute loss. passing ``"mse"`` will use
            mean squared error. a custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be averaged across Probes). passing None will
            use the objective passed to the last call of
            :meth:`.Simulator.train`.
        """

        if self.closed:
            raise SimulatorClosed("Loss cannot be computed after simulator is "
                                  "closed.")

        for n, x in inputs.items():
            if x.shape[1] != self.step_blocks:
                raise SimulationError(
                    "Length of input sequence (%s) does not match "
                    "`step_blocks` (%s)" % (x.shape[1], self.step_blocks))
            if x.shape[2] != n.size_out:
                raise SimulationError(
                    "Dimensionality of input sequence (%d) does not match "
                    "node.size_out (%d)" % (x.shape[2], n.size_out))

        for p, x in targets.items():
            if x.shape[1] != self.step_blocks:
                raise SimulationError(
                    "Length of target sequence (%s) does not match "
                    "`step_blocks` (%s)" % (x.shape[1], self.step_blocks))
            if x.shape[2] != p.size_in:
                raise SimulationError(
                    "Dimensionality of target sequence (%d) does not match "
                    "probe.size_in (%d)" % (x.shape[2], p.size_in))

        # build optimizer op
        if objective is None:
            loss = self.tensor_graph.loss
        else:
            loss = self.tensor_graph.build_loss(objective, targets)

        loss_val = 0
        for i, (inp, tar) in enumerate(utils.minibatch_generator(
                inputs, targets, self.minibatch_size, rng=self.rng)):
            loss_val += self.sess.run(
                loss, feed_dict=self._fill_feed(self.step_blocks, inp, tar))
        loss_val /= i + 1

        return loss_val

    def _fill_feed(self, n_steps, inputs, targets=None, start=0):
        """Create a feed dictionary containing values for all the placeholder
        inputs in the network, which will be passed to ``tf.Session.run``.

        Parameters
        ----------
        n_steps : int
            the number of execution steps
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            override the values of input Nodes with the given data.  arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}, optional
            values for target placeholders (only necessary if loss is being
            computed, e.g. when training the network)
        start : int, optional
            initial value of simulator timestep

        Returns
        -------
        dict of {``tf.Tensor``: :class:`~numpy:numpy.ndarray`}
            feed values for placeholder tensors in the network
        """

        # fill in loop variables
        feed_dict = {
            self.tensor_graph.step_var: start,
            self.tensor_graph.stop_var: start + n_steps
        }

        # fill in values for base variables from previous run
        feed_dict.update(
            {k: v for k, v in zip(
                self.tensor_graph.base_vars,
                self.final_bases) if k.op.type == "Placeholder"})

        # fill in input values
        tmp = self._generate_inputs(inputs, n_steps)
        feed_dict.update(tmp)

        # fill in target values
        if targets is not None:
            feed_dict.update(
                {self.tensor_graph.target_phs[p]: np.moveaxis(t, 0, -1)
                 for p, t in targets.items()})

        return feed_dict

    def _generate_inputs(self, input_feeds, n_steps):
        """Generate inputs for the network (the output values of each Node with
        no incoming connections).

        Parameters
        ----------
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            override the values of input Nodes with the given data.  arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        n_steps : int
            number of simulation timesteps for which to generate input data
        """

        if input_feeds is None:
            input_feeds = {}
        else:
            # validate inputs
            for n, v in input_feeds.items():
                target_shape = (self.minibatch_size, n_steps, n.size_out)
                if v.shape != target_shape:
                    raise SimulationError(
                        "Input feed for node %s has wrong shape; expected %s, "
                        "saw %s" % (n, target_shape, v.shape))

        feed_vals = {}
        for n in self.tensor_graph.invariant_inputs:
            # if the output signal is not in sig map, that means no operators
            # use the output of this node. similarly, if node.size_out is 0,
            # the node isn't producing any output values.
            using_output = (
                self.model.sig[n]["out"] in self.tensor_graph.sig_map and
                n.size_out > 0)

            if using_output:
                if n in input_feeds:
                    # move minibatch dimension to the end
                    feed_val = np.moveaxis(input_feeds[n], 0, -1)
                elif isinstance(n.output, np.ndarray):
                    feed_val = np.tile(n.output[None, :, None],
                                       (n_steps, 1, self.minibatch_size))
                else:
                    func = self.tensor_graph.invariant_funcs[n]

                    feed_val = []
                    for i in range(self.n_steps + 1,
                                   self.n_steps + n_steps + 1):
                        # note: need to copy the output of func, as func
                        # may mutate its outputs in-place on subsequent calls
                        feed_val += [np.array(func(i * self.dt))]

                    feed_val = np.stack(feed_val, axis=0)
                    feed_val = np.tile(feed_val[..., None],
                                       (1, 1, self.minibatch_size))

                feed_vals[self.tensor_graph.invariant_ph[n]] = feed_val
            elif (not isinstance(n.output, np.ndarray) and
                  n.output in self.tensor_graph.invariant_funcs.values()):
                # note: we still call the function even if the output
                # is not being used, because it may have side-effects
                func = self.tensor_graph.invariant_funcs[n]
                for i in range(self.n_steps + 1, self.n_steps + n_steps + 1):
                    func(i * self.dt)

        return feed_vals

    def _update_probe_data(self, probe_data, start, n_steps):
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

    def save_params(self, path):
        """Save trainable network parameters to the given ``path``.

        Parameters
        ----------
        path : str
            filepath of parameter output file
        """
        if self.closed:
            raise SimulationError("Simulation has been closed, cannot save "
                                  "parameters")

        path = tf.train.Saver().save(self.sess, path)
        print("Model parameters saved to %s" % path)

    def load_params(self, path):
        """Load trainable network parameters from the given ``path``.

        Parameters
        ----------
        path : str
            filepath of parameter input file
        """
        if self.closed:
            raise SimulationError("Simulation has been closed, cannot load "
                                  "parameters")

        tf.train.Saver().restore(self.sess, path)

    def print_params(self, msg=None):
        """Print current values of trainable network parameters.

        Parameters
        ----------
        msg : str, optional
            title for print output, useful to differentiate multiple print
            calls
        """

        if self.closed:
            raise SimulationError("Simulation has been closed, cannot print "
                                  "parameters")

        params = {k: v for k, v in self.tensor_graph.signals.bases.items()
                  if v.dtype._is_ref_dtype}
        param_sigs = {k: v for k, v in self.tensor_graph.sig_map.items()
                      if k.trainable}

        param_vals = self.sess.run(params)

        print("%s:" % "Parameters" if msg is None else msg)
        for sig, tens in param_sigs.items():
            print("-" * 10)
            print(sig)
            print(param_vals[tens.key][tens.indices])

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
        """Raise a RuntimeWarning if the Simulator is deallocated while open.
        """

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

    def check_gradients(self, atol=1e-4, rtol=1e-3):
        """Perform gradient checks for the network (used to verify that the
        analytic gradients are correct).

        Raises a simulation error if the difference between analytic and
        numeric gradient is greater than ``atol + rtol * numeric_grad``
        (elementwise).

        Parameters
        ----------
        atol : float, optional
            absolute error tolerance
        rtol : float, optional
            relative (to numeric grad) error tolerance
        """

        feed = self._fill_feed(
            self.step_blocks, {n: np.zeros((self.minibatch_size,
                                            self.step_blocks, n.size_out))
                               for n in self.tensor_graph.invariant_inputs},
            {p: np.zeros((self.minibatch_size, self.step_blocks, p.size_in))
             for p in self.model.probes})

        # check gradient wrt inp
        for node, inp in self.tensor_graph.invariant_ph.items():
            analytic, numeric = tf.test.compute_gradient(
                inp, inp.get_shape().as_list(), self.tensor_graph.loss, (1,),
                x_init_value=np.zeros(inp.get_shape().as_list()),
                extra_feed_dict=feed)
            if np.any(np.isnan(analytic)) or np.any(np.isnan(numeric)):
                raise SimulationError("NaNs detected in gradient")
            if np.any(abs(analytic - numeric) >= atol + rtol * abs(numeric)):
                raise SimulationError(
                    "Gradient check failed for input %s\n"
                    "numeric gradient:\n%s\n"
                    "analytic gradient:\n%s\n" % (node, numeric, analytic))


class ProbeDict(Mapping):
    """Map from :class:`~nengo:nengo.Probe` -> :class:`~numpy:numpy.ndarray`,
    used to access output of the model after simulation.

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.

    Parameters
    ----------
    raw : dict of {:class:`~nengo:nengo.Probe`: \
                   list of :class:`~numpy:numpy.ndarray`}
        the raw probe output from the simulator (a list of arrays containing
        the output from each ``step_blocks`` execution segment)
    minibatches : dict of {:class:`~nengo:nengo.Probe`: int or None}
        the minibatch size for each probe in the dictionary (or -1 if the
        probed signal does not have a minibatch dimension)

    Notes
    -----
    ProbeDict should never be created/accessed directly by the user, but rather
    via ``sim.data`` (which is an instance of ProbeDict).
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
