from __future__ import print_function, division

import collections
import datetime
import logging
import os
import sys
import tempfile
import time
import warnings

from nengo import Process, Ensemble, Connection, Probe, Network
from nengo import version as nengo_version
from nengo.builder import Model
from nengo.builder.connection import BuiltConnection
from nengo.builder.ensemble import BuiltEnsemble
from nengo.ensemble import Neurons
from nengo.exceptions import (
    ReadonlyError, SimulatorClosed, NengoWarning, SimulationError,
    ValidationError, ConfigError)
from nengo.solvers import NoSolver
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradient_checker

from nengo_dl import utils, DATA_DIR
from nengo_dl.builder import NengoBuilder
from nengo_dl.tensor_graph import TensorGraph

logger = logging.getLogger(__name__)

if sys.version_info < (3, 4):
    import backports.tempfile as tempfile  # noqa: F811


class Simulator(object):
    """Simulate network using the ``nengo_dl`` backend.

    Parameters
    ----------
    network : :class:`~nengo:nengo.Network` or None
        A network object to be built and then simulated. If None,
        then a built model must be passed to ``model`` instead
    dt : float, optional
        Length of a simulator timestep, in seconds
    seed : int, optional
        Seed for all stochastic operators used in this simulator
    model : :class:`~nengo:nengo.builder.Model`, optional
        Pre-built model object
    dtype : ``tf.DType``, optional
        Floating point precision to use for simulation
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``, optional
        Device on which to execute computations (if None then uses the
        default device as determined by TensorFlow)
    unroll_simulation : int, optional
        Unroll simulation loop by explicitly building the given number of
        iterations into the computation graph (improves simulation speed
        but increases build time)
    minibatch_size : int, optional
        The number of simultaneous inputs that will be passed through the
        network
    tensorboard : str, optional
        If not None, save network output in the TensorFlow summary format to
        the given directory, which can be loaded into TensorBoard
    progress_bar : bool, optional
        If True (default), display progress information when building a model
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

        ("nengo/tests/test_simulator.py:test_simulator_progress_bars",
         "nengo_dl uses a different progress bar system (see "
         "tests/test_utils.py:test_progress_bar"),

        ("nengo/tests/test_node.py:test_args",
         "time is passed as np.float32, not a float (see "
         "tests/test_nengo_tests.py:test_args"),

        ("nengo/tests/test_node.py:test_unconnected_node",
         "need to set `unroll_simulation` to ensure node runs the correct "
         "number of times (see "
         "tests/test_nengo_tests.py:test_unconnected_node"),

        ("nengo/tests/test_synapses.py:test_alpha",
         "need to set looser tolerances due to float32 implementation (see "
         "tests/test_processes.py:test_alpha"),

        ("nengo/tests/test_ensemble.py:test_gain_bias",
         "use allclose instead of array_equal (see "
         "tests/test_simulator.py:test_gain_bias")
    ]

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 dtype=tf.float32, device=None, unroll_simulation=1,
                 minibatch_size=None, tensorboard=None, progress_bar=True):
        self.closed = False
        self.unroll = unroll_simulation
        self.minibatch_size = 1 if minibatch_size is None else minibatch_size
        self.data = SimulationData(self, minibatch_size is not None)
        self.seed = (np.random.randint(np.iinfo(np.int32).max) if seed is None
                     else seed)

        # TODO: multi-GPU support

        ProgressBar = (utils.ProgressBar if progress_bar else
                       utils.NullProgressBar)

        # build model (uses default nengo builder)
        if model is None:
            self.model = Model(dt=float(dt), label="%s, dt=%f" % (network, dt),
                               builder=NengoBuilder())
        else:
            if dt != model.dt:
                warnings.warn("Model dt (%g) does not match Simulator "
                              "dt (%g)" % (model.dt, dt), NengoWarning)
            self.model = model

        if network is not None:
            if nengo_version.version_info < (2, 7, 1):
                print("Building network")
                start = time.time()
                self.model.build(network, progress_bar=None)
                print("\rBuild finished in %s " %
                      datetime.timedelta(seconds=int(time.time() - start)))
            else:
                p = ProgressBar("Building network", "Build")
                self.model.build(network, progress=p)

        # set up tensorflow graph plan
        with ProgressBar("Optimizing graph", "Optimization",
                         max_value=None) as progress:
            self.tensor_graph = TensorGraph(
                self.model, self.dt, unroll_simulation, dtype,
                self.minibatch_size, device, progress)

        # construct graph
        with ProgressBar("Constructing graph", "Construction",
                         max_value=None) as progress:
            self.tensor_graph.build(progress)

        # output simulation data for viewing via TensorBoard
        if tensorboard is not None:
            if not os.path.exists(tensorboard):
                os.makedirs(tensorboard)

            run_number = max(
                [int(x[4:]) for x in os.listdir(tensorboard)
                 if x.startswith("run")] or [-1]) + 1
            self.summary = tf.summary.FileWriter(
                os.path.join(tensorboard, "run_%d" % run_number),
                graph=self.tensor_graph.graph)
        else:
            self.summary = None

        # start session

        config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
        )

        # set any config options specified by user
        try:
            config_settings = network.config[Network].session_config
        except (ConfigError, AttributeError):
            config_settings = {}
        for c, v in config_settings.items():
            attrs = c.split(".")
            x = config
            for a in attrs[:-1]:
                x = getattr(x, a)
            setattr(x, attrs[-1], v)

        # TODO: XLA compiling doesn't seem to provide any benefit at the
        # moment, revisit later after tensorflow has developed it further
        # config.graph_options.optimizer_options.global_jit_level = (
        #     tf.OptimizerOptions.ON_1)

        self.sess = tf.Session(graph=self.tensor_graph.graph,
                               config=config)

        self.reset(seed=seed)

    def reset(self, seed=None):
        """Resets the simulator to initial conditions.

        Parameters
        ----------
        seed : int, optional
            If not None, overwrite the default simulator seed with this value
            (note: this becomes the new default simulator seed)
        """

        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        self.input_funcs = {}
        self.n_steps = 0
        self.time = 0.0

        # initialize variables
        self.sess.run(self.tensor_graph.constant_init_op,
                      feed_dict=self.tensor_graph.signals.constant_phs)
        self.soft_reset(include_trainable=True, include_probes=True)

        # execute post-build processes (we do this here because
        # seed can change each call to reset)
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        tf.set_random_seed(self.seed)

        self.tensor_graph.build_post(self.sess, self.rng)

    def soft_reset(self, include_trainable=False, include_probes=False):
        """Resets the internal state of the simulation, but doesn't
        rebuild the graph.

        Parameters
        ----------
        include_trainable : bool, optional
            If True, also reset any training that has been performed on
            network parameters (e.g., connection weights)
        include_probes : bool, optional
            If True, also clear probe data
        """

        init_ops = [self.tensor_graph.local_init_op,
                    self.tensor_graph.global_init_op]
        if include_trainable:
            init_ops.append(self.tensor_graph.trainable_init_op)
        self.sess.run(init_ops, feed_dict={
            ph: v for _, ph, v in self.tensor_graph.base_vars.values()})

        if include_probes:
            for p in self.model.probes:
                self.model.params[p] = []
            self.n_steps = 0

    def step(self, **kwargs):
        """Run the simulation for one time step.

        Parameters
        ----------
        kwargs : dict
            See :meth:`.run_steps`

        Notes
        -----
        Progress bar is disabled by default when running via this method.
        """

        kwargs.setdefault("progress_bar", False)

        self.run_steps(1, **kwargs)

    def run(self, time_in_seconds, **kwargs):
        """Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Run the simulator for the given number of simulated seconds
        kwargs : dict
            See :meth:`.run_steps`
        """

        if time_in_seconds < 0:
            raise ValidationError(
                "Must be positive (got %g)" % (time_in_seconds,),
                attr="time_in_seconds")

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn("%g results in running for 0 timesteps. Simulator "
                          "still at time %g." % (time_in_seconds, self.time))
        else:
            self.run_steps(steps, **kwargs)

    def run_steps(self, n_steps, input_feeds=None, profile=False,
                  progress_bar=True, extra_feeds=None):
        """Simulate for the given number of steps.

        Parameters
        ----------
        n_steps : int
            The number of simulation steps to be executed
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            Override the values of input Nodes with the given data.  Arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        profile : bool, optional
            If True, collect TensorFlow profiling information while the
            simulation is running (this will slow down the simulation).
            Can also pass a dict of `config options for the profiler
            <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md>`__.
        progress_bar : bool, optional
            If True, print information about the simulation status to standard
            output.
        extra_feeds : dict of {``tf.Tensor``: :class:`~numpy:numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)

        Notes
        -----
        If ``unroll_simulation=x`` is specified, and ``n_steps > x``, this will
        repeatedly execute ``x`` timesteps until the the number of steps
        executed is >= ``n_steps``.
        """

        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        actual_steps = self.unroll * int(np.ceil(n_steps / self.unroll))

        if actual_steps != n_steps:
            warnings.warn(
                "Number of steps (%d) is not an even multiple of "
                "`unroll_simulation` (%d).  Simulation will run for %d steps, "
                "which may have unintended side effects." %
                (n_steps, self.unroll, actual_steps), RuntimeWarning)

        if input_feeds is not None:
            self._check_data(input_feeds, mode="input",
                             n_batch=self.minibatch_size, n_steps=n_steps)

        feed = self._fill_feed(actual_steps, input_feeds,
                               start=self.n_steps)
        if extra_feeds is not None:
            feed.update(extra_feeds)

        if profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        progress = (
            utils.ProgressBar("Simulating", "Simulation", max_value=None)
            if progress_bar else utils.NullProgressBar())

        # execute the simulation loop
        with progress:
            try:
                steps_run, probe_data = self.sess.run(
                    [self.tensor_graph.steps_run,
                     list(self.tensor_graph.probe_arrays.values())],
                    feed_dict=feed, options=run_options,
                    run_metadata=run_metadata)
            except (tf.errors.InternalError, tf.errors.UnknownError) as e:
                if e.op is not None and e.op.type == "PyFunc":
                    raise SimulationError(
                        "Function '%s' caused an error (see error log above)" %
                        e.op.name)
                else:
                    raise e  # pragma: no cover

            # update probe data
            self._update_probe_data(probe_data, self.n_steps, n_steps)

            # update n_steps
            # note: we update n_steps according to the number of steps that the
            # user asked for, not the number of steps that were actually run (
            # in the case of uneven unroll_simulation)
            assert steps_run == actual_steps
            self.n_steps += n_steps
            self.time = self.n_steps * self.dt

        if profile:
            filename = os.path.join(DATA_DIR, "nengo_dl_profile.json")
            options = tf.profiler.ProfileOptionBuilder.time_and_memory()
            options["output"] = "timeline:outfile=%s" % filename
            options["min_bytes"] = 0
            if isinstance(profile, dict):
                options.update(profile)
            tf.profiler.profile(
                self.tensor_graph.graph, run_meta=run_metadata,
                cmd="scope", options=options)

    def train(self, inputs, targets, optimizer, n_epochs=1, objective="mse",
              shuffle=True, truncation=None, summaries=None, profile=False,
              extra_feeds=None):
        """
        Optimize the trainable parameters of the network using the given
        optimization method, minimizing the objective value over the given
        inputs and targets.

        Parameters
        ----------
        inputs : dict of {:class:`~nengo:nengo.Node`: \
                          :class:`~numpy:numpy.ndarray`}
            Input values for Nodes in the network; arrays should have shape
            ``(batch_size, n_steps, node.size_out)``
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}
            Desired output value at Probes, corresponding to each value in
            ``inputs``; arrays should have shape
            ``(batch_size, n_steps, probe.size_in)``
        optimizer : ``tf.train.Optimizer``
            TensorFlow optimizer, e.g.
            ``tf.train.GradientDescentOptimizer(learning_rate=0.1)``
        n_epochs : int, optional
            Run training for the given number of epochs (complete passes
            through ``inputs``)
        objective : ``"mse"`` or callable or ``None``, optional
            The objective to be minimized. Passing ``"mse"`` will train with
            mean squared error. A custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be summed across Probes).  Passing ``None``
            indicates that the error is being computed outside the simulation,
            and the value passed to ``targets`` directly specifies the output
            error gradient. Note that by default the same objective will be
            used for all probes in ``targets``; a dictionary of
            ``{probe: obj, ...}`` can be passed for ``objective`` to specify a
            different objective for each probe.
        shuffle : bool, optional
            If True, randomize the data into different minibatches each epoch
        truncation: int, optional
            If not None, use truncated backpropagation when training the
            network, with the given truncation length.
        summaries : list of :class:`~nengo:nengo.Connection` or \
                            :class:`~nengo:nengo.Ensemble` or \
                            :class:`~nengo:nengo.ensemble.Neurons` or \
                            ``"loss"`` or \
                            ``tf.Tensor``}
            If not None, collect data during the training process using
            TensorFlow's ``tf.summary`` format.  The summary objects can be a
            Connection (in which case data on the corresponding weights will be
            collected), Ensemble (encoders), Neurons (biases), or ``"loss"``
            (the loss value for ``objective``).  The user can also create their
            own summaries and pass in the Tensors representing the summary ops.
        profile : bool, optional
            If True, collect TensorFlow profiling information while training
            (this will slow down the training).  Can also pass a dict of
            `config options for the profiler
            <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md>`__.
        extra_feeds : dict of {``tf.Tensor``: :class:`~numpy:numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)

        Notes
        -----
        Most deep learning methods require the network to be differentiable,
        which means that trying to train a network with non-differentiable
        elements will result in an error.  Examples of common
        non-differentiable elements include :class:`~nengo:nengo.LIF`,
        :class:`~nengo:nengo.Direct`, or processes/neurons that don't have a
        custom TensorFlow implementation (see
        :class:`.process_builders.SimProcessBuilder`/
        :class:`.neuron_builders.SimNeuronsBuilder`)
        """

        batch_size, n_steps = next(iter(inputs.values())).shape[:2]

        # error checking
        if self.closed:
            raise SimulatorClosed("Simulator cannot be trained because it is "
                                  "closed.")
        self._check_data(inputs, mode="input")
        self._check_data(targets, mode="target", n_steps=n_steps,
                         n_batch=batch_size)
        if n_steps % self.unroll != 0:
            raise ValidationError(
                "The number of timesteps in training data must be evenly "
                "divisible by unroll_simulation", "inputs")
        if truncation is not None and truncation % self.unroll != 0:
            raise ValidationError(
                "Truncation length must be evenly divisible by "
                "unroll_simulation", "inputs")

        # check for non-differentiable elements in graph
        # utils.find_non_differentiable(
        #     [self.tensor_graph.input_ph[n] for n in inputs],
        #     [self.tensor_graph.probe_arrays[self.model.probes.index(p)]
        #      for p in targets])

        # apply objective to all probes if individual objectives weren't given
        if not isinstance(objective, dict):
            objective = {p: objective for p in targets}

        # build optimizer op
        opt_op, opt_slots_init = self.tensor_graph.build_optimizer(
            optimizer, objective)
        fetches = [opt_op]

        # initialize any variables that were created by the optimizer
        if opt_slots_init is not None:
            self.sess.run(opt_slots_init)

        # increment training step
        fetches.append(self.tensor_graph.training_step_inc)

        # get loss op
        loss = self.tensor_graph.build_loss(objective)
        if loss is not None:
            fetches.append(loss)

        # add summaries
        summary_op = None
        if summaries is not None:
            if self.summary is None:
                warnings.warn("Simulator was created with tensorboard=False; "
                              "ignoring requested summaries")
            else:
                for i, v in enumerate(summaries):
                    if isinstance(v, str) and v == "loss":
                        summaries[i] = objective
                summary_op = self.tensor_graph.build_summaries(summaries)
                fetches.append(summary_op)

        # save the internal state of the simulator
        tmpdir = tempfile.TemporaryDirectory()
        self.save_params(os.path.join(tmpdir.name, "tmp"), include_local=True,
                         include_global=False)

        if profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            profiler = tf.profiler.Profiler(self.tensor_graph.graph)
        else:
            run_options = None
            run_metadata = None

        progress = utils.ProgressBar(
            "Training", max_value=n_epochs * batch_size // self.minibatch_size,
            vars=["loss"])

        # run training
        with progress:
            for n in range(n_epochs):
                for offset, inp, tar in utils.minibatch_generator(
                        inputs, targets, self.minibatch_size, rng=self.rng,
                        shuffle=shuffle, truncation=truncation):
                    if offset == 0:
                        self.soft_reset()

                    steps = next(iter(inp.values())).shape[1]
                    feed = self._fill_feed(steps, inp, tar, start=offset)
                    if extra_feeds is not None:
                        feed.update(extra_feeds)
                    outputs = self.sess.run(
                        fetches, feed_dict=feed,
                        options=run_options, run_metadata=run_metadata)

                    if summary_op is not None:
                        self.summary.add_summary(outputs[-1], outputs[1])

                    if profile:
                        profiler.add_step(int(outputs[1]), run_metadata)

                    if offset == 0:
                        progress.step(loss="%.4f" % (
                            np.nan if loss is None else outputs[2]))

        # restore internal state of simulator
        self.load_params(os.path.join(tmpdir.name, "tmp"), include_local=True,
                         include_global=False)
        tmpdir.cleanup()

        if profile:
            filename = os.path.join(DATA_DIR, "nengo_dl_profile.json")
            options = tf.profiler.ProfileOptionBuilder.time_and_memory()
            options["output"] = "timeline:outfile=%s" % filename
            options["min_bytes"] = 0
            if isinstance(profile, dict):
                options.update(profile)
            profiler.profile_name_scope(options)

    def loss(self, inputs, targets, objective, extra_feeds=None):
        """
        Compute the loss value for the given objective and inputs/targets.

        Parameters
        ----------
        inputs : dict of {:class:`~nengo:nengo.Node`: \
                          :class:`~numpy:numpy.ndarray`}
            Input values for Nodes in the network; arrays should have shape
            ``(batch_size, n_steps, node.size_out)``
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}
            Desired output value at Probes, corresponding to each value in
            ``inputs``; arrays should have shape
            ``(batch_size, n_steps, probe.size_in)``
        objective : ``"mse"`` or callable
            The objective used to compute loss. Passing ``"mse"`` will use
            mean squared error. A custom function
            ``f(output, target) -> loss`` can be passed that consumes the
            actual output and target output for a probe in ``targets``
            and returns a ``tf.Tensor`` representing the scalar loss value for
            that Probe (loss will be summed across Probes). Note that by
            default the same objective will be used for all probes in
            ``targets``; a dictionary of ``{probe: obj, ...}`` can be passed
            for ``objective`` to specify a different objective for each probe.
        extra_feeds : dict of {``tf.Tensor``: :class:`~numpy:numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)
        """

        batch_size, n_steps = next(iter(inputs.values())).shape[:2]

        # error checking
        if self.closed:
            raise SimulatorClosed("Loss cannot be computed after simulator is "
                                  "closed.")
        self._check_data(inputs, mode="input")
        self._check_data(targets, mode="target", n_steps=n_steps,
                         n_batch=batch_size)
        if n_steps % self.unroll != 0:
            raise ValidationError(
                "The number of timesteps in loss data must be evenly "
                "divisible by unroll_simulation", "inputs")

        # apply objective to all probes if individual objectives weren't given
        if not isinstance(objective, dict):
            objective = {p: objective for p in targets}

        # get loss op
        loss = self.tensor_graph.build_loss(objective)

        # save the internal state of the simulator
        tmpdir = tempfile.TemporaryDirectory()
        self.save_params(os.path.join(tmpdir.name, "tmp"), include_local=True,
                         include_global=False)

        # compute loss on data
        loss_val = 0
        for i, (_, inp, tar) in enumerate(utils.minibatch_generator(
                inputs, targets, self.minibatch_size, rng=self.rng)):
            self.soft_reset()
            feed = self._fill_feed(n_steps, inp, tar)
            if extra_feeds is not None:
                feed.update(extra_feeds)
            loss_val += self.sess.run(loss, feed_dict=feed)
        loss_val /= i + 1

        # restore internal state of simulator
        self.load_params(os.path.join(tmpdir.name, "tmp"), include_local=True,
                         include_global=False)
        tmpdir.cleanup()

        return loss_val

    def save_params(self, path, include_global=True, include_local=False):
        """Save network parameters to the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter output file
        include_global : bool, optional
            If True (default True), save global/trainable network variables
        include_local : bool, optional
            If True (default False), save local (non-trainable) network
            variables

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        :meth:`.get_nengo_params`.
        """
        if self.closed:
            raise SimulationError("Simulation has been closed, cannot save "
                                  "parameters")

        with self.tensor_graph.graph.as_default():
            vars = []
            if include_global:
                vars.extend(tf.global_variables())
            if include_local:
                vars.extend(tf.local_variables())

            with tf.device("/cpu:0"):
                path = tf.train.Saver(vars).save(self.sess, path)

        logger.info("Model parameters saved to %s", path)

    def load_params(self, path, include_global=True, include_local=False):
        """Load network parameters from the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter input file
        include_global : bool, optional
            If True (default True), load global (trainable) network variables
        include_local : bool, optional
            If True (default False), load local (non-trainable) network
            variables

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        :meth:`.get_nengo_params`.
        """
        if self.closed:
            raise SimulationError("Simulation has been closed, cannot load "
                                  "parameters")

        with self.tensor_graph.graph.as_default():
            vars = []
            if include_global:
                vars.extend(tf.global_variables())
            if include_local:
                vars.extend(tf.local_variables())

            with tf.device("/cpu:0"):
                tf.train.Saver(vars).restore(self.sess, path)

        logger.info("Model parameters loaded from %s", path)

    def get_nengo_params(self, nengo_objs, as_dict=False):
        """Extract model parameters in a form that can be used to initialize
        Nengo objects in a different model.

        For example:

        .. code-block:: python

            with nengo.Network() as net:
                a = nengo.Ensemble(10, 1)
                b = nengo.Ensemble(10, 1)
                c = nengo.Connection(a, b)

            with nengo_dl.Simulator(net) as sim:
                # < do some optimization >
                params = sim.get_nengo_params([a, b, c])

            with nengo.Network() as new_net:
                # < build some other network >

                # now we want to insert two connected ensembles with the same
                # parameters as our previous network:
                d = nengo.Ensemble(10, 1, **params[0])
                e = nengo.Ensemble(10, 1, **params[1])
                f = nengo.Connection(d, e, **params[2])

        Parameters
        ----------
        nengo_objs : (list of) :class:`~nengo:nengo.Ensemble` or \
                               :class:`~nengo:nengo.Connection`
            A single object or list of objects for which we want to get the
            parameters.
        as_dict : bool, optional
            If True, return the values as a dictionary keyed by object label,
            instead of a list (the default).  Note that in this case labels
            must be unique.

        Returns
        -------
        (list or dict) of dicts
            kwarg dicts corresponding to ``nengo_objs`` (passing these
            dicts as kwargs when creating new Nengo objects will result in a
            new object with the same parameters as the source object).  A
            single kwarg dict if a single object was passed in, or a list
            (dict if ``as_dict=True``) of kwargs corresponding to multiple
            input objects.
        """

        if isinstance(nengo_objs, (list, tuple)):
            scalar = False
        else:
            scalar = True
            nengo_objs = [nengo_objs]

        # convert neurons to the parent ensemble
        nengo_objs = [obj.ensemble if isinstance(obj, Neurons) else obj
                      for obj in nengo_objs]

        params = []
        for obj in nengo_objs:
            data = self.data[obj]
            if isinstance(obj, Connection):
                if isinstance(obj.pre_obj, Ensemble):
                    params.append({
                        "solver": NoSolver(data.weights.T, weights=False),
                        "function": lambda x, data=data: np.zeros(
                            data.weights.shape[0]),
                        "transform": 1})
                else:
                    params.append({"transform": data.weights})
            elif isinstance(obj, Ensemble):
                params.append({"gain": data.gain, "bias": data.bias,
                               "encoders": data.encoders})
            else:
                raise ValueError(
                    "Can only get Nengo parameters for Ensembles or "
                    "Connections")

        if scalar:
            return params[0]

        if as_dict:
            param_dict = {}
            for obj, p in zip(nengo_objs, params):
                if obj.label in param_dict:
                    raise ValueError(
                        "Duplicate label ('%s') detected; cannot return "
                        "parameters with as_dict=True" % obj.label)
                else:
                    param_dict[obj.label] = p
            params = param_dict

        return params

    def check_gradients(self, outputs=None, atol=1e-5, rtol=1e-3):
        """Perform gradient checks for the network (used to verify that the
        analytic gradients are correct).

        Raises a simulation error if the difference between analytic and
        numeric gradient is greater than ``atol + rtol * numeric_grad``
        (elementwise).

        Parameters
        ----------
        outputs : ``tf.Tensor`` or list of ``tf.Tensor`` or \
                  list of :class:`~nengo:nengo.Probe`
            Compute gradients wrt this output (if None, computes wrt each
            output probe)
        atol : float, optional
            Absolute error tolerance
        rtol : float, optional
            Relative (to numeric grad) error tolerance

        Notes
        -----
        Calling this function will reset all values in the network, so it
        should not be intermixed with calls to :meth:`.Simulator.run`.
        """

        delta = 1e-3
        n_steps = self.unroll * 2

        feed = self._fill_feed(
            n_steps, {n: np.zeros((self.minibatch_size, n_steps, n.size_out))
                      for n in self.tensor_graph.invariant_inputs},
            {p: np.zeros((self.minibatch_size, n_steps, p.size_in))
             for p in self.tensor_graph.target_phs})

        if outputs is None:
            # note: the x + 0 is necessary because `gradient_checker`
            # doesn't work properly if the output variable is a tensorarray
            outputs = [x + 0 for x in self.tensor_graph.probe_arrays.values()]
        elif isinstance(outputs, tf.Tensor):
            outputs = [outputs]
        else:
            outputs = [self.tensor_graph.probe_arrays[p] + 0 for p in outputs]

        # check gradient wrt inp
        for node, inp in self.tensor_graph.input_ph.items():
            inp_shape = inp.get_shape().as_list()
            inp_shape = [n_steps if x is None else x for x in inp_shape]
            inp_tens = self.tensor_graph.input_ph[node]
            feed[inp_tens] = np.ascontiguousarray(feed[inp_tens])
            inp_val = np.ravel(feed[inp_tens])
            for out in outputs:
                out_shape = out.get_shape().as_list()
                out_shape = [n_steps if x is None else x for x in out_shape]

                # we need to compute the numeric jacobian manually, to
                # correctly handle variables (tensorflow doesn't expect
                # state ops in `compute_gradient`, because it doesn't define
                # gradients for them)
                numeric = np.zeros((np.prod(inp_shape, dtype=np.int32),
                                    np.prod(out_shape, dtype=np.int32)))

                for i in range(numeric.shape[0]):
                    self.soft_reset()
                    inp_val[i] = delta
                    plus = self.sess.run(out, feed_dict=feed)

                    self.soft_reset()
                    inp_val[i] = -delta
                    minus = self.sess.run(out, feed_dict=feed)

                    numeric[i] = np.ravel((plus - minus) / (2 * delta))

                    inp_val[i] = 0

                self.soft_reset()

                dx, dy = gradient_checker._compute_dx_and_dy(
                    inp, out, out_shape)

                with self.sess.as_default():
                    analytic = gradient_checker._compute_theoretical_jacobian(
                        inp, inp_shape, np.zeros(inp_shape), dy, out_shape, dx,
                        extra_feed_dict=feed)

                if np.any(np.isnan(analytic)) or np.any(np.isnan(numeric)):
                    raise SimulationError("NaNs detected in gradient")
                fail = abs(analytic - numeric) >= atol + rtol * abs(numeric)
                if np.any(fail):
                    raise SimulationError(
                        "Gradient check failed for input %s and output %s\n"
                        "numeric values:\n%s\n"
                        "analytic values:\n%s\n" % (node, out, numeric[fail],
                                                    analytic[fail]))

        self.soft_reset()

        logger.info("Gradient check passed")

    def trange(self, dt=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        dt : float, optional
            The sampling period of the probe to create a range for;
            if None, the simulator's ``dt`` will be used.
        """

        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def close(self):
        """Close the simulation, freeing resources.

        Notes
        -----
        The simulation cannot be restarted after it is closed.  This is not a
        technical limitation, just a design decision made for all Nengo
        simulators.
        """

        if not self.closed:
            # note: we use getattr in case it crashes before the object is
            # created
            if getattr(self, "sess", None) is not None:
                self.sess.close()
            self.sess = None

            if getattr(self, "summary", None) is not None:
                self.summary.close()

            self.closed = True

    def _fill_feed(self, n_steps, inputs, targets=None, start=0):
        """Create a feed dictionary containing values for all the placeholder
        inputs in the network, which will be passed to ``tf.Session.run``.

        Parameters
        ----------
        n_steps : int
            The number of execution steps
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            Override the values of input Nodes with the given data.  Arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        targets : dict of {:class:`~nengo:nengo.Probe`: \
                           :class:`~numpy:numpy.ndarray`}, optional
            Values for target placeholders (only necessary if loss is being
            computed, e.g. when training the network)
        start : int, optional
            Initial value of simulator timestep

        Returns
        -------
        dict of {``tf.Tensor``: :class:`~numpy:numpy.ndarray`}
            Feed values for placeholder tensors in the network
        """

        # fill in loop variables
        feed_dict = {
            self.tensor_graph.step_var: start,
            self.tensor_graph.stop_var: start + n_steps
        }

        # fill in input values
        tmp = self._generate_inputs(inputs, n_steps)
        feed_dict.update(tmp)

        # fill in target values
        if targets is not None:
            feed_dict.update(
                {self.tensor_graph.target_phs[p]: t
                 for p, t in targets.items()})

        return feed_dict

    def _generate_inputs(self, input_feeds, n_steps):
        """Generate inputs for the network (the output values of each Node with
        no incoming connections).

        Parameters
        ----------
        input_feeds : dict of {:class:`~nengo:nengo.Node`: \
                               :class:`~numpy:numpy.ndarray`}
            Override the values of input Nodes with the given data.  Arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        n_steps : int
            Number of simulation timesteps for which to generate input data

        Returns
        -------
        dict of {:class:`~nengo:nengo.Node`: :class:`~numpy:numpy.ndarray}
            Simulation values for all the input Nodes in the network.
        """

        if input_feeds is None:
            input_feeds = {}

        feed_vals = {}
        for n in self.tensor_graph.invariant_inputs:
            # if the output signal is not in sig map, that means no operators
            # use the output of this node. similarly, if node.size_out is 0,
            # the node isn't producing any output values.
            using_output = (
                self.model.sig[n]["out"] in self.tensor_graph.sig_map and
                n.size_out > 0)

            if (not isinstance(n.output, np.ndarray) and
                    (n, n.output) not in self.input_funcs):
                # note: we include n.output in the input_funcs hash to handle
                # the case where the node output is changed after the model
                # is constructed.  this isn't technically supported behaviour
                # in nengo, but the gui does it.

                if isinstance(n.output, Process):
                    self.input_funcs[(n, n.output)] = [
                        n.output.make_step(
                            (n.size_in,), (n.size_out,), self.dt,
                            n.output.get_rng(self.rng))
                        for _ in range(self.minibatch_size)]
                elif n.size_out > 0:
                    self.input_funcs[(n, n.output)] = [utils.align_func(
                        (n.size_out,), self.tensor_graph.dtype)(n.output)]
                else:
                    # a node with no inputs and no outputs, but it can still
                    # have side effects
                    self.input_funcs[(n, n.output)] = [n.output]

            if using_output:
                if n in input_feeds:
                    # move minibatch dimension to the end
                    feed_val = np.moveaxis(input_feeds[n], 0, -1)
                elif isinstance(n.output, np.ndarray):
                    feed_val = np.tile(n.output[None, :, None],
                                       (n_steps, 1, self.minibatch_size))
                else:
                    feed_val = np.zeros(
                        (n_steps, n.size_out, self.minibatch_size),
                        dtype=self.tensor_graph.dtype.as_numpy_dtype)

                    for i in range(n_steps):
                        # note: need to copy the output of func, as func
                        # may mutate its outputs in-place on subsequent calls
                        feed_val[i] = np.transpose([
                            func((i + self.n_steps + 1) * self.dt)
                            for func in self.input_funcs[(n, n.output)]])

                feed_vals[self.tensor_graph.input_ph[n]] = feed_val
            elif not isinstance(n.output, np.ndarray):
                # note: we still call the function even if the output
                # is not being used, because it may have side-effects
                for i in range(self.n_steps + 1, self.n_steps + n_steps + 1):
                    for func in self.input_funcs[(n, n.output)]:
                        func(i * self.dt)

        return feed_vals

    def _update_probe_data(self, probe_data, start, n_steps):
        """Updates the stored probe data (since the last reset) with the data
        from the latest run.

        Downsamples the probe data returned from TensorFlow (from every
        simulation timestep) according to probe `sample_every` and the number
        of steps run.

        Parameters
        ----------
        probe_data : list of `np.ndarray`
            Probe data from every timestep
        start : int
            The simulation timestep at which probe data starts
        n_steps : int
            The number of timesteps over which we want to collect data
        """

        # remove any extra timesteps (due to `unroll_simulation` mismatch)
        probe_data = [p[:, :n_steps] for p in probe_data]

        for i, p in enumerate(self.model.probes):
            if p.sample_every is not None:
                # downsample probe according to `sample_every`
                period = p.sample_every / self.dt
                steps = np.arange(start, start + n_steps)
                probe_data[i] = probe_data[i][:, (steps + 1) % period < 1]

            # update stored probe data
            self.model.params[p].append(probe_data[i])

    def _check_data(self, data, mode="input", n_batch=None, n_steps=None):
        """Performs error checking on simulation data.

        Parameters
        ----------
        data : dict of {:class:`~nengo:nengo.Node` or \
                            :class:`~nengo:nengo.Probe`: \
                        :class:`~numpy:numpy.ndarray`}
            Array of data associated with given objects in model (Nodes if
            mode=="input" or Probes if mode=="target")
        mode : "input" or "target", optional
            Whether this data corresponds to an input or target value
        n_batch : int, optional
            Number of elements in batch (if None, will just verify that all
            data items have same batch size)
        n_steps : int, optional
            Number of simulation steps (if None, will just verify that all
            data items have same number of steps)
        """

        for d, x in data.items():
            if x.ndim != 3:
                raise ValidationError(
                    "should have rank 3 (batch_size, n_steps, dimensions), "
                    "found rank %d" % x.ndim,
                    "%s data" % mode)

            if mode == "input":
                if d not in self.tensor_graph.invariant_inputs:
                    raise ValidationError(
                        "%s is not an input Node (a nengo.Node with "
                        "size_in==0), or is from a different network." % d,
                        "%s data" % mode)
            else:
                if d not in self.model.probes:
                    raise ValidationError(
                        "%s is not a Probe, or is from a different "
                        "network" % d, "%s data" % mode)

        args = [n_batch, n_steps]
        labels = ["batch size", "number of timesteps"]

        for i in range(2):
            if args[i] is None:
                val = next(iter(data.values())).shape[i]
                for n, x in data.items():
                    if x.shape[i] != val:
                        raise ValidationError(
                            "Elements have different %s: %s vs %s" %
                            (labels[i], val, x.shape[0]), "%s data" % mode)
            else:
                for n, x in data.items():
                    if x.shape[i] != args[i]:
                        raise ValidationError(
                            "Data for %s has %s=%s, which does not match "
                            "expected size (%s)" % (n, labels[i], x.shape[i],
                                                    args[i]),
                            "%s data" % mode)

        for n, x in data.items():
            if x.shape[0] < self.minibatch_size:
                raise ValidationError(
                    "Size of minibatch (%d) for %s data less than Simulation "
                    "`minibatch_size` (%d)" % (x.shape[0], n,
                                               self.minibatch_size),
                    "%s data" % mode)

            d = n.size_out if mode == "input" else n.size_in
            if x.shape[2] != d:
                raise ValidationError(
                    "Dimensionality of data (%s) does not match "
                    "dimensionality of %s (%s)" % (x.shape[2], n, d),
                    "%s data" % mode)

    @property
    def dt(self):
        """(float) The time step of the simulator."""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise ReadonlyError(attr='dt', obj=self)

    @property
    def training_step(self):
        return self.tensor_graph.training_step

    def __enter__(self):
        self._graph_context = self.tensor_graph.graph.as_default()
        self._device_context = self.tensor_graph.graph.device(
            self.tensor_graph.device)
        self._graph_context.__enter__()
        self._device_context.__enter__()
        return self

    def __exit__(self, *args):
        self._device_context.__exit__(*args)
        self._graph_context.__exit__(*args)
        self.close()

    def __del__(self):
        """Raise a RuntimeWarning if the Simulator is deallocated while open.
        """

        if self.closed is not None and not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. "
                "Simulators should be closed manually to ensure resources "
                "are properly freed." % self.model, RuntimeWarning)
            self.close()


class SimulationData(collections.Mapping):
    """Data structure used to access simulation data from the model.

    The main use case for this is to access Probe data; for example,
    ``probe_data = sim.data[my_probe]``.  However, it is also
    used to access the parameters of objects in the model; for example, after
    the model has been optimized via :meth:`.Simulator.train`, the updated
    encoder values for an ensemble can be accessed via
    ``trained_encoders = sim.data[my_ens].encoders``.

    Parameters
    ----------
    sim : :class:`.Simulator`
        The simulator from which data will be drawn
    minibatched : bool
        If False, discard the minibatch dimension on probe data

    Notes
    -----
    SimulationData shouldn't be created/accessed directly by the user, but
    rather via ``sim.data`` (which is an instance of SimulationData).
    """

    def __init__(self, sim, minibatched):
        self.sim = sim
        self.minibatched = minibatched

    def __getitem__(self, obj):
        """Return the data associated with ``obj``.

        Parameters
        ----------
        obj : :class:`~nengo:nengo.Probe` or :class:`~nengo:nengo.Ensemble` \
              or :class:`~nengo:nengo.Connection`
            Object whose simulation data is being accessed

        Returns
        -------
        :class:`~numpy:numpy.ndarray` or \
                :class:`~nengo:nengo.builder.ensemble.BuiltEnsemble` or \
                :class:`~nengo:nengo.builder.connection.BuiltConnection`
            Array containing probed data if ``obj`` is a
            :class:`~nengo:nengo.Probe`, otherwise the corresponding
            parameter object
        """

        if obj not in self.sim.model.params:
            raise ValidationError("Object is not in parameters of model %s" %
                                  self.sim.model, str(obj))

        data = self.sim.model.params[obj]

        if isinstance(obj, Probe):
            if len(data) == 0:
                return []
            data = np.concatenate(data, axis=1)
            if not self.minibatched:
                data = data[0]

            data.setflags(write=False)
        elif isinstance(obj, Ensemble):
            # get the live simulation values
            scaled_encoders = self.get_param(obj, "scaled_encoders")
            bias = self.get_param(obj, "bias")

            # infer the related values (rolled into scaled_encoders)
            gain = (obj.radius * np.linalg.norm(scaled_encoders, axis=-1) /
                    np.linalg.norm(data.encoders, axis=-1))
            encoders = obj.radius * scaled_encoders / gain[:, None]

            # figure out max_rates/intercepts from neuron model
            max_rates, intercepts = (
                obj.neuron_type.max_rates_intercepts(gain, bias))

            data = BuiltEnsemble(data.eval_points, encoders, intercepts,
                                 max_rates, scaled_encoders, gain, bias)
        elif isinstance(obj, Connection):
            # get the live simulation values
            weights = self.get_param(obj, "weights")

            # impossible to recover transform
            transform = None

            data = BuiltConnection(data.eval_points, data.solver_info, weights,
                                   transform)

        return data

    def get_param(self, obj, attr):
        """Returns the current parameter value for the given object.

        Parameters
        ----------
        obj : ``NengoObject``
            The nengo object for which we want to know the parameters
        attr : str
            The parameter of ``obj`` to be returned

        Returns
        -------
        :class:`~numpy:numpy.ndarray`
            Current value of the parameters associated with the given object

        Notes
        -----
        Parameter values should be accessed through ``sim.data``
        (which will call this function if necessary), rather than directly
        through this function.
        """

        if self.sim.closed:
            warnings.warn("Checking %s.%s after simulator is closed; cannot "
                          "fetch live value, so the initial value will be "
                          "returned." % (obj, attr))

            return getattr(self.sim.model.params[obj], attr)

        sig_obj, sig_attr = self._attr_map(obj, attr)

        try:
            sig = self.sim.model.sig[sig_obj][sig_attr]
        except KeyError:
            # sig_attr doesn't exist for this attribute
            return None

        if sig not in self.sim.tensor_graph.sig_map:
            # if sig isn't in sig_map then that means it isn't used anywhere
            # in the simulation (and therefore never changes), so we can
            # safely return the static build value
            param = getattr(self.sim.model.params[obj], attr)
        else:
            param = self.sim.sess.run(self.sim.tensor_graph.get_tensor(sig))

        if sig.minibatched:
            if not self.minibatched:
                param = param[..., 0]
            else:
                param = np.moveaxis(param, -1, 0)

        return param

    def _attr_map(self, obj, attr):
        """Maps from ``sim.data[obj].attr`` to the equivalent
        ``model.sig[obj][attr]``.

        Parameters
        ----------
        obj : ``NengoObject``
            The nengo object for which we want to know the parameters
        attr : str
            The parameter of ``obj`` to be returned

        Returns
        -------
        obj : ``NengoObject``
            The nengo object to key into ``model.sig``
        attr : str
            The name of the signal corresponding to input attr

        """

        if isinstance(obj, Ensemble) and attr == "bias":
            return obj.neurons, attr
        elif isinstance(obj, Ensemble) and attr == "scaled_encoders":
            return obj, "encoders"

        return obj, attr

    def __len__(self):
        return len(self.sim.model.params)

    def __iter__(self):
        return iter(self.sim.model.params)
