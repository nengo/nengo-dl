"""
The Simulator class is the access point for the main features of NengoDL,
including `running <.Simulator.run_steps>` and `training <.Simulator.train>`
a model.
"""

import collections
import copy
import logging
import warnings

from nengo import Ensemble, Connection, Probe, Network, Direct, Node
from nengo.builder.connection import BuiltConnection
from nengo.builder.ensemble import BuiltEnsemble
from nengo.ensemble import Neurons
from nengo.exceptions import (
    ReadonlyError,
    SimulatorClosed,
    NengoWarning,
    SimulationError,
    ValidationError,
)
from nengo.solvers import NoSolver
from nengo.utils.magic import decorator
from nengo.version import version_info as nengo_version
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client.timeline import Timeline
from tensorflow.python.ops import gradient_checker

from nengo_dl import utils, config, objectives
from nengo_dl.builder import NengoBuilder, NengoModel
from nengo_dl.compat import tf_compat, Convolution
from nengo_dl.tensor_graph import TensorGraph


logger = logging.getLogger(__name__)


@decorator
def with_self(wrapped, instance, args, kwargs):
    """A decorator that can be used to ensure that any TensorFlow operations happening
    within a method will use the settings associated with this Simulator."""

    keras_dtype = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx(instance.tensor_graph.dtype)
    with instance.graph.as_default(), instance.graph.device(
        instance.tensor_graph.device
    ):
        output = wrapped(*args, **kwargs)
    tf.keras.backend.set_floatx(keras_dtype)

    return output


class Simulator:  # pylint: disable=too-many-public-methods
    """
    Simulate network using the ``nengo_dl`` backend.

    Parameters
    ----------
    network : `~nengo.Network` or None
        A network object to be built and then simulated. If None,
        then a built model must be passed to ``model`` instead
    dt : float
        Length of a simulator timestep, in seconds
    seed : int
        Seed for all stochastic operators used in this simulator
    model : `~nengo.builder.Model`
        Pre-built model object
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        Device on which to execute computations (if None then uses the
        default device as determined by TensorFlow)
    unroll_simulation : int
        Unroll simulation loop by explicitly building the given number of
        iterations into the computation graph (improves simulation speed
        but increases build time)
    minibatch_size : int
        The number of simultaneous inputs that will be passed through the
        network
    tensorboard : str
        If not None, save network output in the TensorFlow summary format to
        the given directory, which can be loaded into TensorBoard
    progress_bar : bool
        If True (default), display progress information when building a model
    """

    def __init__(
        self,
        network,
        dt=0.001,
        seed=None,
        model=None,
        device=None,
        unroll_simulation=1,
        minibatch_size=None,
        tensorboard=None,
        progress_bar=True,
    ):
        self.closed = None
        self.unroll = unroll_simulation
        self.minibatch_size = 1 if minibatch_size is None else minibatch_size
        self.data = SimulationData(self, minibatch_size is not None)

        if seed is None:
            if network is not None and network.seed is not None:
                seed = network.seed + 1
            else:
                seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # TODO: multi-GPU support

        if device is None and not utils.tf_gpu_installed:
            warnings.warn(
                "No GPU support detected. It is recommended that you "
                "install tensorflow-gpu (`pip install tensorflow-gpu`)."
            )
            logger.info("Running on CPU")
        else:
            logger.info(
                "Running on %s",
                "CPU/GPU" if device is None else ("CPU" if "cpu" in device else "GPU"),
            )

        ProgressBar = utils.ProgressBar if progress_bar else utils.NullProgressBar

        # build model (uses default nengo builder)
        if model is None:
            self.model = NengoModel(
                dt=float(dt),
                label="%s, dt=%f" % (network, dt),
                builder=NengoBuilder(),
                fail_fast=False,
            )
        else:
            if dt != model.dt:
                warnings.warn(
                    "Model dt (%g) does not match Simulator "
                    "dt (%g)" % (model.dt, dt),
                    NengoWarning,
                )
            if nengo_version <= (2, 8, 0):
                # 2.8 has a bug where unpickling a model results in step having the
                # wrong dtype, which we fix here
                model.step._initial_value = model.step._initial_value.astype(np.int64)
            self.model = model

        if network is not None:
            p = ProgressBar("Building network", "Build")
            self.model.build(network, progress=p)

        # set up tensorflow graph plan
        with ProgressBar(
            "Optimizing graph", "Optimization", max_value=None
        ) as progress:
            self.tensor_graph = TensorGraph(
                self.model,
                self.dt,
                unroll_simulation,
                config.get_setting(self.model, "dtype", "float32"),
                self.minibatch_size,
                device,
                progress,
                self.rng,
            )

        # build keras models
        self.graph = tf.Graph()
        self._build_keras()

        self.reset()

        self.closed = False

    @with_self
    def _build_keras(self):
        tf.random.set_seed(self.seed)
        tf.config.set_soft_device_placement(False)

        # output simulation data for viewing via TensorBoard
        # if tensorboard is not None:
        #     if not os.path.exists(tensorboard):
        #         os.makedirs(tensorboard)
        #
        #     run_number = (
        #         max(
        #             [
        #                 int(x[4:])
        #                 for x in os.listdir(tensorboard)
        #                 if x.startswith("run")
        #             ]
        #             or [-1]
        #         )
        #         + 1
        #     )
        #     self.summary = tf_compat.summary.FileWriter(
        #         os.path.join(tensorboard, "run_%d" % run_number),
        #         graph=tf_compat.get_default_graph(),
        #     )
        # else:
        #     self.summary = None

        n_steps, self.node_inputs = self.tensor_graph.build_inputs()
        inputs = [n_steps] + list(self.node_inputs.values())
        outputs = self.tensor_graph(inputs)
        self.keras_model = keras.Model(inputs=inputs, outputs=outputs)
        self.keras_model_save = keras.Model(
            inputs=inputs, outputs=[self.tensor_graph.steps_run_and_save] + outputs[1:]
        )

    @with_self
    def reset(self, seed=None):
        """
        Resets the simulator to initial conditions.

        Parameters
        ----------
        seed : int
            If not None, overwrite the default simulator seed with this value
            (note: this becomes the new default simulator seed)

        Notes
        -----
        Changing the TensorFlow seed only affects ops created from then on; it has
        no impact on existing ops (either changing their seed or resetting their random
        state). So calling `.reset` will likely have no impact on any TensorFlow
        randomness (it will still affect numpy randomness, such as in
        `~nengo:nengo.Processes`, as normal).
        """

        if self.closed is not None and self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        # initialize variables and internal simulation state
        self.soft_reset(include_params=True, include_probes=True)

        # update rng
        if seed is not None:
            warnings.warn(
                "Changing the seed will not affect any TensorFlow operations "
                "created before the seed was updated"
            )
            self.seed = seed
        self.rng.seed(self.seed)
        tf.random.set_seed(self.seed)

        # execute post-build processes
        self.tensor_graph.build_post()

    @with_self
    def soft_reset(self, include_params=False, include_probes=False):
        """
        Resets the internal state of the simulation, but doesn't
        rebuild the graph.

        Parameters
        ----------
        include_params : bool
            If True, also reset any training that has been performed on
            network parameters (e.g., connection weights)
        include_probes : bool
            If True, also clear probe data
        """

        # reset saved state
        var_vals = []
        for key, var in self.tensor_graph.signals.saved_state.items():
            try:
                val = self.tensor_graph.base_arrays_init[False][key]
            except KeyError:
                # this is state created by `signals.make_internal`
                val = np.zeros(var.shape, dtype=var.dtype.as_numpy_dtype())
            var_vals.append((var, val))
        keras.backend.batch_set_value(var_vals)

        if include_params:
            # reset base params
            keras.backend.batch_set_value(
                list(
                    zip(
                        self.tensor_graph.signals.base_params.values(),
                        self.tensor_graph.base_arrays_init[True].values(),
                    )
                )
            )

        if include_probes:
            for p in self.model.probes:
                self.model.params[p] = []

        self._update_steps()

    def predict(self, inputs=None, n_steps=None, update_state=False, **kwargs):
        """
        Generate output predictions for the input samples.

        Computation is (optionally) done in batches.

        This function implements the `tf.keras.Model.predict
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_ API.

        Parameters
        ----------
        inputs
            Inputs can be specified as:
            - A dictionary of {`nengo:nengo.Node` or str: `~numpy:numpy.ndarray`}
              indicating the input values for the given nodes. Nodes can be referred
              to by the Node object itself or by a string name, which will be
              ``Node.label`` if one was specified, or ``"node_i"`` where ``i``
              indexes the nodes according to the order they were added to the model
              (this corresponds to the order found in `nengo:nengo.Network.all_nodes`).
            - A list of `~numpy:numpy.ndarray` indicating the input values for each
              input Node, ordered according to the order in which the Nodes were
              added to the model (this corresponds to the order found in
              `nengo:nengo.Network.all_nodes`).
            - An `~numpy:numpy.ndarray` indicating the input value for a single input
              Node.
            - A generator or ``tf.data.Dataset`` that produces one of the above. These
              input types must also explicitly pass a value for the ``n_steps`` input,
              which should have shape ``(batch_size, 1)``. This input should come first
              in the input list or use the string name "n_steps" (if passing a dict).

            All inputs should have shape ``(batch_size, n_steps, node.size_out)``.

            If an input value is not specified for one of the Nodes in the model then
            data will be filled in according to the Node definition (e.g., by calling
            the output function associated with that Node).  However, generator input
            types must explicitly specify values for all the input nodes.
        n_steps : int
            Number of simulation timesteps
        update_state : bool
            If True, update the saved simulation state at the end of the run, so that
            future operations will resume from the terminal state of this run.
        kwargs: dict
            Will be passed on to `tf.keras.Model.predict
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.

        Returns
        -------
        probe_values : dict of {`~nengo:nengo.Probe`: `~numpy:numpy.ndarray`}
            Output values from all the Probes in the network.
        """

        if "batch_size" in kwargs:
            # note: the keras "batch size" parameter refers to minibatch size
            # (i.e., the number of elements passed to the network in each iteration,
            # rather than the total number of elements in the data)
            warnings.warn(
                "Batch size is determined statically via Simulator.minibatch_size; "
                "ignoring value passed to `predict`"
            )
        kwargs["batch_size"] = None

        return self._predict(
            "predict",
            inputs=inputs,
            n_steps=n_steps,
            update_state=update_state,
            **kwargs,
        )

    def predict_on_batch(self, inputs=None, n_steps=None, update_state=False, **kwargs):
        """
        Generate output predictions for the input samples.

        Computation is done on a single minibatch of inputs (i.e., inputs must have
        shape ``(sim.minibatch_size, n_steps, node.size_in)``.

        This function implements the `tf.keras.Model.predict_on_batch
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_on_batch>`_
        API.

        Parameters
        ----------
        inputs
            Inputs can be specified as:
            - A dictionary of {`nengo:nengo.Node` or str: `~numpy:numpy.ndarray`}
              indicating the input values for the given nodes. Nodes can be referred
              to by the Node object itself or by a string name, which will be
              ``Node.label`` if one was specified, or ``"node_i"`` where ``i``
              indexes the nodes according to the order they were added to the model
              (this corresponds to the order found in `nengo:nengo.Network.all_nodes`).
            - A list of `~numpy:numpy.ndarray` indicating the input values for each
              input Node, ordered according to the order in which the Nodes were
              added to the model (this corresponds to the order found in
              `nengo:nengo.Network.all_nodes`).
            - An `~numpy:numpy.ndarray` indicating the input value for a single input
              Node.
            - A generator or ``tf.data.Dataset`` that produces one of the above. These
              input types must also explicitly pass a value for the ``n_steps`` input,
              which should have shape ``(batch_size, 1)``. This input should come first
              in the input list or use the string name "n_steps" (if passing a dict).

            All inputs should have shape
            ``(sim.minibatch_size, n_steps, node.size_out)``.

            If an input value is not specified for one of the Nodes in the model then
            data will be filled in according to the Node definition (e.g., by calling
            the output function associated with that Node).  However, generator input
            types must explicitly specify values for all the input nodes.
        n_steps : int
            Number of simulation timesteps
        update_state : bool
            If True, update the saved simulation state at the end of the run, so that
            future operations will resume from the terminal state of this run.
        kwargs: dict
            Will be passed on to `tf.keras.Model.predict_on_batch
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_on_batch>`_.

        Returns
        -------
        probe_values : dict of {`~nengo:nengo.Probe`: `~numpy:numpy.ndarray`}
            Output values from all the Probes in the network.
        """
        return self._predict(
            "predict_on_batch",
            inputs=inputs,
            n_steps=n_steps,
            update_state=update_state,
            **kwargs,
        )

    def predict_generator(self, inputs=None, update_state=False, **kwargs):
        """
        Generate output predictions for the input samples.

        Uses a generator or ``tf.data.Dataset`` as input. Generator must produce inputs
        as expected by `.predict_on_batch` (i.e., with shape
        ``(sim.minibatch_size, n_steps, node.size_in)``).

        This function implements the `tf.keras.Model.predict_generator
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_generator>`_
        API.

        Parameters
        ----------
        inputs
            A generator or ``tf.data.Dataset`` that produces inputs in one of the forms
            supported by `.predict_on_batch`.

            These generators must also explicitly pass a value for the ``n_steps``
            input, which should have shape ``(batch_size, 1)``. This input should come
            first in the input list or use the string name "n_steps"
            (if generating a dict).

            All generated inputs should have shape
            ``(sim.minibatch_size, n_steps, node.size_out)``.

            Inputs must be explicitly specified for all input Nodes in the network.
        update_state : bool
            If True, update the saved simulation state at the end of the run, so that
            future operations will resume from the terminal state of this run.
        kwargs: dict
            Will be passed on to `tf.keras.Model.predict_generator
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_generator>`_.

        Returns
        -------
        probe_values : dict of {`~nengo:nengo.Probe`: `~numpy:numpy.ndarray`}
            Output values from all the Probes in the network.
        """
        return self._predict(
            "predict_generator", inputs=inputs, update_state=update_state, **kwargs
        )

    @with_self
    def _predict(
        self, predict_type, inputs=None, n_steps=None, update_state=False, **kwargs
    ):
        """
        Internal base function for all the predict functions.

        Parameters
        ----------
        predict_type : "predict" or "predict_on_batch" or "predict_generator"
            The underlying function to call on the Keras model.
        inputs
            See description in documentation of ``<predict_type>`` function.
        n_steps
            See description in documentation of ``<predict_type>`` function.
        update_state
            See description in documentation of ``<predict_type>`` function.
        kwargs
            See description in documentation of ``<predict_type>`` function.

        Returns
        -------
            See description in documentation of ``<predict_type>`` function.
        """
        # TODO: double check this doesn't rebuild the graph each time it is called
        #  (e.g. with different values for n_steps)

        # batch size is determined from data in `predict`; others are single batch so
        # we know the size should be minibatch_size
        batch_size = None if predict_type == "predict" else self.minibatch_size
        inputs = self._generate_inputs(
            data=inputs, n_steps=n_steps, batch_size=batch_size
        )
        self._check_data(inputs, n_steps=n_steps, batch_size=batch_size)

        # call predict function
        model = self.keras_model_save if update_state else self.keras_model
        outputs = getattr(model, predict_type)(inputs, **kwargs)

        # reorganize results (will be flattened) back into dict
        if not isinstance(outputs, list):
            outputs = [outputs]
        output_dict = {p: outputs[1 + i] for i, p in enumerate(self.model.probes)}

        return output_dict

    def step(self, **kwargs):
        """
        Run the simulation for one time step.

        Parameters
        ----------
        kwargs : dict
            See `.run_steps`

        Notes
        -----
        Progress bar is disabled by default when running via this method.
        """

        kwargs.setdefault("progress_bar", False)

        self.run_steps(1, **kwargs)

    def run(self, time_in_seconds, **kwargs):
        """
        Simulate for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Run the simulator for the given number of simulated seconds
        kwargs : dict
            See `.run_steps`
        """

        if time_in_seconds < 0:
            raise ValidationError(
                "Must be positive (got %g)" % (time_in_seconds,), attr="time_in_seconds"
            )

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn(
                "%g results in running for 0 timesteps. Simulator "
                "still at time %g." % (time_in_seconds, self.time)
            )
        else:
            self.run_steps(steps, **kwargs)

    def run_steps(self, n_steps, data=None, progress_bar=True, extra_feeds=None):
        """
        Simulate for the given number of steps.

        Parameters
        ----------
        n_steps : int
            The number of simulation steps to be executed
        data : dict of {`~nengo.Node`: `~numpy.ndarray`}
            Override the values of input Nodes with the given data.  Arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        progress_bar : bool
            If True, print information about the simulation status to standard
            output.
        extra_feeds : dict of {``tf.Tensor``: `~numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)

        Notes
        -----
        If ``unroll_simulation=x`` is specified, and ``n_steps > x``, this will
        repeatedly execute ``x`` timesteps until the the number of steps
        executed is >= ``n_steps``.
        """

        actual_steps = self.unroll * int(np.ceil(n_steps / self.unroll))

        # error checking
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")
        if actual_steps != n_steps:
            warnings.warn(
                "Number of steps (%d) is not an even multiple of "
                "`unroll_simulation` (%d).  Simulation will run for %d steps, "
                "which may have unintended side effects."
                % (n_steps, self.unroll, actual_steps),
                RuntimeWarning,
            )
        if data is not None:
            # note: we only need to check the shape of the first item, because
            # check_data (inside predict) will ensure that all the items
            # have the same shape
            batch_size, input_steps = next(iter(data.values())).shape[:2]
            if batch_size != self.minibatch_size:
                raise ValidationError(
                    "Input data must have batch size == sim.minibatch_size "
                    "(%d != %d)" % (batch_size, self.minibatch_size),
                    "data",
                )
            if input_steps != actual_steps:
                raise ValidationError(
                    "Number of timesteps in input data (%d) does not "
                    "match requested number of steps (%d)"
                    % (input_steps, actual_steps),
                    "data",
                )

        progress = (
            utils.ProgressBar("Simulating", "Simulation", max_value=None)
            if progress_bar
            else utils.NullProgressBar()
        )

        with progress:
            # run the simulation
            try:
                output = self.predict_on_batch(
                    data, n_steps=actual_steps, update_state=True
                )
            except (tf.errors.InternalError, tf.errors.UnknownError) as e:
                if "nengo.exceptions.SimulationError" in e.message:
                    raise SimulationError(
                        "SimulationError detected; this most likely means that a "
                        "Python function (e.g. in a Node or Direct ensemble) caused "
                        "an error. See the full error log above."
                    )
                else:
                    raise e  # pragma: no cover (unknown errors)

        # update n_steps/time
        # note: we only need to do this here, because only run_steps updates the state
        self._update_steps()

        # update stored probe data
        for probe, val in output.items():
            if probe.sample_every is not None:
                # downsample probe according to `sample_every`
                period = probe.sample_every / self.dt
                steps = np.arange(
                    self.n_steps - actual_steps, self.n_steps
                )  # TODO: off by 1?
                val = val[:, (steps + 1) % period < 1]

            self.model.params[probe].append(val)

    def train(
        self,
        data,
        optimizer,
        n_epochs=1,
        objective=None,
        shuffle=True,
        truncation=None,
        summaries=None,
        profile=False,
        extra_feeds=None,
        progress_bar=True,
    ):
        """
        Optimize the trainable parameters of the network using the given
        optimization method, minimizing the objective value over the given
        inputs and targets.

        Parameters
        ----------
        data : dict of {`~nengo.Node` or `~nengo.Probe`: \
                        `~numpy.ndarray`} or int
            Input values for Nodes in the network or target values for Probes;
            arrays should have shape ``(batch_size, n_steps,
            node.size_out/probe.size_in)``.  If no input data is required,
            an integer can be given specifying the number of timesteps to
            run the simulation.
        optimizer : ``tf.train.Optimizer``
            TensorFlow optimizer, e.g.
            ``tf.train.GradientDescentOptimizer(learning_rate=0.1)``
        n_epochs : int
            Run training for the given number of epochs (complete passes
            through ``data``)
        objective : dict of {(tuple of) `~nengo.Probe`: callable or ``None``}
            The objective to be minimized. The default applies
            `.objectives.mse` to all probes in ``data``.  This can be
            overridden by passing a dictionary mapping Probes to functions
            ``f(output, target) -> loss`` that consume the actual output and
            target output for the given probe(s) and return a ``tf.Tensor``
            representing a scalar loss value.  The function may also accept a
            single argument ``f(output) -> loss`` if targets are not required.
            Some common objective functions can be found in
            `nengo_dl.objectives`.

            Passing ``None`` as the probe value (instead of a callable)
            indicates that the error is being computed outside the simulation,
            and the value passed for that probe in ``data`` directly specifies
            the output error gradient.

            If multiple probes are specified as the key, then the corresponding
            output/target values will be passed as a list to the objective
            function.

            The overall loss value being minimized will be the sum across all
            the objectives specified.
        shuffle : bool
            If True, randomize the data into different minibatches each epoch
        truncation: int
            If not None, use truncated backpropagation when training the
            network, with the given truncation length.
        summaries : list of `~nengo.Connection` or \
                            `~nengo.Ensemble` or \
                            `~nengo.ensemble.Neurons` or \
                            ``"loss"`` or \
                            ``tf.Tensor``
            If not None, collect data during the training process using
            TensorFlow's ``tf.summary`` format.  The summary objects can be a
            Connection (in which case data on the corresponding weights will be
            collected), Ensemble (encoders), Neurons (biases), or ``"loss"``
            (the loss value for ``objective``).  The user can also create their
            own summaries and pass in the Tensors representing the summary ops.
        profile : bool
            If True, collect TensorFlow profiling information while the
            simulation is running (this will slow down the simulation).
            Can also pass a string specifying a non-default filename for the
            saved profile data.
        extra_feeds : dict of {``tf.Tensor``: `~numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)
        progress_bar : bool
            If True, print information about the simulation status to standard
            output.

        Notes
        -----
        Most deep learning methods require the network to be differentiable,
        which means that trying to train a network with non-differentiable
        elements will result in an error.  Examples of common
        non-differentiable elements include `~nengo.LIF`,
        `~nengo.Direct`, or processes/neurons that don't have a
        custom TensorFlow implementation (see
        `.process_builders.SimProcessBuilder`/
        `.neuron_builders.SimNeuronsBuilder`)
        """

        if isinstance(data, int):
            batch_size = self.minibatch_size
            n_steps = data
        else:
            batch_size, n_steps = next(iter(data.values())).shape[:2]

        # error checking
        synapses = [
            x.synapse is not None
            for x in (
                self.model.toplevel.all_connections
                + (
                    list(p for p in data if isinstance(p, Probe))
                    if isinstance(data, dict)
                    else []
                )
            )
        ]
        if n_steps == 1 and self.model.toplevel is not None and any(synapses):
            warnings.warn(
                "Training for one timestep, but the network contains "
                "synaptic filters (which will introduce at least a "
                "one-timestep delay); did you mean to set synapse=None?"
            )
        if isinstance(optimizer, dict):
            raise ValidationError(
                "The second argument to `sim.train` should be a "
                "tf.train.Optimizer, not a dictionary; it is likely that this "
                "code was written for NengoDL 1.x and needs to be updated for "
                "NengoDL 2.x; see "
                "https://www.nengo.ai/nengo-dl/project.html#release-history",
                "optimizer",
            )

        # fill in default objective
        if objective is None:
            if isinstance(data, int):
                raise ValidationError(
                    "Must specify an explicit objective if no input data given",
                    "objective",
                )
            objective = {p: objectives.mse for p in data if isinstance(p, Probe)}

        if not isinstance(objective, dict):
            raise ValidationError(
                "Must be a dictionary mapping Probes to objective functions",
                "objective",
            )

        # build the objective
        loss, init_ops = self.tensor_graph.build_outputs(
            {k: v for k, v in objective.items() if v is not None}
        )
        if init_ops is not None:
            self.sess.run(init_ops)

        # create the optimizer function
        apply_optimizer = self.tensor_graph.build_optimizer_func(
            optimizer, loss, direct_grads=[p for p, g in objective.items() if g is None]
        )

        extra_fetches = dict()

        # add summaries
        if summaries is not None:
            if self.summary is None:
                warnings.warn(
                    "Simulator was created with tensorboard=False; "
                    "ignoring requested summaries"
                )
            else:
                for i, v in enumerate(summaries):
                    if isinstance(v, str) and v == "loss":
                        summaries[i] = objective
                summary_op, init = self.tensor_graph.build_summaries(summaries)
                if init is not None:
                    # initialize any variables created when building summaries
                    self.sess.run(init)
                extra_fetches["summaries"] = summary_op

        progress = (
            utils.ProgressBar(
                "Training",
                max_value=(
                    n_epochs
                    * (batch_size // self.minibatch_size)
                    * (1 if truncation is None else n_steps // truncation)
                ),
                vars=["loss"],
            )
            if progress_bar
            else utils.NullProgressBar()
        )

        objective_probes = tuple(objective.keys())

        def callback(out_vals, extra_vals):
            # update progress bar, with loss value
            loss = out_vals[objective_probes][1]
            # loss will be {} if only direct grads used when calculating
            # gradient
            kwargs = {} if loss == {} else dict(loss="%.4f" % loss)
            progress.step(**kwargs)

            # export summaries to tensorboard
            if "summaries" in extra_vals:
                # note: the first output value is the new value of the
                # global training_step
                self.summary.add_summary(
                    extra_vals["summaries"], out_vals[objective_probes][0]
                )

        # run training
        with progress:
            self.run_batch(
                data,
                {objective_probes: apply_optimizer},
                n_epochs=n_epochs,
                combine=lambda x: None,
                extra_feeds=extra_feeds,
                extra_fetches=extra_fetches,
                truncation=truncation,
                profile=profile,
                shuffle=shuffle,
                training=True,
                callback=callback,
            )

    def loss(
        self,
        data,
        objective=None,
        combine=np.mean,
        extra_feeds=None,
        progress_bar=True,
        training=False,
    ):
        """
        Compute the loss value for the given objective and inputs/targets.

        Parameters
        ----------
        data : dict of {`~nengo.Node` or `~nengo.Probe`: \
                        `~numpy.ndarray`} or int
            Input values for Nodes in the network or target values for Probes;
            arrays should have shape ``(batch_size, n_steps,
            node.size_out/probe.size_in)``.  If no input data is required,
            an integer can be given specifying the number of timesteps to
            run the simulation.
        objective : dict of {(tuple of) `~nengo.Probe`: callable}
            The objective to compute the loss. The default applies
            `.objectives.mse` to all probes in ``data``.  This can be
            overridden by passing a dictionary mapping Probes to functions
            ``f(output, target) -> loss`` that consume the actual output and
            target output for the given probe(s) and return a ``tf.Tensor``
            representing a scalar loss value.  The function may also accept a
            single argument ``f(output) -> loss`` if targets are not required.
            Some common objective functions can be found in
            `nengo_dl.objectives`.

            If multiple probes are specified as the key, then the corresponding
            output/target values will be passed as a list to the objective
            function.

            The overall value returned will be the sum across all
            the objectives specified.
        combine : callable
            Function used to combine objective values from each minibatch.
        extra_feeds : dict of {``tf.Tensor``: `~numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)
        progress_bar : bool
            If True, print information about the simulation status to standard
            output.
        training : bool
            If True, run the network in training mode (where, e.g., spiking
            neuron models are swapped for the equivalent differentiable
            approximation).

        Returns
        -------
        loss : float
            Sum of computed error values for each function in ``objective``.
        """

        batch_size = (
            self.minibatch_size
            if isinstance(data, int)
            else next(iter(data.values())).shape[0]
        )

        # fill in default objective
        if objective is None:
            if isinstance(data, int):
                raise ValidationError(
                    "Must specify an explicit objective if no input data given",
                    "objective",
                )
            objective = {p: objectives.mse for p in data if isinstance(p, Probe)}

        if not isinstance(objective, dict):
            raise ValidationError(
                "Must be a dictionary mapping Probes to objective functions",
                "objective",
            )

        progress = (
            utils.ProgressBar(
                "Calculating loss",
                "Calculation",
                max_value=batch_size // self.minibatch_size,
            )
            if progress_bar
            else utils.NullProgressBar()
        )
        with progress:
            loss = self.run_batch(
                data,
                objective,
                extra_feeds=extra_feeds,
                callback=lambda *_: progress.step(),
                combine=combine,
                training=training,
            )

        # sum across objectives
        loss = np.sum(list(loss.values()))

        return loss

    def run_batch(
        self,
        data,
        outputs,
        extra_feeds=None,
        extra_fetches=None,
        n_epochs=1,
        truncation=None,
        shuffle=False,
        profile=False,
        training=False,
        callback=None,
        combine=np.stack,
        isolate_state=True,
    ):
        """
        Run the simulation on a batch of input data, computing the given
        output functions.

        Parameters
        ----------
        data : dict of {`~nengo.Node` or `~nengo.Probe`: \
                        `~numpy.ndarray`} or int
            Input values for Nodes in the network or target values for Probes;
            arrays should have shape ``(batch_size, n_steps,
            node.size_out/probe.size_in)``.  If no input data is required,
            an integer can be given specifying the number of timesteps to
            run the simulation.
        outputs : dict of {(tuple of) `~nengo.Probe`: callable or None}
            Functions to apply to probe outputs.  Functions can accept one
            positional argument (the output from that probe on one minibatch)
            or two (also passed the corresponding target value from ``data``).
            If a tuple of Probes are given as the key then the first
            argument will be a list of probe outputs, and the second
            argument will be the corresponding list of target values.  The
            function can return a ``tf.Tensor``, or tuple of Tensors,
            which will be evaluated on each minibatch of data.  If ``None``
            is given then the return value will be the output value from that
            probe.
        extra_feeds : dict of {``tf.Tensor``: `~numpy.ndarray`}
            Can be used to feed a value for arbitrary Tensors in the simulation
            (will be passed directly to the TensorFlow session)
        extra_fetches : (list/tuple/dict of) ``tf.Tensor``
            Can be used to fetch arbitrary (structures of) Tensor values from
            the simulation (will be fetched directly from the TensorFlow
            session).
        n_epochs : int
            Repeat ``data`` for ``n_epochs`` iterations.
        truncation : int
            If not None, run the simulation ``truncation`` timesteps at a time.
            Outputs from each truncation block will be passed sequentially to
            ``combine``, in the same way as minibatch blocks.  Note
            that the simulation state is preserved between truncation blocks,
            so the sequence forms one continuous run within each minibatch.
        shuffle : bool
            If True, randomize the data into different minibatches each epoch.
        profile : bool
            If True, collect TensorFlow profiling information while the
            simulation is running (this will slow down the simulation).
            Can also pass a string specifying a non-default filename for the
            saved profile data.
        training : bool
            If True, run the network in training mode, otherwise run it in
            inference mode (this can affect things like the neuron model
            used).
        callback : callable
            A function that will be called after each minibatch is evaluated.
            The function is passed two arguments; the first is a dictionary
            corresponding to ``outputs`` with the output values from each
            function, and the second is the value of ``extra_feeds``.
        combine : callable
            The function that will be used to combine the outputs from each
            minibatch/truncation block.  The values from each output function
            on each minibatch will be formed into a list and passed to
            ``combine`` in order to compute the final return values from
            this function.  Note that if the output function returns multiple
            values, then ``combine`` will be applied separately to each of
            those outputs across the minibatches.
        isolate_state : bool
            If True (default), isolate the simulation state for this run
            from the rest of the simulation (so the execution of this run
            is not affected by previous runs and will not affect future runs).
            If False, then this run begins from the terminal state of the
            last run, each minibatch will continue in sequence from the state
            of the previous, and future runs will resume from the terminal
            state of the last minibatch of this run.

        Returns
        -------
        output_vals : dict of {(tuple of) `~nengo.Probe`: \
                               (tuple of) `~numpy.ndarray`}
            The result of computing ``outputs`` on simulation probe values,
            given ``data``.  This pseudocode may help to understand how the
            return values are constructed given the various parameters of this
            function:

            .. code-block:: python

                output_vals = {}
                for probe, func in outputs.items():
                    probe_vals = []
                    for i in range(n_epochs):
                        for minibatch in data:
                            network_output = run_network(minibatch)
                            probe_vals.append(func(network_output[probe]))
                    output_vals[probe] = combine(output_values)

            Note that this is not how the values are computed in practice,
            as it would be quite inefficient.  This pseudocode also omits
            some of the finer details (e.g. truncation and state isolation).

        Notes
        -----
        In general, users should call one of the wrappers for this function
        (e.g., `.run_steps`, `.train`, or `.loss`),
        according to their use case.  However, this function can be called
        directly to run the simulation in a customized way.
        """

        n_steps = data if isinstance(data, int) else next(iter(data.values())).shape[1]

        # error checking
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")
        if not isinstance(data, int):
            self._check_data(data)
        if n_steps % self.unroll != 0:
            raise ValidationError(
                "The number of timesteps in batch data must be evenly "
                "divisible by unroll_simulation",
                "data",
            )
        if truncation is not None and truncation % self.unroll != 0:
            raise ValidationError(
                "Truncation length must be evenly divisible by unroll_simulation",
                "truncation",
            )
        if training and self.tensor_graph.inference_only:
            raise ValidationError(
                "Network was created with inference_only=True, cannot "
                "be run in training mode",
                "inference_only",
            )

        if extra_fetches is None:
            extra_fetches = []

        # apply functions (if any) to output probes
        output_ops, init_ops = self.tensor_graph.build_outputs(outputs)

        # initialize any new variables
        if init_ops is not None:
            self.sess.run(init_ops)

        # save the internal state of the simulator
        if isolate_state:
            saved_state = self.sess.run(self.tensor_graph.signals.saved_state)

        # set up profiling
        if profile:
            run_options = tf_compat.RunOptions(
                trace_level=tf_compat.RunOptions.FULL_TRACE
            )
            run_metadata = tf_compat.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # compute outputs on batch
        output_vals = collections.defaultdict(list)
        for _ in range(n_epochs):
            for offset, mini_data in utils.minibatch_generator(
                data,
                self.minibatch_size,
                truncation=truncation,
                shuffle=shuffle,
                rng=self.rng,
            ):
                if offset == 0 and isolate_state:
                    self.soft_reset()

                # fill in feed_dict values
                if isinstance(mini_data, int):
                    steps = mini_data
                    mini_data = None
                else:
                    steps = next(iter(mini_data.values())).shape[1]
                feed = self._fill_feed(
                    steps,
                    data=mini_data,
                    training=training,
                    start=offset + (0 if isolate_state else self.n_steps),
                )
                if extra_feeds is not None:
                    feed.update(extra_feeds)

                # run the simulation
                try:
                    out_vals, state, extra_vals = self.sess.run(
                        (
                            output_ops,
                            self.tensor_graph.final_internal_state,
                            extra_fetches,
                        ),
                        feed_dict=feed,
                        options=run_options,
                        run_metadata=run_metadata,
                    )
                except (tf.errors.InternalError, tf.errors.UnknownError) as e:
                    if e.op is not None and e.op.type == "PyFunc":
                        raise SimulationError(
                            "Function '%s' caused an error (see error log "
                            "above)" % e.op.name
                        )
                    else:
                        raise e  # pragma: no cover (unknown errors)

                if callback is not None:
                    callback(out_vals, extra_vals)

                for k, v in out_vals.items():
                    output_vals[k].append(v)

                # update saved state
                self.sess.run(
                    [
                        var.assign(val)
                        for var, val in zip(self.signals.saved_state.values(), state)
                    ]
                )

        if isolate_state:
            # restore internal state of simulator
            self.sess.run(
                [
                    var.assign(val)
                    for var, val in zip(
                        self.tensor_graph.signals.saved_state.values(),
                        saved_state.values(),
                    )
                ]
            )

        # combine outputs from each minibatch
        for probe, vals in output_vals.items():
            # if the output function returns multiple items, keep those
            # arrays separate
            if isinstance(vals[0], (list, tuple)):
                output_vals[probe] = tuple(combine(v) for v in zip(*vals))
            else:
                output_vals[probe] = combine(vals)

        # convert back from defaultdict
        output_vals = dict(output_vals)

        # output profile data to file
        self._profile_output(profile, run_metadata)

        return output_vals

    def save_params(self, path, include_internal=False):
        """
        Save network parameters to the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter output file
        include_internal : bool
            If True (default False) also save information representing
            internal simulation state.

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        `.get_nengo_params`.
        """
        if self.closed:
            raise SimulatorClosed("Simulation has been closed, cannot save parameters")

        vars = self.tensor_graph.signals.all_variables

        if include_internal:
            vars.extend(self.tensor_graph.signals.saved_state.values())

        with tf.device("/cpu:0"):
            path = tf_compat.train.Saver(vars).save(self.sess, path)

        logger.info("Model parameters saved to %s", path)

    def load_params(self, path, include_internal=False):
        """
        Load network parameters from the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter input file
        include_internal : bool
            If True (default False) also load information representing
            internal simulation state.

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        `.get_nengo_params`.
        """
        if self.closed:
            raise SimulatorClosed("Simulation has been closed, cannot load parameters")

        vars = self.tensor_graph.signals.all_variables

        if include_internal:
            vars.extend(self.tensor_graph.signals.saved_state.values())

        with tf.device("/cpu:0"):
            tf_compat.train.Saver(vars).restore(self.sess, path)

        logger.info("Model parameters loaded from %s", path)

    def freeze_params(self, objs):
        """
        Stores the live parameter values from the simulation back into a
        Nengo object definition.

        This can be helpful for reusing a NengoDL model inside a different
        Simulator.  For example:

        .. code-block:: python

            with nengo.Network() as net:
                < build network >

            with nengo_dl.Simulator(net) as sim:
                < run some optimization >
                sim.freeze_params(net)

            with nengo.Simulator(net) as sim2:
                # run the network in the default Nengo simulator, with the
                # trained parameters
                sim2.run(1.0)

        Parameters
        ----------
        obj : (list of) ``NengoObject``
            The Nengo object(s) into which parameter values will be stored.
            Note that these objects must be members of the Network used to
            initialize the Simulator.

        Notes
        -----
        This modifies the source object in-place, and it may slightly modify
        the structure of that object.  The goal is to have the object produce
        the same output as it would if run in the NengoDL simulator.  It may
        not be possible to accurately freeze all possible object; if you run
        into errors in this process, try manually extracting the parameters you
        need in your model (from ``sim.data``).
        """

        if self.closed:
            raise SimulatorClosed(
                "Simulation has been closed, cannot freeze parameters"
            )

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        for obj in objs:
            if obj not in [self.model.toplevel] + self.model.toplevel.all_objects:
                raise ValueError(
                    "%s is not a member of the Network used to "
                    "initialize the Simulator"
                )

            if not isinstance(obj, (Network, Ensemble, Connection)):
                raise TypeError(
                    "Objects of type %s do not have parameters to store" % type(obj)
                )

            if isinstance(obj, Network):
                todo = obj.all_ensembles + obj.all_connections
            else:
                todo = [obj]

            for o, params in zip(todo, self.get_nengo_params(todo)):
                for k, v in params.items():
                    setattr(o, k, v)

    def get_nengo_params(self, nengo_objs, as_dict=False):
        """
        Extract model parameters in a form that can be used to initialize
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

                # now we want to insert two connected ensembles with
                # the same parameters as our previous network:
                d = nengo.Ensemble(10, 1, **params[0])
                e = nengo.Ensemble(10, 1, **params[1])
                f = nengo.Connection(d, e, **params[2])

        Parameters
        ----------
        nengo_objs : (list of) `~nengo.Ensemble` or `~nengo.Connection`
            A single object or list of objects for which we want to get the
            parameters.
        as_dict : bool
            If True, return the values as a dictionary keyed by object label,
            instead of a list (the default).  Note that in this case labels
            must be unique.

        Returns
        -------
        params : (list or dict) of dicts
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
        nengo_objs = [
            obj.ensemble if isinstance(obj, Neurons) else obj for obj in nengo_objs
        ]

        # find all the data we need to fetch
        fetches = []
        for obj in nengo_objs:
            if isinstance(obj, Connection):
                fetches.append((obj, "weights"))
            elif isinstance(obj, Ensemble):
                if isinstance(obj.neuron_type, Direct):
                    # we cannot transfer direct ensemble parameters, because
                    # the nengo builder ignores the encoders specified for
                    # a direct ensemble
                    raise ValueError(
                        "get_nengo_params will not work correctly for "
                        "Direct neuron ensembles. Try manually translating "
                        "your network using `sim.data` instead."
                    )

                fetches.extend([(obj, "scaled_encoders"), (obj, "bias")])
            else:
                raise ValueError(
                    "Can only get Nengo parameters for Ensembles or Connections"
                )

        # get parameter values from simulation
        data = self.data.get_params(*fetches)

        # store parameter values in a form that can be loaded in nengo
        params = []
        idx = 0
        for obj in nengo_objs:
            if isinstance(obj, Connection):
                weights = data[idx]
                idx += 1
                if isinstance(obj.pre_obj, Ensemble):
                    params.append(
                        {
                            "solver": NoSolver(weights.T, weights=False),
                            "function": lambda x, weights=weights: np.zeros(
                                weights.shape[0]
                            ),
                            "transform": 1,
                        }
                    )
                elif isinstance(obj.transform, Convolution):
                    transform = copy.copy(obj.transform)
                    # manually bypass the read-only check (we are sure that
                    # nothing else has a handle to the new transform at this
                    # point, so this won't cause any problems)
                    Convolution.init.data[transform] = weights
                    params.append({"transform": transform})
                else:
                    if all(x == 1 for x in weights.shape):
                        weights = np.squeeze(weights)
                    params.append({"transform": weights})
            else:
                # note: we don't want to change the original gain (even though
                # it is rolled into the encoder values), because connections
                # direct to `ens.neurons` will still use the gains (and those
                # gains are not updated during training, only the encoders)
                gain = self.model.params[obj].gain

                # the encoders we get from the simulation are the actual
                # weights we want in the simulation. but during the build
                # process, gains and radius will be applied to the encoders.
                # so we need to undo that scaling here, so that the build
                # process will result in the correct values.
                encoders = data[idx] * obj.radius / gain[:, None]

                params.append(
                    {
                        "encoders": encoders,
                        "normalize_encoders": False,
                        "gain": gain,
                        "bias": data[idx + 1],
                        "max_rates": Ensemble.max_rates.default,
                        "intercepts": Ensemble.intercepts.default,
                    }
                )
                idx += 2

        # return params in appropriate format
        if scalar:
            return params[0]

        if as_dict:
            param_dict = {}
            for obj, p in zip(nengo_objs, params):
                if obj.label in param_dict:
                    raise ValueError(
                        "Duplicate label ('%s') detected; cannot return "
                        "parameters with as_dict=True" % obj.label
                    )
                else:
                    param_dict[obj.label] = p
            params = param_dict

        return params

    def check_gradients(self, outputs=None, atol=1e-5, rtol=1e-3):
        """
        Perform gradient checks for the network (used to verify that the
        analytic gradients are correct).

        Raises a simulation error if the difference between analytic and
        numeric gradient is greater than ``atol + rtol * numeric_grad``
        (elementwise).

        Parameters
        ----------
        outputs : ``tf.Tensor`` or list of ``tf.Tensor`` or \
                  list of `~nengo.Probe`
            Compute gradients wrt this output (if None, computes wrt each
            output probe)
        atol : float
            Absolute error tolerance
        rtol : float
            Relative (to numeric grad) error tolerance

        Notes
        -----
        Calling this function will reset all values in the network, so it
        should not be intermixed with calls to `.Simulator.run`.
        """

        if self.tensor_graph.inference_only:
            raise ValidationError(
                "Network was created with inference_only=True, cannot "
                "compute gradients",
                "inference_only",
            )

        delta = 1e-3
        n_steps = self.unroll * 2

        data = {
            n: np.zeros((self.minibatch_size, n_steps, n.size_out))
            for n in self.tensor_graph.invariant_inputs
        }
        data.update(
            {
                p: np.zeros((self.minibatch_size, n_steps, p.size_in))
                for p in self.tensor_graph.target_phs
            }
        )
        feed = self._fill_feed(n_steps, data=data, training=True)

        if outputs is None:
            # note: the x + 0 is necessary because `gradient_checker`
            # doesn't work properly if the output variable is a tensorarray
            outputs = [x + 0 for x in self.tensor_graph.probe_arrays.values()]
        elif isinstance(outputs, tf.Tensor):
            outputs = [outputs]
        else:
            outputs = [self.tensor_graph.probe_arrays[p] + 0 for p in outputs]

        # check gradient wrt inp
        for node, inp in self.tensor_graph.input_phs.items():
            inp_shape = inp.get_shape().as_list()
            inp_shape = [n_steps if x is None else x for x in inp_shape]
            inp_tens = self.tensor_graph.input_phs[node]
            feed[inp_tens] = np.ascontiguousarray(feed[inp_tens])
            inp_val = np.ravel(feed[inp_tens])
            for out in outputs:
                out_shape = out.get_shape().as_list()
                out_shape = [n_steps if x is None else x for x in out_shape]

                # we need to compute the numeric jacobian manually, to
                # correctly handle variables (tensorflow doesn't expect
                # state ops in `compute_gradient`, because it doesn't define
                # gradients for them)
                numeric = np.zeros(
                    (
                        np.prod(inp_shape, dtype=np.int32),
                        np.prod(out_shape, dtype=np.int32),
                    )
                )

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

                dx, dy = gradient_checker._compute_dx_and_dy(inp, out, out_shape)

                with self.sess.as_default():
                    analytic = gradient_checker._compute_theoretical_jacobian(
                        inp,
                        inp_shape,
                        np.zeros(inp_shape),
                        dy,
                        out_shape,
                        dx,
                        extra_feed_dict=feed,
                    )

                if np.any(np.isnan(analytic)) or np.any(np.isnan(numeric)):
                    raise SimulationError("NaNs detected in gradient")
                fail = abs(analytic - numeric) >= atol + rtol * abs(numeric)
                if np.any(fail):
                    raise SimulationError(
                        "Gradient check failed for input %s and output %s\n"
                        "numeric values:\n%s\n"
                        "analytic values:\n%s\n"
                        % (node, out, numeric[fail], analytic[fail])
                    )

        self.soft_reset()

        logger.info("Gradient check passed")

    def trange(self, sample_every=None, dt=None):
        """
        Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        sample_every : float (Default: None)
            The sampling period of the probe to create a range for.
            If None, a time value for every ``dt`` will be produced.
        """

        if dt is not None:
            if sample_every is not None:
                raise ValidationError(
                    "Cannot specify both `dt` and `sample_every`. "
                    "Use `sample_every` only.",
                    attr="dt",
                    obj=self,
                )
            warnings.warn("`dt` is deprecated. Use `sample_every` instead.")
            sample_every = dt

        period = 1 if sample_every is None else sample_every / self.dt
        steps = np.arange(1, self.n_steps + 1)
        return self.dt * steps[steps % period < 1]

    def close(self):
        """
        Close the simulation, freeing resources.

        Notes
        -----
        The simulation cannot be restarted after it is closed.  This is not a
        technical limitation, just a design decision made for all Nengo
        simulators.
        """

        if not self.closed:
            if getattr(self, "summary", None) is not None:
                self.summary.close()

            # TODO: delete some data structures to free up memory?

            self.closed = True

    def _generate_inputs(self, data=None, n_steps=None, batch_size=None):
        """
        Generate inputs for the network (the output values of each Node with
        no incoming connections).

        Parameters
        ----------
        data : list or dict of {`~nengo.Node` or str: `~numpy.ndarray`}
            Override the values of input Nodes with the given data.  Arrays
            should have shape ``(sim.minibatch_size, n_steps, node.size_out)``.
        n_steps : int
            Number of simulation timesteps for which to generate input data

        Returns
        -------
        node_vals : dict of {str: `~numpy.ndarray}
            Simulation values for all the input Nodes in the network.
        """

        if data is None:
            data = {}

        if not isinstance(data, (list, tuple, dict, np.ndarray)):
            # data is some kind of generator, so we don't try to modify it (too many
            # different types of generators this could be)
            # TODO: basically what we'd like to do is map _generate_inputs to each
            #  item returned from the generator. is there a general way to do that?

            if n_steps is not None:
                raise SimulationError(
                    "Cannot automatically add n_steps to generator with type %s; "
                    "please specify n_steps manually as the first element in the "
                    "values yielded from generator, remembering that it needs to "
                    "be repeated to have shape (batch_size, 1)" % type(data)
                )

            return data

        if isinstance(data, np.ndarray):
            # convert unary inputs to length-1 lists
            data = [data]
        if isinstance(data, (list, tuple)):
            # convert list to named dict
            data = collections.OrderedDict(
                (input.op.name, val)
                for input, val in zip(self.keras_model.inputs[1:], data)
            )
        elif isinstance(data, dict):
            # convert nodes to string names
            new_data = collections.OrderedDict()
            for node, val in data.items():
                if isinstance(node, Node):
                    if node not in self.node_inputs:
                        raise ValidationError(
                            "%s is not an input Node (a nengo.Node with "
                            "size_in==0), or is from a different network." % node,
                            "data",
                        )

                    name = self.node_inputs[node].op.name
                else:
                    name = node
                new_data[name] = val
            data = new_data

        if len(data) == 0:
            data_batch = data_steps = None
        else:
            data_batch, data_steps = next(iter(data.values())).shape[:2]

        if batch_size is None:
            batch_size = self.minibatch_size if data_batch is None else data_batch
        if n_steps is None:
            if data_steps is None:
                raise ValidationError(
                    "Must specify either input data or n_steps", "data"
                )
            n_steps = data_steps

        input_vals = collections.OrderedDict()

        # fill in n_steps
        input_vals["n_steps"] = np.resize(n_steps, (batch_size, 1))

        # fill in data for input nodes
        for node, output in self.tensor_graph.input_funcs.items():
            name = self.node_inputs[node].op.name

            if name in data:
                node_val = data[name]
            elif isinstance(output, np.ndarray):
                # tile to n_steps/minibatch size
                node_val = np.tile(output[None, None, :], (batch_size, n_steps, 1))
            else:
                # call output function to determine value
                node_val = np.zeros(
                    (batch_size, n_steps, node.size_out),
                    dtype=np.dtype(self.tensor_graph.dtype),
                )

                for i in range(n_steps):
                    # note: need to copy the output of func, as func
                    # may mutate its outputs in-place on subsequent calls.
                    # this assignment will broadcast the output along the
                    # minibatch dimension if required.
                    # note: we still call the function even if the output
                    # is not being used in the graph, because it may have side-effects
                    node_val[:, i] = [
                        func((i + self.n_steps + 1) * self.dt) for func in output
                    ]

            input_vals[name] = node_val

        return input_vals

    def _check_data(self, data, batch_size=None, n_steps=None):
        """
        Performs error checking on simulation data.

        Parameters
        ----------
        data : dict of {str: `~numpy.ndarray` or ``tf.Tensor``}
            Array of data associated with given objects in model (Nodes or
            Probes)
        batch_size : int
            Number of elements in batch (if None, will just verify that all
            data items have same batch size)
        n_steps : int
            Number of simulation steps (if None, will just verify that all
            data items have same number of steps)
        """

        if not isinstance(data, dict):
            # data is a generator, so don't perform validation
            # TODO: could map this into the generator process as well?
            return

        if "n_steps" not in data:
            raise ValidationError("Must specify 'n_steps' in input data", "data")
        if (
            batch_size is None
            and (data["n_steps"].ndim != 2 or data["n_steps"].shape[1] != 1)
        ) or (batch_size is not None and data["n_steps"].shape != (batch_size, 1)):
            raise ValidationError(
                "'n_steps' has wrong shape; should be %s (note that this is just the "
                "integer n_steps value repeated)" % ((batch_size, 1),),
                "data",
            )
        if not np.all(data["n_steps"] == data["n_steps"][0, 0]):
            raise ValidationError("'n_steps' should all have the same value", "data")

        # exclude n_steps from further data checking
        data = {k: val for k, val in data.items() if k != "n_steps"}

        for name, x in data.items():
            if name not in [n.op.name for n in self.node_inputs.values()]:
                raise ValidationError(
                    "'%s' is not a valid input name; perhaps the name is wrong (it "
                    "should match the `label` on the Node), or this is not an input "
                    "Node (a Node with size_in==0) in this network." % name,
                    "data",
                )

            if len(x.shape) != 3:
                raise ValidationError(
                    "should have rank 3 (batch_size, n_steps, dimensions), "
                    "found rank %d" % len(x.shape),
                    "%s data" % name,
                )

            # elif isinstance(d, Probe):
            #     if d not in self.model.probes:
            #         raise ValidationError("%s is from a different network" % d,
            #                               "data")

            if x.shape[0] < self.minibatch_size:
                raise ValidationError(
                    "Size of minibatch (%d) less than Simulation `minibatch_size` (%d)"
                    % (x.shape[0], self.minibatch_size),
                    "%s data" % name,
                )

        args = [batch_size, n_steps]
        labels = ["batch size", "number of timesteps"]

        for i in range(2):
            if args[i] is None:
                val = next(iter(data.values())).shape[i]
                for n, x in data.items():
                    if x.shape[i] != val:
                        raise ValidationError(
                            "Elements have different %s: %s vs %s"
                            % (labels[i], val, x.shape[0]),
                            "data",
                        )
            else:
                for n, x in data.items():
                    if x.shape[i] != args[i]:
                        raise ValidationError(
                            "Data for %s has %s=%s, which does not match "
                            "expected size (%s)" % (n, labels[i], x.shape[i], args[i]),
                            "data",
                        )

        for node, input in self.node_inputs.items():
            if data[input.op.name].shape[2] != node.size_out:
                raise ValidationError(
                    "Dimensionality of data (%s) does not match "
                    "dimensionality of %s (%s)" % (x.shape[2], node, node.size_out),
                    "data",
                )

    def _profile_output(self, profile, run_metadata):
        """
        Outputs profile information to file.

        Parameters
        ----------
        profile : bool or str
            If True or a string (filename), output profile information to file
        run_metadata : ``tf.RunMetadata``
            TensorFlow RunMetadata proto populated with profiling data
        """

        if not profile:
            return

        trace = Timeline(step_stats=run_metadata.step_stats)
        if isinstance(profile, str):
            filename = profile
        else:
            filename = "nengo_dl_profile.json"
        with open(filename, "w") as f:
            f.write(trace.generate_chrome_trace_format())

    @with_self
    def _update_steps(self):
        if not hasattr(self, "_step_tensors"):
            # cache these so we aren't adding new ops every time we call this function
            self._step_tensors = [
                self.tensor_graph.get_tensor(self.model.step),
                self.tensor_graph.get_tensor(self.model.time),
            ]

        self._n_steps, self._time = [
            x.item() for x in keras.backend.batch_get_value(self._step_tensors)
        ]

    @property
    def dt(self):
        """The time (in seconds) represented by one simulation timestep."""
        return self.model.dt

    @dt.setter
    def dt(self, _):
        raise ReadonlyError(attr="dt", obj=self)

    @property
    def n_steps(self):
        """The current simulation timestep."""
        return self._n_steps

    @property
    def time(self):
        """The current simulation time."""
        return self._time

    def __enter__(self):
        self._graph_context = self.graph.as_default()
        self._device_context = self.graph.device(self.tensor_graph.device)

        self._graph_context.__enter__()
        self._device_context.__enter__()

        return self

    def __exit__(self, *args):
        self._device_context.__exit__(*args)
        self._graph_context.__exit__(*args)

        self.close()

    def __del__(self):
        """
        Raise a RuntimeWarning if the Simulator is deallocated while open.
        """

        if self.closed is not None and not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. "
                "Simulators should be closed manually to ensure resources "
                "are properly freed." % self.model,
                RuntimeWarning,
            )
            self.close()

    def __getstate__(self):
        raise NotImplementedError(
            "TensorFlow does not support pickling; see "
            "https://www.nengo.ai/nengo-dl/training.html"
            "#saving-and-loading-parameters "
            "for information on how to save/load a NengoDL model."
        )


class SimulationData(collections.Mapping):
    """
    Data structure used to access simulation data from the model.

    The main use case for this is to access Probe data; for example,
    ``probe_data = sim.data[my_probe]``.  However, it is also
    used to access the parameters of objects in the model; for example, after
    the model has been optimized via `.Simulator.train`, the updated
    encoder values for an ensemble can be accessed via
    ``trained_encoders = sim.data[my_ens].encoders``.

    Parameters
    ----------
    sim : `.Simulator`
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
        obj : `~nengo.Probe` or `~nengo.Ensemble` or `~nengo.Connection`
            Object whose simulation data is being accessed

        Returns
        -------
        data : `~numpy.ndarray` or \
               `~nengo.builder.ensemble.BuiltEnsemble` or \
               `~nengo.builder.connection.BuiltConnection`
            Array containing probed data if ``obj`` is a
            `~nengo.Probe`, otherwise the corresponding
            parameter object
        """

        if obj not in self.sim.model.params:
            raise ValidationError(
                "Object is not in parameters of model %s" % self.sim.model, str(obj)
            )

        data = self.sim.model.params[obj]

        if isinstance(obj, Probe):
            if len(data) == 0:
                return []
            data = np.concatenate(data, axis=1)
            if not self.minibatched:
                data = data[0]

            data.setflags(write=False)
        elif isinstance(obj, Ensemble):
            if isinstance(obj.neuron_type, Direct):
                # direct mode ensemble
                gain = bias = None
                scaled_encoders = encoders = self.get_params((obj, "scaled_encoders"))[
                    0
                ]
            else:
                # get the live simulation values
                scaled_encoders, bias = self.get_params(
                    (obj, "scaled_encoders"), (obj, "bias")
                )

                # infer the related values (rolled into scaled_encoders)
                gain = (
                    obj.radius
                    * np.linalg.norm(scaled_encoders, axis=-1)
                    / np.linalg.norm(data.encoders, axis=-1)
                )
                encoders = obj.radius * scaled_encoders / gain[:, None]

            # figure out max_rates/intercepts from neuron model
            max_rates, intercepts = obj.neuron_type.max_rates_intercepts(gain, bias)

            data = BuiltEnsemble(
                data.eval_points,
                encoders,
                intercepts,
                max_rates,
                scaled_encoders,
                gain,
                bias,
            )
        elif isinstance(obj, Connection):
            # get the live simulation values
            weights = self.get_params((obj, "weights"))[0]

            # impossible to recover transform
            transform = None

            data = BuiltConnection(
                data.eval_points, data.solver_info, weights, transform
            )

        return data

    def get_params(self, *obj_attrs):
        """
        Returns the current parameter values for the given objects.

        Parameters
        ----------
        obj_attrs : list of (``NengoObject``, str)
            The Nengo object and attribute of that object for which we want
            to know the parameter values (each object-attribute pair specified
            as a tuple argument to the function).

        Returns
        -------
        params : list of `~numpy.ndarray`
            Current values of the requested parameters

        Notes
        -----
        Parameter values should be accessed through ``sim.data``
        (which will call this function if necessary), rather than directly
        through this function.
        """

        if self.sim.closed:
            warnings.warn(
                "Checking parameters after simulator is closed; "
                "cannot fetch live values, so the initial values "
                "will be returned."
            )

            return [
                getattr(self.sim.model.params[obj], attr) for obj, attr in obj_attrs
            ]

        params = []
        sigs = []
        fetches = {}
        for obj, attr in obj_attrs:

            sig_obj, sig_attr = self._attr_map(obj, attr)
            sig = self.sim.model.sig[sig_obj][sig_attr]
            sigs.append(sig)

            if sig not in self.sim.tensor_graph.signals:
                # if sig isn't in sig_map then that means it isn't used
                # anywhere in the simulation (and therefore never changes), so
                # we can safely return the static build value
                params.append(getattr(self.sim.model.params[obj], attr))
            else:
                # this is a live parameter value we need to fetch from the
                # simulation. we queue them up and fetch them all at once to
                # be more efficient
                placeholder = object()
                fetches[placeholder] = self.sim.tensor_graph.get_tensor(sig)
                params.append(placeholder)

        # get the live parameter values
        fetched = dict(
            zip(fetches.keys(), keras.backend.batch_get_value(list(fetches.values())))
        )

        # final updating of parameters
        for i, sig in enumerate(sigs):
            # fill in placeholder values
            if type(params[i]) == object:
                params[i] = fetched[params[i]]

            if sig.minibatched and not self.minibatched:
                # drop minibatch dimension
                params[i] = params[i][0]

        return params

    def _attr_map(self, obj, attr):
        """
        Maps from ``sim.data[obj].attr`` to the equivalent
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

    # def __contains__(self, x):
    #     return any(type(x) == type(y) and x == y for y in self)
