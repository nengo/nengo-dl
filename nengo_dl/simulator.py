"""
The Simulator class is the access point for the main features of NengoDL,
including `running <.Simulator.run_steps>` and `training <.Simulator.fit>`
a model.
"""

import collections
import contextlib
import copy
import logging
import textwrap
import warnings
from functools import partial

import jinja2
import numpy as np
import tensorflow as tf
from nengo import Connection, Direct, Ensemble, Network, Node, Probe
from nengo import rc as nengo_rc
from nengo.builder.connection import BuiltConnection
from nengo.builder.ensemble import BuiltEnsemble
from nengo.ensemble import Neurons
from nengo.exceptions import (
    NengoWarning,
    ReadonlyError,
    SimulationError,
    SimulatorClosed,
    ValidationError,
)
from nengo.solvers import NoSolver
from nengo.transforms import Convolution, Dense, Sparse, SparseMatrix
from nengo.utils.magic import decorator

from nengo_dl import callbacks, compat, config, utils
from nengo_dl.builder import NengoBuilder, NengoModel
from nengo_dl.tensor_graph import TensorGraph

logger = logging.getLogger(__name__)


@decorator
def with_self(wrapped, instance, args, kwargs):
    """A decorator that can be used to ensure that any TensorFlow operations happening
    within a method will use the settings associated with this Simulator."""

    keras_dtype = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx(instance.tensor_graph.dtype)
    try:
        with tf.device(instance.tensor_graph.device):
            output = wrapped(*args, **kwargs)
    finally:
        tf.keras.backend.set_floatx(keras_dtype)

    return output


@decorator
def require_open(wrapped, instance, args, kwargs):
    """A decorator that can be used to mark methods that require the Simulator to
    be open."""

    if instance.closed:
        raise SimulatorClosed(
            f"Cannot call {wrapped.__name__} after simulator is closed"
        )

    return wrapped(*args, **kwargs)


def fill_docs(*args, **kwargs):
    """Stores documentation for common arguments in one place, to avoid duplication,
    and then fills them in automatically in the docstring."""

    docs = {
        "x": """
        {% set uses_y = func_name in ("fit", "evaluate") %}

        {% if func_name in ("predict_on_batch", "run_steps") %}
            {% set batch_size = 1 %}
        {% else %}
            {% set batch_size = 50 %}
        {% endif %}

        Data for input Nodes in the model. This argument is optional; if
        it is not specified, then data will automatically be generated
        according to the inputs specified in the Node definitions (e.g., by calling
        the output function associated with that Node).

        ``{{ param_name }}`` can be specified as:

        - A dictionary of {`nengo.Node` or str: `numpy.ndarray`}
          indicating the input values for the given nodes. Nodes can be referred
          to by the Node object itself or by a string name, which will be
          ``Node.label`` if one was specified or ``"node"``
          (duplicate names will have a number appended, corresponding to the order
          found in `nengo.Network.all_nodes`).
        - A list of `numpy.ndarray` indicating the input values for each
          input Node, ordered according to the order in which the Nodes were
          added to the model (this corresponds to the order found in
          `nengo.Network.all_nodes`).
        - A `numpy.ndarray` indicating the input value for a single input Node.
        {% if func_name not in ("predict_on_batch", "run_steps") %}
        - A generator or ``tf.data.Dataset`` that produces one of the above.
        {% endif %}

        All inputs should have shape ``(batch_size, n_steps, node.size_out)``.

        For example, if the model only has a single input Node, then
        ``{{ param_name }}`` can simply be an ndarray of data for that Node.

        .. testcode::

            with nengo.Network() as net:
                a = nengo.Node([0])
                p = nengo.Probe(a)

            with nengo_dl.Simulator(net) as sim:
                {% if uses_y %}
                sim.compile(loss="mse")
                sim.{{ func_name }}(
                    {{ param_name }}=np.ones((50, 10, 1)), y=np.ones((50, 10, 1)))
                {% elif func_name == "run_steps" %}
                sim.{{ func_name }}(
                    10, {{ param_name }}=np.ones(({{ batch_size }}, 10, 1)))
                {% else %}
                sim.{{ func_name }}(
                    {{ param_name }}=np.ones(({{ batch_size }}, 10, 1)))
                {% endif %}

        {% if uses_y %}
        .. testoutput::
            :hide:

            ...
        {% endif %}

        If the network has multiple inputs, then ``{{ param_name }}`` can be specified
        as a dictionary mapping `nengo.Node` objects to arrays, e.g.

        .. testcode::

            with nengo.Network() as net:
                a = nengo.Node([0])
                b = nengo.Node([0, 0])
                p = nengo.Probe(a)

            with nengo_dl.Simulator(net) as sim:
                {% if uses_y %}
                sim.compile(loss="mse")
                sim.{{ func_name }}(
                    {{ param_name }}={
                        a: np.ones((50, 10, 1)),
                        b: np.ones((50, 10, 2))
                    },
                    y=np.ones((50, 10, 1))
                )
                {% elif func_name == "run_steps" %}
                sim.{{ func_name }}(
                    10,
                    {{ param_name }}={
                        a: np.ones(({{ batch_size }}, 10, 1)),
                        b: np.ones(({{ batch_size }}, 10, 2))
                    }
                )
                {% else %}
                sim.{{ func_name }}(
                    {{ param_name }}={
                        a: np.ones(({{ batch_size }}, 10, 1)),
                        b: np.ones(({{ batch_size }}, 10, 2))
                    }
                )
                {% endif %}

        {% if uses_y %}
        .. testoutput::
            :hide:

            ...
        {% endif %}

        If an input value is not specified for one of the Nodes in the model then
        data will be filled in automatically according to the Node definition.

        {% if func_name not in ("predict_on_batch", "run_steps") %}
        For dynamic input types (e.g., ``tf.data`` pipelines or generators), NengoDL
        tries to avoid introspecting/altering the data before the
        simulation starts, as this may have unintended side-effects. So data must be
        specified via one of the standard Keras methods (arrays, list of arrays, or
        string name dictionary; using a dictionary of Node objects is not supported).
        In addition, data must be explicitly provided for all input nodes (it will not
        be automatically generated if data is not specified).

        In addition, when using dynamic inputs, data must be provided for the
        special ``"n_steps"`` input. This specifies the number of timesteps that the
        simulation will run for. Technically this is just a single scalar value
        (e.g., ``10``). But Keras requires that all input data be batched, so that
        input value needs to be duplicated into an array with size
        ``(batch_size, 1)`` (where all entries have the same value, e.g. ``10``).

        {% if uses_y %}
        Also keep in mind that when using a dynamic input for ``x`` the ``y`` parameter
        is unused, and instead the generator should return ``(x, y)`` pairs.
        {% endif %}

        .. testcode::

            with nengo.Network() as net:
                a = nengo.Node([0], label="a")
                p = nengo.Probe(a, label="p")

            with nengo_dl.Simulator(net) as sim:
                {% if uses_y %}
                dataset = tf.data.Dataset.from_tensor_slices(
                    ({"a": tf.ones((50, 10, 1)),
                      "n_steps": tf.ones((50, 1), dtype=tf.int32) * 10},
                     {"p": tf.ones((50, 10, 1))})
                ).batch(sim.minibatch_size)

                sim.compile(loss="mse")
                sim.{{ func_name }}({{ param_name }}=dataset)
                {% else %}
                dataset = tf.data.Dataset.from_tensor_slices(
                    {"a": tf.ones((50, 10, 1)),
                     "n_steps": tf.ones((50, 1), dtype=tf.int32) * 10}
                ).batch(sim.minibatch_size)

                sim.{{ func_name }}({{ param_name }}=dataset)
                {% endif %}

        {% if uses_y %}
        .. testoutput::
            :hide:

            ...
        {% endif %}
        {% endif %}
        """,
        "y": """
        Target values for Probes in the model. These can be specified in the same
        ways as the input values in ``x``, except using Probes instead of Nodes.
        All targets should have shape ``(batch_size, n_steps, probe.size_in)``.

        For example,

        .. testcode::

            with nengo.Network() as net:
                a = nengo.Node([0])
                p = nengo.Probe(a)

            with nengo_dl.Simulator(net) as sim:
                sim.compile(loss="mse")
                sim.{{ func_name }}(
                    x={a: np.zeros((50, 10, 1))}, y={p: np.zeros((50, 10, 1))})

        .. testoutput::
            :hide:

            ...

        Note that data is only specified for the probes used in the loss function
        (specified when calling `.Simulator.compile`).  For example, if we have two
        probes, but only one is used during training (the other is used for data
        collection during inference), we could set this up like:

        .. testcode::

            with nengo.Network() as net:
                a = nengo.Node([0])
                b = nengo.Node([0])
                p_a = nengo.Probe(a)
                p_b = nengo.Probe(b)

            with nengo_dl.Simulator(net) as sim:
                # compiled loss function only depends on p_a
                sim.compile(loss={p_a: "mse"})

                # only specify data for p_a
                sim.{{ func_name }}(
                    x={a: np.zeros((50, 10, 1))},  y={p_a: np.zeros((50, 10, 1))})

        .. testoutput::
            :hide:

            ...

        ``y`` is not used if ``x`` is a generator. Instead, the generator passed to
        ``x`` should yield ``(x, y)`` tuples, where ``y`` is in one of the formats
        described above.
        """,
        "n_steps": """
        The number of simulation steps to be executed.  This parameter is optional;
        if not specified, the number of simulation steps will be inferred from the
        input data. However, this parameter can be useful if you don't want to
        specify input data (you just want to use the inputs defined by the Nengo
        Nodes), or if your model does not have any input Nodes (so there is no data
        to be passed in).
        """,
        "stateful": """
        This parameter controls whether or not the saved internal stimulation state
        will be updated after a run completes. If ``stateful=False`` then the initial
        state of future runs will be unaffected by this run. With ``stateful=True``,
        future runs will begin from the terminal state of this run.

        For example,

        .. code-block:: python

            # begins in state0, terminates in state1
            sim.{{ func_name }}(..., stateful=False)
            # begins in state0, terminates in state2
            sim.{{ func_name }}(..., stateful=True)
            # begins in state2, terminates in state3
            sim.{{ func_name }}(..., stateful=False)
            # begins in state2, terminates in state4
            sim.{{ func_name }}(..., stateful=True)

        Note that `.Simulator.reset` can be used to reset the state to initial
        conditions at any point.
        """,
    }

    # use default name for args
    for arg in args:
        kwargs[arg] = arg

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def fill_documentation(func):
        rendered_docs = {}
        for name, template in kwargs.items():
            doc = docs[template]

            # fill in variables
            doc = env.from_string(doc).render(param_name=name, func_name=func.__name__)

            # correct indentation
            doc = textwrap.indent(doc, " " * 4)
            doc = doc.strip()

            rendered_docs[name] = doc

        # insert result into docstring
        func.__doc__ = env.from_string(func.__doc__).render(**rendered_docs)

        return func

    return fill_documentation


class Simulator:  # pylint: disable=too-many-public-methods
    """
    Simulate network using the ``nengo_dl`` backend.

    Parameters
    ----------
    network : `nengo.Network`
        A network object to be built and then simulated.
    dt : float
        Length of a simulator timestep, in seconds.
    seed : int
        Seed for all stochastic operators used in this simulator.
    model : `~nengo.builder.Model`
        Pre-built model object (mainly used for debugging).
    device : None or ``"/cpu:0"`` or ``"/gpu:[0-n]"``
        This specifies the computational device on which the simulation will
        run.  The default is ``None``, which means that operations will be assigned
        according to TensorFlow's internal logic (generally speaking, this means that
        things will be assigned to the GPU if GPU support is available,
        otherwise everything will be assigned to the CPU).  The device can be set
        manually by passing the `TensorFlow device specification
        <https://www.tensorflow.org/api_docs/python/tf/Graph#device>`_ to this
        parameter.  For example, setting ``device="/cpu:0"`` will force everything
        to run on the CPU.  This may be worthwhile for small models, where the extra
        overhead of communicating with the GPU outweighs the actual computations.  On
        systems with multiple GPUs, ``device="/gpu:0"``/``"/gpu:1"``/etc. will select
        which one to use.
    unroll_simulation : int
        This controls how many simulation iterations are executed each time through
        the outer simulation loop.  That is, we could run 20 timesteps as

        .. code-block:: python

            for i in range(20):
                <run 1 step>

        or

        .. code-block:: python

            for i in range(5):
                <run 1 step>
                <run 1 step>
                <run 1 step>
                <run 1 step>

        This is an optimization process known as "loop unrolling", and
        ``unroll_simulation`` controls how many simulation steps are unrolled.  The
        first example above would correspond to ``unroll_simulation=1``, and the
        second would be ``unroll_simulation=4``.

        Unrolling the simulation will result in faster simulation speed, but increased
        build time and memory usage.

        In general, unrolling the simulation will have no impact on the output of a
        simulation.  The only case in which unrolling may have an impact is if
        the number of simulation steps is not evenly divisible by
        ``unroll_simulation``.  In that case extra simulation steps will be executed,
        which could change the internal state of the simulation and
        will affect any subsequent calls to ``sim.run``.  So it is recommended that the
        number of steps always be evenly divisible by ``unroll_simulation``.
    minibatch_size : int
        The number of simultaneous inputs that will be passed through the
        network. For example, a single call to `.Simulator.run` will process
        ``minibatch_size`` input instances in parallel. Or when calling
        `.Simulator.predict`/`.Simulator.fit` with a batch of data, that data will be
        divided up into ``minibatch_size`` chunks.
    progress_bar : bool
        If True (default), display progress information when building a model. This will
        also be the default for the ``progress_bar`` argument within `.Simulator.run`
        and `.Simulator.run_steps`.

    Attributes
    ----------
    data : `.SimulationData`
        Stores simulation data and parameter values (in particular, the recorded output
        from probes after calling `.Simulator.run` can be accessed through
        ``sim.data[my_probe]``).
    model : `nengo.builder.Model`
        Built Nengo model, containing the data that defines the network to be simulated.
    keras_model : ``tf.keras.Model``
        Keras Model underlying the simulation (implements the inference/training loops).
    tensor_graph : `.tensor_graph.TensorGraph`
        Keras Layer implementing the Nengo simulation (built into ``keras_model``).
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

        if device is None and not utils.tf_gpu_installed:
            warnings.warn(
                "No GPU support detected. See "
                "https://www.nengo.ai/nengo-dl/installation.html#installing-tensorflow "
                "for instructions on setting up TensorFlow with GPU support."
            )
            logger.info("Running on CPU")
        else:
            logger.info(
                "Running on %s",
                "CPU/GPU"
                if device is None
                else ("CPU" if "cpu" in device.lower() else "GPU"),
            )

        self.progress_bar = progress_bar
        ProgressBar = utils.ProgressBar if progress_bar else utils.NullProgressBar

        # build model (uses default nengo builder)
        nengo_precision = nengo_rc.get("precision", "bits")
        nengo_rc.set(
            "precision",
            "bits",
            config.get_setting(model or network, "dtype", "float32")[-2:],
        )
        if model is None:
            self.model = NengoModel(
                dt=float(dt),
                label=f"{network}, dt={dt:f}",
                builder=NengoBuilder(),
                fail_fast=False,
            )
        else:
            if dt != model.dt:
                warnings.warn(
                    f"Model dt ({model.dt:g}) does not match Simulator dt ({dt:g})",
                    NengoWarning,
                )
            self.model = model

        if network is not None:
            p = ProgressBar("Building network", "Build")
            self.model.build(network, progress=p)

        nengo_rc.set("precision", "bits", nengo_precision)

        self.stateful = config.get_setting(self.model, "stateful", True)

        # set up tensorflow graph plan
        with ProgressBar(
            "Optimizing graph", "Optimization", max_value=None
        ) as progress:
            self.tensor_graph = TensorGraph(
                self.model,
                self.dt,
                unroll_simulation,
                self.minibatch_size,
                device,
                progress,
                seed,
            )

        # build keras models
        with ProgressBar(
            "Constructing graph", "Construction", max_value=None
        ) as progress:
            self._build_keras(progress)

        # initialize sim attributes
        self._n_steps = self._time = 0
        for p in self.model.probes:
            self.model.params[p] = []

        self.closed = False

    @with_self
    def _build_keras(self, progress=None):
        """
        Build the underlying Keras model that drives the simulation.

        Parameters
        ----------
        progress : `.utils.ProgressBar`
            Progress bar for construction stage.
        """

        self.node_inputs, n_steps = self.tensor_graph.build_inputs()
        inputs = list(self.node_inputs.values()) + [n_steps]

        outputs = self.tensor_graph(inputs, stateful=self.stateful, progress=progress)

        self.keras_model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="keras_model"
        )

        # set more informative output names
        # keras names them like LayerName_i, whereas we would like to have the names
        # associated with the probes
        self.keras_model.output_names = [
            self.tensor_graph.io_names[p] for p in self.model.probes
        ] + ["steps_run"]

        self.tensor_graph.build_post()

    @require_open
    @with_self
    def reset(
        self,
        seed=None,
        include_trainable=True,
        include_probes=True,
        include_processes=True,
    ):
        """
        Resets the simulator to initial conditions.

        Parameters
        ----------
        seed : int
            If not None, overwrite the default simulator seed with this value
            (note: this becomes the new default simulator seed).
        include_trainable : bool
            If True (default), also reset any online or offline training that has been
            performed on simulator parameters (e.g., connection weights).
        include_probes : bool
            If True (default), also clear probe data.
        include_processes: bool
            If True (default), also reset all `nengo.Process` objects in the model.

        Notes
        -----
        Changing the TensorFlow seed only affects ops created from then on; it has
        no impact on existing ops (either changing their seed or resetting their random
        state). So calling `.Simulator.reset` will likely have no impact on any
        TensorFlow randomness (it will still affect numpy randomness, such as in a
        `nengo.Process`, as normal).
        """

        reset_vars = (
            list(self.tensor_graph.saved_state.items()) if self.stateful else []
        )
        if include_trainable:
            reset_vars.extend(self.tensor_graph.base_params.items())

        if compat.eager_enabled():
            for key, var in reset_vars:
                var.assign(
                    # TODO: cache these instead of regenerating each time
                    self.tensor_graph.initial_values[key](var.shape, dtype=var.dtype)
                )
        else:
            tf.keras.backend.batch_get_value([var.initializer for _, var in reset_vars])

        if include_probes:
            for p in self.model.probes:
                self.model.params[p] = []

        self._update_steps()

        # update rng
        if seed is not None:
            warnings.warn(
                "Changing the seed will not affect any TensorFlow operations "
                "created before the seed was updated"
            )
            self.tensor_graph.seed = seed

        if include_processes:
            self.tensor_graph.build_post()

    @require_open
    @with_self
    def soft_reset(self, include_trainable=False, include_probes=False):
        """
        Deprecated, use `.Simulator.reset` instead.
        """

        warnings.warn(
            "Simulator.soft_reset is deprecated, use Simulator.reset("
            "include_trainable=False, include_probes=False, include_processes=False) "
            "instead",
            DeprecationWarning,
        )
        self.reset(
            seed=None,
            include_trainable=include_trainable,
            include_probes=include_probes,
            include_processes=False,
        )

    @require_open
    @fill_docs("x", "n_steps", "stateful")
    def predict(self, x=None, n_steps=None, stateful=False, **kwargs):
        """
        Generate output predictions for the input samples.

        Computation is (optionally) done in batches.

        This function implements the `tf.keras.Model.predict
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_ API.

        Parameters
        ----------
        x
            {{ x }}
        n_steps : int
            {{ n_steps }}
        stateful : bool
            {{ stateful}}
        kwargs: dict
            Will be passed on to `tf.keras.Model.predict
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.

        Returns
        -------
        probe_values : dict of {`nengo.Probe`: `numpy.ndarray`}
            Output values from all the Probes in the network.
        """

        return self._call_keras(
            "predict", x=x, n_steps=n_steps, stateful=stateful, **kwargs
        )

    @require_open
    @fill_docs("x", "n_steps", "stateful")
    def predict_on_batch(self, x=None, n_steps=None, stateful=False, **kwargs):
        """
        Generate output predictions for a single minibatch of input samples.

        Batch size is determined by ``sim.minibatch_size`` (i.e., inputs must have
        shape ``(sim.minibatch_size, n_steps, node.size_in)``.

        This function implements the `tf.keras.Model.predict_on_batch
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_on_batch>`_
        API.

        Parameters
        ----------
        x
            {{ x }}
        n_steps : int
            {{ n_steps }}
        stateful : bool
            {{ stateful }}
        kwargs: dict
            Will be passed on to `tf.keras.Model.predict_on_batch
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict_on_batch>`_.

        Returns
        -------
        probe_values : dict of {`nengo.Probe`: `numpy.ndarray`}
            Output values from all the Probes in the network.
        """

        # need to reset if simulator is stateful but this call is not stateful
        need_reset = not stateful and self.stateful

        # predict_on_batch doesn't support callbacks, so we do it manually
        if need_reset:
            cbk = callbacks.IsolateState(self)

        # note: setting stateful to self.stateful so that the inner _call_keras won't
        # try to do any resetting
        output = self._call_keras(
            "predict_on_batch", x=x, n_steps=n_steps, stateful=self.stateful, **kwargs
        )

        if need_reset:
            cbk.reset()
            self._update_steps()

        return output

    @require_open
    @with_self
    def compile(self, *args, loss=None, metrics=None, loss_weights=None, **kwargs):
        """
        Configure the model for training/evaluation.

        Parameters
        ----------
        args
            Will be passed on to `tf.keras.Model.compile
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
        loss
            Loss functions define the error that will be minimized during
            training.

            Losses can be specified as:

            - A `tf.losses.Loss
              <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_ instance.
            - A string matching the name of one of the loss functions above.
            - A function that accepts two arguments (``y_true, y_pred``) and returns
              a loss value (represented as a ``tf.Tensor``).
            - A list of some combination of the above, indicating different loss
              functions for each output Probe (ordered according to the order in
              which Probes were added to the model, which corresponds to the order
              found in ``Simulator.model.probes``).
            - A dictionary mapping Probe instances or names to loss functions.

            The total loss minimized during training will be the sum over the loss
            computed on each Probe (possibly weighted by ``loss_weights``).

            For example,

            .. testcode::

                with nengo.Network() as net:
                    node0 = nengo.Node([0])
                    node1 = nengo.Node([0])
                    probe0 = nengo.Probe(node0)
                    probe1 = nengo.Probe(node1)

                with nengo_dl.Simulator(net) as sim:
                    sim.compile(loss={probe0: "mse", probe1: tf.losses.mae})

            would compile ``probe0`` to use mean squared error and ``probe1`` to use
            mean absolute error.

        metrics
            Metrics are additional values (generally different kinds of losses) that
            will be computed during training for tracking purposes, but do not affect
            the result of the training.

            They can be specified in all the same ways as ``loss`` above.

            In addition, multiple metrics can be specified for each output Probe when
            using a list or dict, by providing multiple functions in a list (e.g.,
            ``metrics={my_probe: ["mae", "mse"]}``).
        loss_weights : list or dict
            Scalar weights that will be applied to the loss value computed for each
            output probe before summing them to compute the overall training loss. Can
            be a list (order corresponding to the order in ``loss``) or a dict mapping
            Probe instances/names to weights.
        kwargs
            Will be passed on to `tf.keras.Model.compile
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
        """

        # convert inputs to canonical name dict form
        loss = self._standardize_data(loss, self.model.probes, broadcast_unary=True)
        metrics = self._standardize_data(
            metrics, self.model.probes, broadcast_unary=True
        )
        loss_weights = self._standardize_data(loss_weights, self.model.probes)

        self.keras_model.compile(
            *args, loss=loss, metrics=metrics, loss_weights=loss_weights, **kwargs
        )

    @require_open
    @fill_docs("x", "y", "n_steps", "stateful")
    def fit(self, x=None, y=None, n_steps=None, stateful=False, **kwargs):
        """
        Trains the model on some dataset.

        Note that if the model contains spiking neurons, during the execution of this
        function those neurons will be swapped for the equivalent non-spiking
        implementation (as opposed to, e.g., `Simulator.evaluate`, which will
        use the spiking implementation).

        Optimizer and loss functions are defined separately in `.Simulator.compile`.

        This function implements the `tf.keras.Model.fit
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_ API.

        Parameters
        ----------
        x
            {{ x }}
        y
            {{ y }}
        n_steps : int
            {{ n_steps }}
        stateful : bool
            {{ stateful }}
        kwargs: dict
            Will be passed on to `tf.keras.Model.fit
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns
        -------
        history : ``tf.keras.callbacks.History``
            The history has two attributes: ``history.epoch`` is the list of epoch
            numbers, and ``history.history`` is a dictionary keyed by metric names
            (e.g., "loss") containing a list of values of those metrics from each epoch.
        """

        # if validation data is None or a dataset we don't do anything, but
        # otherwise we apply the same data augmentation/validation
        # as for x and y
        if isinstance(kwargs.get("validation_data", None), (list, tuple)):
            validation_data = kwargs["validation_data"]

            x_val = validation_data[0]
            x_val = self._generate_inputs(x_val, n_steps=n_steps)
            self._check_data(x_val, n_steps=n_steps)

            y_val = validation_data[1]
            y_val = self._standardize_data(y_val, self.model.probes)
            self._check_data(y_val, n_steps=None, nodes=False)

            if len(validation_data) == 2:
                kwargs["validation_data"] = (x_val, y_val)
            else:
                kwargs["validation_data"] = (x_val, y_val, validation_data[2])

        return self._call_keras(
            "fit", x=x, y=y, n_steps=n_steps, stateful=stateful, **kwargs
        )

    @require_open
    @fill_docs("x", "y", "n_steps", "stateful")
    def evaluate(self, x=None, y=None, n_steps=None, stateful=False, **kwargs):
        """
        Compute the loss and metric values for the network.

        Loss functions and other metrics are defined separately in `.Simulator.compile`.

        This function implements the `tf.keras.Model.evaluate
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate>`_ API.

        Parameters
        ----------
        x
            {{ x }}
        y
            {{ y }}
        n_steps : int
            {{ n_steps }}
        stateful : bool
            {{ stateful }}
        kwargs: dict
            Will be passed on to `tf.keras.Model.evaluate
            <https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate>`_.

        Returns
        -------
        outputs : dict of {str: `numpy.ndarray`}
            Computed loss/metric values. The overall loss will be in
            ``outputs["loss"]``, and values for each Probe will be in
            ``outputs["probe_name_loss"]`` or ``outputs["probe_name_metric_name"]``.
        """

        return self._call_keras(
            "evaluate", x=x, y=y, n_steps=n_steps, stateful=stateful, **kwargs
        )

    @with_self
    def _call_keras(
        self, func_type, x=None, y=None, n_steps=None, stateful=False, **kwargs
    ):
        """
        Internal base function for all the predict, fit, and evaluate functions.

        Parameters
        ----------
        func_type : "predict" or "predict_on_batch" or "fit" or "evaluate"
            The underlying function to call on the Keras model.
        x
            See description in documentation of ``<func_type>`` method.
        y
            See description in documentation of ``<func_type>`` method.
        n_steps : int
            See description in documentation of ``<func_type>`` method.
        stateful : bool
            See description in documentation of ``<func_type>`` method.
        kwargs : dict
            Will be passed to the underlying Keras function.

        Returns
        -------
            See description in documentation of ``<func_type>`` method.
        """

        if func_type.startswith("fit") and self.tensor_graph.inference_only:
            raise SimulationError(
                "Network was created with inference_only=True, cannot "
                "be run in training mode"
            )

        if stateful and not self.stateful:
            warnings.warn(
                "Ignoring stateful=True, since the model was created with the "
                "stateful=False config setting."
            )

        if "batch_size" in kwargs:
            # note: the keras "batch size" parameter refers to minibatch size
            # (i.e., the number of elements passed to the network in each iteration,
            # rather than the total number of elements in the data)
            warnings.warn(
                "Batch size is determined statically via Simulator.minibatch_size; "
                f"ignoring value passed to `{func_type}`"
            )
        if "on_batch" not in func_type:
            kwargs["batch_size"] = (
                self.minibatch_size if compat.eager_enabled() else None
            )

        # TODO: apply standardize/generate/check data to generator somehow
        # maybe move it into a callback where the generated data is available?

        x = self._generate_inputs(x, n_steps=n_steps)
        self._check_data(
            x,
            n_steps=n_steps,
            batch_size=self.minibatch_size if "on_batch" in func_type else None,
        )

        if isinstance(x, dict):
            input_steps = x["n_steps"][0, 0]
            input_batch = x["n_steps"].shape[0]
        else:
            input_steps = None
            input_batch = self.minibatch_size if "on_batch" in func_type else None

        if y is not None:
            y = self._standardize_data(y, self.model.probes)
            # we set n_steps=None because targets do not necessarily need to have
            # the same number of timesteps as input (depending on the loss function)
            self._check_data(y, n_steps=None, batch_size=input_batch, nodes=False)

        if kwargs.get("validation_split", 0) != 0 and input_batch is not None:
            # validation_split is only a kwarg in `fit`, but we do it here because
            # we need to know `input_batch`.
            # split math set up to match
            # `keras.engine.training_utils.split_training_and_validation_data`.
            split = int(input_batch * (1 - kwargs["validation_split"]))
            if (
                split % self.minibatch_size != 0
                or (input_batch - split) % self.minibatch_size != 0
            ):
                raise ValidationError(
                    "Split data is not evenly divisible by minibatch size",
                    "validation_split",
                )

        # warn for synapses with n_steps=1
        # note: we don't warn if stateful, since there could be effects across runs
        if not stateful:
            if compat.eager_enabled():
                target_probes = [
                    p
                    for p, e in zip(self.model.probes, self.keras_model.output_names)
                    if self.keras_model.compiled_loss is None
                    or self.keras_model.compiled_loss._losses is None
                    or e in self.keras_model.compiled_loss._losses
                ]
            else:
                target_probes = [
                    p
                    for p, e in zip(
                        self.model.probes,
                        getattr(self.keras_model, "_training_endpoints", []),
                    )
                    if not e.should_skip_target()
                ]

            synapses = [
                x.synapse is not None
                for x in (self.model.toplevel.all_connections + target_probes)
            ]
            if input_steps == 1 and self.model.toplevel is not None and any(synapses):
                warnings.warn(
                    "Running for one timestep, but the network contains "
                    "synaptic filters (which will introduce at least a "
                    "one-timestep delay); did you mean to set synapse=None?"
                )

        # set up callback to reset state after execution.
        # only necessary if simulator is stateful but this call is not stateful
        if not stateful and self.stateful:
            kwargs["callbacks"] = (kwargs.get("callbacks", None) or []) + [
                callbacks.IsolateState(self)
            ]

        # call underlying keras function
        if "predict" in func_type:
            func_args = dict(x=x, **kwargs)
        else:
            func_args = dict(x=x, y=y, **kwargs)

        outputs = getattr(self.keras_model, func_type)(**func_args)

        # update n_steps/time
        if stateful:
            self._update_steps()

        # process keras outputs
        if func_type.startswith("predict"):
            # reorganize results (will be flattened) back into dict
            if not isinstance(outputs, list):
                outputs = [outputs]
            return dict(zip(self.model.probes, outputs))
        elif func_type.startswith("evaluate"):
            # return outputs as named dict
            return dict(zip(self.keras_model.metrics_names, outputs))
        else:
            # return training history
            return outputs

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
        Run the simulation for the given length of time.

        Parameters
        ----------
        time_in_seconds : float
            Run the simulator for the given number of simulated seconds.
        kwargs : dict
            See `.run_steps`
        """

        if time_in_seconds < 0:
            raise ValidationError(
                f"Must be positive (got {time_in_seconds:g})", attr="time_in_seconds"
            )

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn(
                f"{time_in_seconds:g} results in running for 0 timesteps. Simulator "
                f"still at time {self.time:g}."
            )
        else:
            self.run_steps(steps, **kwargs)

    @require_open
    @fill_docs("stateful", data="x")
    def run_steps(self, n_steps, data=None, progress_bar=None, stateful=True):
        """
        Run the simulation for the given number of steps.

        Parameters
        ----------
        n_steps : int
            The number of simulation steps to be executed.
        data :
            {{ data }}
        progress_bar : bool
            If True, print information about the simulation status to standard
            output.
        stateful : bool
            {{ stateful }}

        Notes
        -----
        If ``unroll_simulation=x`` is specified, and ``n_steps > x``, this will
        repeatedly execute ``x`` timesteps until the the number of steps
        executed is >= ``n_steps``.
        """

        actual_steps = self.unroll * int(np.ceil(n_steps / self.unroll))

        # error checking
        if actual_steps != n_steps:
            warnings.warn(
                f"Number of steps ({n_steps}) is not an even multiple of "
                f"`unroll_simulation` ({self.unroll}).  Simulation will run for "
                f"{actual_steps} steps, which may have unintended side effects.",
                RuntimeWarning,
            )

        if progress_bar is None:
            progress_bar = self.progress_bar
        progress = (
            utils.ProgressBar("Simulating", "Simulation", max_value=None)
            if progress_bar
            else utils.NullProgressBar()
        )

        with progress:
            # run the simulation
            try:
                output = self.predict_on_batch(
                    data, n_steps=actual_steps, stateful=stateful
                )
            except (tf.errors.InternalError, tf.errors.UnknownError) as e:
                if "nengo.exceptions.SimulationError" in e.message:
                    raise SimulationError(
                        "SimulationError detected; this most likely means that a "
                        "Python function (e.g. in a Node or Direct ensemble) caused "
                        "an error. See the full error log above."
                    ) from e
                else:
                    raise e  # pragma: no cover (unknown errors)

        # update stored probe data
        for probe, val in output.items():
            if probe.sample_every is not None:
                # downsample probe according to `sample_every`
                period = probe.sample_every / self.dt
                steps = np.arange(self.n_steps - actual_steps, self.n_steps)
                val = val[:, (steps + 1) % period < 1]

            self.model.params[probe].append(val)

    def train(self, *args, **kwargs):
        """Deprecated, use `.Simulator.compile` and `.Simulator.fit` instead."""

        raise SimulationError(
            "Simulator.train has been deprecated, use Simulator.compile/fit instead"
        )

    def loss(self, *args, **kwargs):
        """Deprecated, use `.Simulator.compile` and `.Simulator.evaluate` instead."""

        raise SimulationError(
            "Simulator.loss has been deprecated, use Simulator.compile/evaluate instead"
        )

    @require_open
    @with_self
    def save_params(self, path, include_state=False, include_non_trainable=None):
        """
        Save network parameters to the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter output file.
        include_state : bool
            If True (default False) also save the internal simulation state.

            .. versionchanged:: 3.2.0
               Renamed from ``include_non_trainable`` to ``include_state``.

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        `.get_nengo_params`.
        """

        if include_non_trainable is not None:
            warnings.warn(
                "include_non_trainable is deprecated, use include_state instead",
                DeprecationWarning,
            )
            include_state = include_non_trainable

        params = list(self.keras_model.weights)
        if include_state:
            params.extend(self.tensor_graph.saved_state.values())

        np.savez_compressed(
            str(path) + ".npz", *tf.keras.backend.batch_get_value(params)
        )

        logger.info("Model parameters saved to %s.npz", path)

    @require_open
    @with_self
    def load_params(self, path, include_state=False, include_non_trainable=None):
        """
        Load network parameters from the given ``path``.

        Parameters
        ----------
        path : str
            Filepath of parameter input file.
        include_state : bool
            If True (default False) also save the internal simulation state.

            .. versionchanged:: 3.2.0
               Renamed from ``include_non_trainable`` to ``include_state``.

        Notes
        -----
        This function is useful for saving/loading entire models; for
        saving/loading individual objects within a model, see
        `.get_nengo_params`.
        """

        if include_non_trainable is not None:
            warnings.warn(
                "include_non_trainable is deprecated, use include_state instead",
                DeprecationWarning,
            )
            include_state = include_non_trainable

        params = list(self.keras_model.weights)
        if include_state:
            params.extend(self.tensor_graph.saved_state.values())

        with np.load(str(path) + ".npz") as vals:
            if len(params) != len(vals.files):
                raise SimulationError(
                    f"Number of saved parameters in {path} ({len(vals.files)}) != "
                    f"number of variables in the model ({len(params)})"
                )
            tf.keras.backend.batch_set_value(
                zip(params, (vals[f"arr_{i}"] for i in range(len(vals.files))))
            )

        logger.info("Model parameters loaded from %s.npz", path)

    @require_open
    def freeze_params(self, objs):
        """
        Stores the live parameter values from the simulation back into a
        Nengo object definition.

        This can be helpful for reusing a NengoDL model inside a different
        Simulator.  For example:

        .. testcode::

            with nengo.Network() as net:
                ens = nengo.Ensemble(10, 1)

            with nengo_dl.Simulator(net) as sim:
                # < run some optimization >
                sim.freeze_params(net)

            with nengo.Simulator(net) as sim2:
                # run the network in the default Nengo simulator, with the
                # trained parameters
                sim2.run(1.0)

        .. testoutput::
            :hide:

            ...

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
                    f"Objects of type {type(obj)} do not have parameters to store"
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

        .. testcode::

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

        Note that this function only returns trainable parameters (e.g. connection
        weights, biases, or encoders), or parameters that directly interact with
        those parameters (e.g. gains). Other arguments that are independent of the
        trainable parameters (e.g. ``Ensemble.neuron_type`` or ``Connection.synapse``)
        should be specified manually (since they may change between models).

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
                if compat.conn_has_weights(obj):
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
                if not compat.conn_has_weights(obj):
                    params.append({"transform": None})
                    continue

                weights = data[idx]
                idx += 1
                if isinstance(obj.transform, Convolution):
                    transform = copy.copy(obj.transform)
                    # manually bypass the read-only check (we are sure that
                    # nothing else has a handle to the new transform at this
                    # point, so this won't cause any problems)
                    Convolution.init.data[transform] = weights
                    params.append({"transform": transform})
                elif isinstance(obj.transform, Sparse):
                    transform = copy.copy(obj.transform)
                    if isinstance(transform.init, SparseMatrix):
                        init = SparseMatrix(
                            transform.init.indices, weights, transform.init.shape
                        )
                    else:
                        init = transform.init.tocoo()
                        init = SparseMatrix(
                            np.stack((init.row, init.col), axis=-1), weights, init.shape
                        )
                    Sparse.init.data[transform] = init
                    params.append({"transform": transform})
                elif isinstance(obj.transform, (Dense, compat.NoTransform)):
                    if isinstance(obj.pre_obj, Ensemble):
                        # decoded connection
                        params.append(
                            {
                                "solver": NoSolver(weights.T, weights=False),
                                "function": lambda x, weights=weights: np.zeros(
                                    weights.shape[0]
                                ),
                                "transform": compat.default_transform,
                            }
                        )
                    else:
                        if all(x == 1 for x in weights.shape):
                            weights = np.squeeze(weights)
                        params.append({"transform": weights})
                else:
                    raise NotImplementedError(
                        f"Cannot get parameters of Connections with transform type "
                        f"'{type(obj.transform).__name__}'"
                    )
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
                        f"Duplicate label ('{obj.label}') detected; cannot return "
                        "parameters with as_dict=True"
                    )
                else:
                    param_dict[obj.label] = p
            params = param_dict

        return params

    @require_open
    @with_self
    def check_gradients(self, inputs=None, outputs=None, atol=1e-5, rtol=1e-3):
        """
        Perform gradient checks for the network (used to verify that the
        analytic gradients are correct).

        Raises a simulation error if the difference between analytic and
        numeric gradient is greater than ``atol + rtol * numeric_grad``
        (elementwise).

        Parameters
        ----------
        inputs : list of `numpy.ndarray`
            Input values for all the input Nodes in the model (ordered according to
            the order in which Nodes were added to the model). If None, will use all
            zeros.
        outputs : list of `~nengo.Probe`
            Compute gradients wrt this output (if None, computes wrt each
            output probe).
        atol : float
            Absolute error tolerance.
        rtol : float
            Relative (to numeric grad) error tolerance.

        Notes
        -----
        Calling this method will reset the internal simulation state.
        """

        if self.tensor_graph.inference_only:
            raise SimulationError(
                "Network was created with inference_only=True, cannot "
                "compute gradients"
            )

        if inputs is None:
            n_steps = self.unroll * 2
            inputs = [
                np.zeros(
                    tuple(n_steps if s is None else s for s in x.shape),
                    x.dtype.as_numpy_dtype(),
                )
                for x in self.keras_model.inputs[:-1]
            ]
        else:
            n_steps = inputs[0].shape[1]

        if outputs is None:
            outputs = self.model.probes

        # compute_gradients expects to be called with a function that works in
        # specific ways, so we wrap the model to work in the way it expects
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def arg_func(*args, output=None):
            for i, x in enumerate(args):
                x.set_shape(inputs[i].shape)

            args += (tf.ones((self.minibatch_size, 1), dtype=np.int32) * n_steps,)

            out = self.tensor_graph(args, training=True)
            self.tensor_graph.build_post()

            if self.stateful:
                # reset state
                for key, var in self.tensor_graph.saved_state.items():
                    var.assign(
                        self.tensor_graph.initial_values[key](
                            var.shape, dtype=var.dtype
                        )
                    )

            # drop steps_run
            out = out[:-1]

            # get selected output
            out = out[self.model.probes.index(output)]

            return out

        self.reset(
            include_probes=False, include_trainable=False, include_processes=False
        )

        ctx = (
            # noop
            contextlib.suppress
            if compat.eager_enabled()
            else tf.compat.v1.keras.backend.get_session().as_default
        )

        grads = dict()
        for output in outputs:
            with ctx():
                analytic, numeric = tf.test.compute_gradient(
                    partial(arg_func, output=output), inputs
                )
            grads[output] = dict()
            grads[output]["analytic"] = analytic
            grads[output]["numeric"] = numeric

            for a, n in zip(analytic, numeric):
                if np.any(np.isnan(a)) or np.any(np.isnan(n)):
                    raise SimulationError("NaNs detected in gradient")
                fail = abs(a - n) >= atol + rtol * abs(n)
                if np.any(fail):
                    raise SimulationError(
                        f"Gradient check failed\n"
                        f"numeric values:\n{n[fail]}\n"
                        f"analytic values:\n{a[fail]}\n"
                    )

        logger.info("Gradient check passed")

        return grads

    def trange(self, sample_every=None, dt=None):
        """
        Create a vector of simulation step times matching probed data.

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
            warnings.warn(
                "`dt` is deprecated. Use `sample_every` instead.", DeprecationWarning
            )
            sample_every = dt

        period = 1 if sample_every is None else sample_every / self.dt
        steps = np.arange(1, self.n_steps + 1)
        return self.dt * steps[steps % period < 1]

    def close(self):
        """
        Close the simulation, freeing resources.

        Notes
        -----
        The simulation cannot be restarted after it is closed.
        """

        if not self.closed:
            self.keras_model = None
            self.tensor_graph = None
            self._closed_attrs = ["keras_model", "tensor_graph"]

            self.closed = True

    def get_name(self, obj):
        """
        Returns the standardized string name for input Nodes or output Probes.

        These are used when referring to inputs/outputs by string in Keras.

        Parameters
        ----------
        obj : `nengo.Node` or `nengo.Probe`
            Input Node or output Probe

        Returns
        -------
        name : str
            Name of the given object
        """

        if isinstance(obj, Node):
            if obj not in self.node_inputs:
                raise ValidationError(
                    f"{obj} is not an input Node (a nengo.Node with size_in==0), or is "
                    f"from a different network.",
                    "obj",
                )
        elif isinstance(obj, Probe):
            if obj not in self.tensor_graph.probe_arrays:
                raise ValidationError(f"{obj} is from a different network.", "obj")
        else:
            raise ValidationError(
                f"{obj} is of an unknown type ({type(obj)}); should be nengo.Node or "
                f"nengo.Probe",
                "obj",
            )
        return self.tensor_graph.io_names[obj]

    def _standardize_data(self, data, objects, broadcast_unary=False):
        """
        Converts data to the standardized input format (named string dicts).

        Parameters
        ----------
        data : `numpy.ndarray` or list or dict
            Input data in one of the formats supported by fit/predict/eval.
        objects : list of `nengo.Node` or `nengo.Probe`
            List of input Nodes or output Probes in the model (depending on which
            kind of data is being standardized).
        broadcast_unary: bool
            If True, singular (e.g. non-list/dict) inputs will be applied to all
            ``objects``, otherwise will only be applied to first item in ``objects``.

        Returns
        -------
        data : dict of {str: object}
            Elements of data reorganized into standardized data structure (named
            string dict).
        """

        if data is None:
            return data

        if not isinstance(data, (list, tuple, dict)):
            # convert unary inputs to length-1 lists
            data = [data]
            if broadcast_unary:
                data *= len(objects)

        if isinstance(data, (list, tuple)):
            if len(data) != len(objects):
                warnings.warn(
                    f"Number of elements ({len(data)}) in "
                    f"{[type(d).__name__ for d in data]} does not match number of "
                    f"{type(objects[0]).__name__}s ({len(objects)}); consider "
                    f"using an explicit input dictionary in this case, so that the "
                    f"assignment of data to objects is unambiguous."
                )

            # convert list to named dict
            data = {self.get_name(obj): val for obj, val in zip(objects, data)}
        elif isinstance(data, dict):
            # convert objects to string names
            data = {
                obj if isinstance(obj, str) else self.get_name(obj): val
                for obj, val in data.items()
            }

        return data

    def _generate_inputs(self, data=None, n_steps=None):
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

        if not isinstance(data, (list, tuple, dict, np.ndarray)) and not tf.is_tensor(
            data
        ):
            # data is some kind of generator, so we don't try to modify it (too many
            # different types of generators this could be)
            if n_steps is not None:
                raise SimulationError(
                    f"Cannot automatically add n_steps to generator with type "
                    f"{type(data)}; please specify n_steps manually as the first "
                    f"element in the values yielded from generator, remembering that "
                    f"it needs to be repeated to have shape (batch_size, 1)"
                )

            return data

        data = self._standardize_data(data, list(self.node_inputs.keys()))

        if len(data) == 0:
            data_batch = data_steps = None
        else:
            data_batch, data_steps = next(iter(data.values())).shape[:2]

        batch_size = self.minibatch_size if data_batch is None else data_batch
        if n_steps is None:
            if data_steps is None:
                raise ValidationError(
                    "Must specify either input data or n_steps", "data"
                )
            n_steps = data_steps

        input_vals = {}

        # fill in data for input nodes
        for node, output in self.tensor_graph.input_funcs.items():
            name = self.get_name(node)

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

        for name in data:
            if name not in input_vals:
                raise ValidationError(
                    f"Input contained entry for '{name}', which is not a valid input "
                    f"name",
                    "data",
                )

        # fill in n_steps
        input_vals["n_steps"] = np.resize(n_steps, (batch_size, 1)).astype(np.int32)

        return input_vals

    def _check_data(self, data, batch_size=None, n_steps=None, nodes=True):
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
        nodes : bool
            If True the data being validated is associated with Nodes, if False the
            data is associated with Probes.

        Notes
        -----
        This may modify ``data`` in-place, if it contains data that is not evenly
        divisible by ``Simulator.minibatch_size``.
        """

        if not isinstance(data, dict):
            # data is a generator, so don't perform validation
            return

        # make sure data is evenly divisible by minibatch size
        for k, v in data.items():
            try:
                data_batch = v.shape[0]
            except IndexError:
                # v is a scalar
                continue

            if (
                data_batch > self.minibatch_size
                and data_batch % self.minibatch_size != 0
            ):
                warnings.warn(
                    f"Number of elements in input data ({data_batch}) is not "
                    f"evenly divisible by Simulator.minibatch_size "
                    f"({self.minibatch_size}); input data will be truncated."
                )
                data_batch -= data_batch % self.minibatch_size
                data[k] = v[:data_batch]

        # exclude n_steps from normal data checking
        data_n_steps = data.get("n_steps", None)
        data = {k: val for k, val in data.items() if k != "n_steps"}

        for name, x in data.items():
            # check that name is valid
            if nodes:
                valid_names = [self.get_name(n) for n in self.node_inputs]
                if name not in valid_names:
                    raise ValidationError(
                        f"'{name}' is not a valid node name; perhaps the name is wrong "
                        f"(it should match the `label` on the Node), or this is not an "
                        f"input Node (a Node with size_in==0) in this network. "
                        f"Valid names are: {valid_names}.",
                        "data",
                    )
            else:
                valid_names = [self.get_name(p) for p in self.model.probes]
                if name not in valid_names:
                    raise ValidationError(
                        f"'{name}' is not a valid probe name; perhaps the name is "
                        f"wrong (it should match the `label` on the Probe), or this "
                        f"is not a Probe in this network. "
                        f"Valid names are: {valid_names}.",
                        "data",
                    )

            # generic shape checks
            if len(x.shape) != 3:
                raise ValidationError(
                    f"should have rank 3 (batch_size, n_steps, dimensions), found rank "
                    f"{len(x.shape)}",
                    f"{name} data",
                )
            if x.shape[0] < self.minibatch_size:
                raise ValidationError(
                    f"Batch size of data ({x.shape[0]}) less than Simulator "
                    f"`minibatch_size` ({self.minibatch_size})",
                    f"{name} data",
                )
            if nodes and x.shape[1] % self.unroll != 0:
                raise ValidationError(
                    f"The number of timesteps in input data ({x.shape[1]}) must be "
                    f"evenly divisible by unroll_simulation ({self.unroll})",
                    "data",
                )

        # check that shapes match the given values (if specified) or are
        # internally consistent (if not)
        args = [batch_size, n_steps]
        labels = ["batch size", "number of timesteps"]

        for i in range(2):
            if args[i] is None:
                if i == 1 and not nodes:
                    # we don't apply this check to probes, because target values can
                    # have different values for n_steps (as long as it matches what is
                    # expected by the loss function)
                    continue

                if len(data) > 0:
                    val = next(iter(data.values())).shape[i]
                for n, x in data.items():
                    if x.shape[i] != val:
                        raise ValidationError(
                            f"Elements have different {labels[i]}: {val} vs "
                            f"{x.shape[i]}",
                            "data",
                        )
            else:
                for n, x in data.items():
                    if x.shape[i] != args[i]:
                        raise ValidationError(
                            f"Data for {n} has {labels[i]}={x.shape[i]}, which does "
                            f"not match expected size ({args[i]})",
                            "data",
                        )

        if (
            n_steps is not None
            and not self.tensor_graph.use_loop
            and n_steps != self.unroll
        ):
            raise ValidationError(
                f"When use_loop=False, n_steps ({n_steps}) must exactly match "
                f"unroll_simulation ({self.unroll})",
                "n_steps",
            )

        if nodes:
            # validate special n_steps input

            if data_n_steps is None:
                raise ValidationError("Must specify 'n_steps' in input data", "data")
            if (
                batch_size is None
                and (data_n_steps.ndim != 2 or data_n_steps.shape[1] != 1)
            ) or (batch_size is not None and data_n_steps.shape != (batch_size, 1)):
                raise ValidationError(
                    f"'n_steps' has wrong shape; should be {(batch_size, 1)} (note that"
                    f" this is just the integer n_steps value repeated)",
                    "data",
                )
            if not np.all(data_n_steps == data_n_steps[0, 0]):
                raise ValidationError(
                    "'n_steps' should all have the same value", "data"
                )
            if n_steps is not None and not np.all(data_n_steps == n_steps):
                raise ValidationError(
                    "`n_steps` input does not match the requested number of steps",
                    "data",
                )

    @with_self
    def _update_steps(self):
        self._n_steps = tf.keras.backend.get_value(
            self.tensor_graph.get_tensor(self.model.step)
        ).item()
        self._time = self._n_steps * self.dt

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

    @property
    def seed(self):
        """The simulation random seed."""
        return self.tensor_graph.seed

    @require_open
    def __enter__(self):
        self._device_context = tf.device(self.tensor_graph.device)
        self._device_context.__enter__()

        self._keras_dtype = tf.keras.backend.floatx()
        tf.keras.backend.set_floatx(self.tensor_graph.dtype)

        return self

    @require_open
    def __exit__(self, *args):
        tf.keras.backend.set_floatx(self._keras_dtype)

        self._device_context.__exit__(*args)

        self.close()

    def __del__(self):
        """
        Raise a RuntimeWarning if the Simulator is deallocated while open.
        """

        if self.closed is not None and not self.closed:
            warnings.warn(
                f"Simulator with model={self.model} was deallocated while open. "
                f"Simulators should be closed manually to ensure resources are "
                f"properly freed.",
                RuntimeWarning,
            )
            self.close()

    def __getstate__(self):
        raise NotImplementedError(
            "TensorFlow does not support pickling; see "
            "https://www.nengo.ai/nengo-dl/simulator.html"
            "#saving-and-loading-parameters "
            "for information on how to save/load a NengoDL model."
        )

    def __getattribute__(self, name):
        if super().__getattribute__("closed") and name in super().__getattribute__(
            "_closed_attrs"
        ):
            raise SimulatorClosed(
                f"Cannot access Simulator.{name} after Simulator is closed"
            )

        return super().__getattribute__(name)


class SimulationData(collections.abc.Mapping):
    """
    Data structure used to access simulation data from the model.

    The main use case for this is to access Probe data; for example,
    ``probe_data = sim.data[my_probe]``.  However, it is also
    used to access the parameters of objects in the model; for example, after
    the model has been optimized via `.Simulator.fit`, the updated
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
                f"Object is not in parameters of model {self.sim.model}", str(obj)
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
            weights = (
                self.get_params((obj, "weights"))[0]
                if compat.conn_has_weights(obj)
                else None
            )

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
        Parameter values should be accessed through ``sim.data[my_obj]``
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
            zip(
                fetches.keys(), tf.keras.backend.batch_get_value(list(fetches.values()))
            )
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
