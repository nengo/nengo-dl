"""Tools for automatically converting a Keras model to a Nengo network."""

import collections
import copy
import logging
import warnings

import nengo
import numpy as np
import tensorflow as tf
from packaging import version

from nengo_dl import compat
from nengo_dl.config import configure_settings
from nengo_dl.neurons import LeakyReLU
from nengo_dl.simulator import Simulator
from nengo_dl.tensor_node import Layer, TensorNode

logger = logging.getLogger(__name__)


class Converter:
    """
    Converts a Keras model to a Nengo network composed of native Nengo objects.

    Parameters
    ----------
    model : ``tf.keras.Model``
        Keras model to be converted
    allow_fallback : bool
        If True, allow layers that cannot be converted to native Nengo
        objects to be added as a `.TensorNode` instead. Note that if this occurs, the
        converted Nengo network will only be runnable in the NengoDL simulator.
    inference_only : bool
        Allow layers to be converted in such a way that inference behaviour will
        match the source model but training behaviour will not.
        If ``inference_only=False`` then some
        layers cannot be converted to native Nengo objects (but you can
        still use ``allow_fallback=True`` to use a `.TensorNode` instead).
    max_to_avg_pool : bool
        If True, convert max pooling layers to average pooling layers. Note that this
        will change the behaviour of the network, so parameters will probably need to
        be re-trained in NengoDL. If ``max_to_avg_pool=False`` then max pooling layers
        cannot be converted to native Nengo objects (but you can
        still use ``allow_fallback=True`` to use a `.TensorNode` instead).
    split_shared_weights : bool
        In Keras, applying the same ``Layer`` object to different input layers will
        result in multiple instances of the given layer that share the same weights.
        This is not supported in Nengo. If ``split_shared_weights=True`` then those
        shared weights will be split into independent sets of weights. They will all
        be initialized to the same value, so the initial behaviour of the model will
        be unchanged, but if any further training is performed on the network then
        the weights in each of those instances may diverge.
    swap_activations : dict
        A dictionary mapping from TensorFlow activation functions or Nengo neuron types
        to TensorFlow activation functions or Nengo neuron types. This can be used to
        change all the activation types in a model to some other type. This is in
        addition to the default activation map (see `.LayerConverter`). It can be keyed
        based on either TensorFlow or Nengo activation types, and will be applied both
        before and after the default activation map, in order to support whatever swap
        type is most useful for a given model. In particular, ``swap_activations``
        can be useful for swapping rate neuron types to spiking neuron types,
        through e.g. ``{tf.nn.relu: nengo.SpikingRectifiedLinear()}`` or
        ``{nengo.RectifiedLinear(): nengo.SpikingRectifiedLinear()}``.  Or it can be
        used to swap activation types that don't have a native Nengo implementation,
        e.g. ``{tf.keras.activatons.elu: tf.keras.activations.relu}``.
    scale_firing_rates: float or dict
        Scales the inputs of neurons by ``x``, and the outputs by ``1/x``.
        The idea is that this parameter can be used to increase the firing rates of
        spiking neurons (by scaling the input), without affecting the overall output
        (because the output spikes are being scaled back down). Note that this is only
        strictly true for neuron types with linear activation functions (e.g. ReLU).
        Nonlinear neuron types (e.g. LIF) will be skewed by this linear scaling on the
        input/output. ``scale_firing_rates`` can be specified as a float, which will
        be applied to all layers in the model, or as a dictionary mapping Keras model
        layers to a scale factor, allowing different scale factors to be applied to
        different layers.
    synapse : float or `nengo.synapses.Synapse`
        Synaptic filter to be applied on the output of all neurons. This can be useful
        to smooth out the noise introduced by spiking neurons. Note, however, that
        increasing the synaptic filtering will make the network less responsive to
        rapid changes in the input, and you may need to present each input value
        for more timesteps in order to allow the network output to settle.
    temporal_model : bool
        Set to True if the Keras model contains a temporal
        dimension (e.g. the inputs and outputs of each layer have shape
        ``(batch_size, n_steps, ...)``). Note that all layers must be temporal, the
        Converter cannot handle models with a mix of temporal and non-temporal layers.

    Attributes
    ----------
    model : ``tf.keras.Model``
        The input Keras model (if input was a Sequential model then this will be the
        equivalent Functional model).
    net : `nengo.Network`
        The converted Nengo network.
    inputs : `.Converter.KerasTensorDict`
        Maps from Keras model inputs to input Nodes in the converted Nengo network.
        For example, ``my_node = Converter(my_model).inputs[my_model.input]``.
    outputs : `.Converter.KerasTensorDict`
        Maps from Keras model outputs to output Probes in the converted Nengo network.
        For example, ``my_probe = Converter(my_model).outputs[my_model.output]``.
    layers : `.Converter.KerasTensorDict`
        Maps from Keras model layers to the converted Nengo object.
        For example, ``my_neurons = Converter(my_model).layers[my_layer]``.
    """

    converters = {}

    def __init__(
        self,
        model,
        allow_fallback=True,
        inference_only=False,
        max_to_avg_pool=False,
        split_shared_weights=False,
        swap_activations=None,
        scale_firing_rates=None,
        synapse=None,
        temporal_model=False,
    ):
        self.allow_fallback = allow_fallback
        self.inference_only = inference_only
        self.max_to_avg_pool = max_to_avg_pool
        self.split_shared_weights = split_shared_weights
        self.swap_activations = Converter.TrackedDict(swap_activations or {})
        self.scale_firing_rates = scale_firing_rates
        self.synapse = synapse
        self.temporal_model = temporal_model
        self._layer_converters = {}

        # convert model
        self.net = self.get_converter(model).convert(None)

        # set model from the converter in case the converter has changed the model type
        # (i.e. if the model is sequential, it will be converted to a functional model)
        self.model = self.get_converter(model).layer

        if self.swap_activations.unused_keys():
            warnings.warn(
                f"swap_activations contained {self.swap_activations.unused_keys()}, "
                f"but there were no layers in the model with that activation type"
            )

        self.layers = self.net.layers

        with self.net:
            configure_settings(inference_only=self.inference_only)

            # convert inputs to input nodes
            self.inputs = Converter.KerasTensorDict()
            for input in self.model.inputs:
                input_obj = self.net.inputs[input]
                logger.info("Setting input node %s (%s)", input_obj, input)
                input_size = input_obj.size_in
                input_obj.size_in = 0
                input_obj.output = np.zeros(input_size)
                self.inputs[input] = input_obj

            # add probes to outputs
            self.outputs = Converter.KerasTensorDict()
            for output in self.model.outputs:
                output_obj = self.net.outputs[output]
                logger.info("Probing %s (%s)", output_obj, output)
                self.outputs[output] = nengo.Probe(
                    output_obj,
                    synapse=self.synapse
                    if isinstance(output_obj, nengo.ensemble.Neurons)
                    else None,
                )

    def verify(self, training=False, inputs=None, atol=1e-8, rtol=1e-5):
        """
        Verify that output of converted Nengo network matches the original Keras model.

        Parameters
        ----------
        training : bool
            If True, check that optimizing the converted Nengo network produces the same
            results as optimizing the original Keras model.
        inputs : list of `numpy.ndarray`
            Testing values for model inputs (if not specified, array of ones will be
            used).
        atol : float
            Absolute tolerance for difference between Nengo and Keras outputs.
        rtol : float
            Relative (to Nengo) tolerance for difference between nengo and Keras
            outputs.

        Returns
        -------
        success : bool
            True if output of Nengo network matches output of Keras model.

        Raises
        ------
        ValueError
            If output of Nengo network does not match output of Keras model.
        """

        epochs = 3
        batch_size = 2 if inputs is None else inputs[0].shape[0]
        n_steps = (
            1
            if not self.temporal_model
            else (10 if inputs is None else inputs[0].shape[1])
        )
        shape_prefix = (batch_size, n_steps) if self.temporal_model else (batch_size,)

        inp_vals = (
            [
                np.ones(shape_prefix + x.shape[len(shape_prefix) :])
                for x in self.model.inputs
            ]
            if inputs is None
            else inputs
        )

        # get keras model output
        if training:
            out_vals = [
                np.ones(shape_prefix + x.shape[len(shape_prefix) :])
                for x in self.model.outputs
            ]
            self.model.compile(optimizer=tf.optimizers.SGD(0.1), loss=tf.losses.mse)
            self.model.fit(inp_vals, out_vals, epochs=epochs)

        keras_out = self.model.predict(inp_vals)

        if not isinstance(keras_out, (list, tuple)):
            keras_out = [keras_out]

        # get nengo sim output
        inp_vals = [np.reshape(x, (batch_size, n_steps, -1)) for x in inp_vals]

        with Simulator(self.net, minibatch_size=batch_size) as sim:
            if training:
                keras_params = sum(
                    np.prod(w.shape) for w in self.model.trainable_weights
                )
                nengo_params = sum(
                    np.prod(w.shape) for w in sim.keras_model.trainable_weights
                )
                if keras_params != nengo_params:
                    raise ValueError(
                        f"Number of trainable parameters in Nengo network "
                        f"({nengo_params}) does not match number of trainable "
                        f"parameters in Keras model ({keras_params})"
                    )

                out_vals = [np.reshape(x, (batch_size, n_steps, -1)) for x in out_vals]
                sim.compile(optimizer=tf.optimizers.SGD(0.1), loss=tf.losses.mse)
                sim.fit(inp_vals, out_vals, epochs=epochs)

            sim_out = sim.predict(inp_vals)

        for i, out in enumerate(self.model.outputs):
            keras_vals = np.ravel(keras_out[i])
            nengo_vals = np.ravel(sim_out[self.outputs[out]])
            fails = np.logical_not(
                np.isclose(keras_vals, nengo_vals, atol=atol, rtol=rtol)
            )
            if np.any(fails):
                logger.info("Verification failure")
                logger.info("Keras:\n%s", keras_vals[fails])
                logger.info("Nengo:\n%s", nengo_vals[fails])
                raise ValueError(
                    "Output of Keras model does not match output of converted "
                    "Nengo network (max difference="
                    f"{max(abs(keras_vals[fails] - nengo_vals[fails])):.2E}; "
                    "set log level to INFO to see all failures)"
                )

        return True

    def get_converter(self, layer):
        """
        Get instantiated `.LayerConverter` for the given ``Layer`` instance.

        Note that this caches the results, so calling the function multiple times
        with the same Layer instance will return the same LayerConverter instance.

        Parameters
        ----------
        layer : ``tf.keras.layers.Layer``
            The Keras Layer being converted.

        Returns
        -------
        converter : `.LayerConverter`
            LayerConverter class for converting ``layer`` to Nengo objects.
        """

        if layer in self._layer_converters:
            # already have an instantiated converter for this layer
            converter = self._layer_converters[layer]
            if converter.has_weights and not self.split_shared_weights:
                # TODO: allow fallback
                raise ValueError(
                    f"Multiple applications of layer {layer} detected; this is not "
                    f"supported unless split_shared_weights=True"
                )
            return converter

        # check if there is a registered builder for this layer type
        ConverterClass = self.converters.get(type(layer), None)

        # perform custom checks in layer converters
        if ConverterClass is None:
            convertible = False
            error_msg = (
                f"Layer type {type(layer).__name__} does not have a registered "
                f"converter"
            )
        else:
            convertible, error_msg = ConverterClass.convertible(layer, self)

        if not convertible:
            # this means that there is no LayerConverter compatible with this layer
            # (either because it has an unknown type, or it failed the ``.convertible``
            # check due to its internal parameterization)
            can_fallback = ConverterClass is None or ConverterClass.allow_fallback
            if self.allow_fallback and can_fallback:
                warnings.warn(
                    f"{error_msg + '. ' if error_msg else ''}"
                    f"Falling back to TensorNode."
                )
                ConverterClass = self.converters[None]
            else:
                msg = (
                    f"{error_msg + '. ' if error_msg else ''}Unable to convert layer "
                    f"'{layer.name}' to native Nengo objects; "
                )
                if not self.allow_fallback and can_fallback:
                    msg += (
                        "set allow_fallback=True if you would like to use a "
                        "TensorNode instead, or "
                    )
                msg += (
                    "consider registering a custom LayerConverter for this layer type."
                )
                raise TypeError(msg)

        converter = ConverterClass(layer, self)

        self._layer_converters[layer] = converter

        return converter

    @classmethod
    def register(cls, keras_layer):
        """
        A decorator for adding a class to the converter registry.

        Parameters
        ----------
        keras_layer : ``tf.keras.layers.Layer``
            The Layer associated with the conversion function being registered.
        """

        def register_converter(convert_cls):
            if keras_layer in cls.converters:
                warnings.warn(
                    f"Layer '{keras_layer}' already has a converter. Overwriting."
                )

            cls.converters[keras_layer] = convert_cls

            return convert_cls

        return register_converter

    class KerasTensorDict(collections.abc.Mapping):
        """
        A dictionary-like object that has extra logic to handle Layer/Tensor keys.
        """

        def __init__(self):
            self.dict = {}

        def _get_key(self, key):
            if isinstance(key, tf.keras.layers.Layer):
                if len(key.inbound_nodes) > 1:
                    raise KeyError(
                        f"Layer {key} is ambiguous because it has been called multiple "
                        "times; use a specific set of layer outputs as key instead"
                    )

                # get output tensor
                key = key.output

            key = tuple(x.ref() if tf.is_tensor(x) else x for x in tf.nest.flatten(key))

            return key

        def __setitem__(self, key, val):
            self.dict[self._get_key(key)] = val

        def __getitem__(self, key):
            return self.dict[self._get_key(key)]

        def __iter__(self):
            return iter(self.dict)

        def __len__(self):
            return len(self.dict)

    class TrackedDict(collections.abc.Mapping):
        """
        A dictionary-like object that keeps track of which keys have been accessed.
        """

        def __init__(self, dict):
            self.dict = dict
            self.read = set()

        def __getitem__(self, key):
            self.read.add(key)
            return self.dict[key]

        def __iter__(self):
            return iter(self.dict)

        def __len__(self):
            return len(self.dict)

        def unused_keys(self):
            """Returns any keys in the dictionary that have never been read."""
            return set(self.dict.keys()) - self.read


class LayerConverter:
    """
    Base class for converter classes, which contain the logic for mapping some Keras
    layer type to Nengo objects.

    Subclasses must implement the `.LayerConverter.convert` method. They may optionally
    extend `.LayerConverter.convertible` if this layer type requires custom logic for
    whether or not a layer can be converted to Nengo objects.

    Subclasses should override the ``unsupported_args`` class parameter if there are
    certain non-default Layer attributes that are not supported by the converter.
    This is a list of names for attributes that must have the default value for the
    layer to be convertible. The default is assumed to be ``None``, or a tuple of
    ``("attribute_name", default_value)`` can be specified. If there are parameters
    that are supported in inference mode but not in training mode, they should be
    added to the ``unsupported_training_args`` parameter.

    Subclasses should override the ``has_weights`` class parameter if the layer type
    being converted contains internal weights (this affects how the converter will
    handle duplicate layers).

    Parameters
    ----------
    layer : ``tf.keras.layers.Layer``
        The Layer object being converted.
    converter : `.Converter`
        The parent Converter class running the conversion process.
    """

    # maps from TensorFlow activation functions to Nengo neuron types
    activation_map = {
        None: None,
        tf.keras.activations.linear: None,
        tf.keras.activations.relu: nengo.RectifiedLinear(),
        tf.nn.relu: nengo.RectifiedLinear(),
        tf.keras.activations.sigmoid: nengo.Sigmoid(tau_ref=1),
        tf.nn.sigmoid: nengo.Sigmoid(tau_ref=1),
    }
    if version.parse(nengo.__version__) > version.parse("3.0.0"):
        activation_map.update(
            {
                tf.keras.activations.tanh: compat.Tanh(tau_ref=1),
                tf.nn.tanh: compat.Tanh(tau_ref=1),
            }
        )

    # attributes of the Keras layer that are not supported for non-default values.
    # the default value is assumed to be None, or a tuple of
    # ("attr_name", default_value) can be specified
    unsupported_args = []
    # attributes that are supported in inference_only mode but otherwise not
    unsupported_training_args = []

    # whether or not this layer contains trainable weights (this indicates whether
    # this layer is affected by split_shared_weights)
    has_weights = False

    # whether or not this layer supports a TensorNode fallback
    allow_fallback = True

    def __init__(self, layer, converter):
        self.layer = layer
        self.converter = converter

    def add_nengo_obj(self, node_id, biases=None, activation=None):
        """
        Builds a Nengo object for the given Node of this layer.

        Parameters
        ----------
        node_id : int
            The index of the Keras Node currently being built on this layer.
        biases : `numpy.ndarray` or None
            If not None, add trainable biases with the given value.
        activation : callable or None
            The TensorFlow activation function to be used (``None`` will be
            interpreted as linear activation).

        Returns
        -------
        obj : `nengo.Node` or `nengo.ensemble.Neurons` or `nengo_dl.TensorNode`
            The Nengo object whose output corresponds to the output of the given Keras
            Node.
        """
        name = f"{self.layer.name}.{node_id}"

        # apply manually specified swaps
        activation = self.converter.swap_activations.get(activation, activation)

        if activation in self.activation_map or isinstance(
            activation, nengo.neurons.NeuronType
        ):
            activation = self.activation_map.get(activation, activation)

            # apply any nengo->nengo swaps
            activation = self.converter.swap_activations.get(activation, activation)

            if activation is None:
                # linear activation, uses a passthrough Node
                obj = nengo.Node(
                    size_in=np.prod(self.output_shape(node_id)), label=name
                )
            else:
                # use ensemble to implement the appropriate neuron type

                # apply firing rate scaling
                if self.converter.scale_firing_rates is None:
                    scale_firing_rates = None
                elif isinstance(self.converter.scale_firing_rates, dict):
                    # look up specific layer rate
                    scale_firing_rates = self.converter.scale_firing_rates.get(
                        self.layer, None
                    )
                else:
                    # constant scale applied to all layers
                    scale_firing_rates = self.converter.scale_firing_rates

                if scale_firing_rates is None:
                    gain = 1
                else:
                    gain = scale_firing_rates
                    if biases is not None:
                        biases *= scale_firing_rates

                    if hasattr(activation, "amplitude"):
                        # copy activation so that we can change amplitude without
                        # affecting other instances
                        activation = copy.copy(activation)

                        # bypass read-only protection
                        type(activation).amplitude.data[
                            activation
                        ] /= scale_firing_rates
                    else:
                        warnings.warn(
                            f"Firing rate scaling being applied to activation type "
                            f"that does not support amplitude "
                            f"({type(activation).__name__}); "
                            f"this will change the output"
                        )

                obj = nengo.Ensemble(
                    np.prod(self.output_shape(node_id)),
                    1,
                    neuron_type=activation,
                    gain=nengo.dists.Choice([gain]),
                    bias=nengo.dists.Choice([0]) if biases is None else biases,
                    label=name,
                ).neurons
                if biases is None:
                    # ensembles always have biases, so if biases=None we just use
                    # all-zero biases and mark them as non-trainable
                    self.set_trainable(obj, False)
        elif self.converter.allow_fallback:
            warnings.warn(
                f"Activation type {activation} does not have a native Nengo "
                f"equivalent; falling back to a TensorNode"
            )
            obj = TensorNode(
                activation,
                shape_in=self.output_shape(node_id),
                pass_time=False,
                label=name,
            )
        else:
            raise TypeError(f"Unsupported activation type ({self.layer.activation})")

        if biases is not None and isinstance(obj, (nengo.Node, TensorNode)):
            # obj doesn't have its own biases, so use a connection from a constant node
            # (so that the bias values will be trainable)
            bias_node = nengo.Node([1], label=f"{name}.bias")
            nengo.Connection(bias_node, obj, transform=biases[:, None], synapse=None)

        logger.info("Created %s (size=%d)", obj, obj.size_out)

        return obj

    def add_connection(self, node_id, obj, input_idx=0, trainable=False, **kwargs):
        """
        Adds a Connection from one of the inputs of the Node being built to the
        Nengo object.

        Parameters
        ----------
        node_id : int
            The index of the Keras Node currently being built on this layer.
        obj : ``NengoObject``
            The Nengo object implementing this Node.
        input_idx : int
            Which of the inputs we want to add a Connection for (in the case of
            layers that have multiple inputs).
        trainable : bool
            Whether or not the weights associated with the created Connection
            should be trainable.
        kwargs : dict
            Will be passed on to `nengo.Connection`.

        Returns
        -------
        conn : `nengo.Connection`
            The constructed Connection object.
        """

        pre = self.get_input_obj(node_id, tensor_idx=input_idx)

        kwargs.setdefault(
            "synapse",
            self.converter.synapse if isinstance(pre, nengo.ensemble.Neurons) else None,
        )

        conn = nengo.Connection(
            pre,
            obj,
            **kwargs,
        )
        self.set_trainable(conn, trainable)

        logger.info("Connected %s to %s (trainable=%s)", conn.pre, conn.post, trainable)

        return conn

    def get_input_obj(self, node_id, tensor_idx=0):
        """
        Returns the Nengo object corresponding to the given input of this layer.

        Parameters
        ----------
        node_id : int
            The index of the Keras Node currently being built on this layer.
        tensor_idx : int
            The index of the input we want to look up (for layers with multiple inputs).

        Returns
        -------
        obj : `nengo.Node` or `nengo.ensemble.Neurons` or `nengo_dl.TensorNode`
            The Nengo object whose output corresponds to the given input of this layer.
        """
        input_node = self.layer.inbound_nodes[node_id]

        input_tensors = input_node.input_tensors

        if isinstance(input_tensors, (list, tuple)):
            tensor = input_tensors[tensor_idx]
        else:
            assert tensor_idx == 0
            tensor = input_tensors

        input_layer, input_node_id, input_tensor_id = tensor._keras_history

        return nengo.Network.context[-1].layer_map[input_layer][input_node_id][
            input_tensor_id
        ]

    def _get_shape(self, input_output, node_id, full_shape=False):
        """
        Looks up the input or output shape of this Node.

        Parameters
        ----------
        input_output : "input" or "output"
            Whether we want the input or output shape.
        node_id : int
            The node whose shape we want to look up.
        full_shape : bool
            Whether or not the returned shape should include the batch/time dimensions.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one input/output, or a list of shapes
            if the node as multiple inputs/outputs.
        """

        # note: layer.get_input/output_shape_at is generally equivalent to
        # layer.input/output_shape, except when the layer is called multiple times
        # with different shapes, in which case input/output_shape is not well defined
        func = getattr(self.layer, f"get_{input_output}_shape_at")

        # get the shape
        shape = func(node_id)

        if not full_shape:
            to_skip = 2 if self.converter.temporal_model else 1
            if isinstance(shape, list):
                # multiple inputs/outputs; trim the batch from each one
                shape = [s[to_skip:] for s in shape]
            else:
                shape = shape[to_skip:]

        return shape

    def input_shape(self, node_id, full_shape=False):
        """
        Returns the input shape of the given node.

        Parameters
        ----------
        node_id : int
            The node whose shape we want to look up.
        full_shape : bool
            Whether or not the returned shape should include the batch/time dimensions.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one input, or a list of shapes
            if the node as multiple inputs.
        """
        return self._get_shape("input", node_id, full_shape=full_shape)

    def output_shape(self, node_id, full_shape=False):
        """
        Returns the output shape of the given node.

        Parameters
        ----------
        node_id : int
            The node whose shape we want to look up.
        full_shape : bool
            Whether or not the returned shape should include the batch/time dimensions.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one output, or a list of shapes
            if the node as multiple outputs.
        """
        return self._get_shape("output", node_id, full_shape=full_shape)

    @staticmethod
    def set_trainable(obj, trainable):
        """
        Set trainable config attribute of object.

        Parameters
        ----------
        obj : ``NengoObject``
            The object to be assigned.
        trainable: bool
            Trainability value of ``obj``.
        """

        nengo.Network.context[-1].config[obj].trainable = trainable

    @classmethod
    def convertible(cls, layer, converter):
        """
        Check whether the given Keras layer is convertible to native Nengo objects.

        Parameters
        ----------
        layer : ``tf.keras.layers.Layer``
            The Keras Layer we want to convert.
        converter : `.Converter`
            The Converter object running the conversion process.

        Returns
        -------
        convertible : bool
            True if the layer can be converted to native Nengo objects, else False.
        """
        # check if the layer uses any unsupported arguments
        unsupported = cls.unsupported_args
        if not converter.inference_only:
            unsupported = unsupported + cls.unsupported_training_args
        for arg in unsupported:
            if isinstance(arg, str):
                default = None
            else:
                arg, default = arg

            val = getattr(layer, arg)
            if val != default:
                msg = (
                    f"{layer.name}.{arg} has value {val} != {default}, "
                    "which is not supported"
                )
                if arg in cls.unsupported_training_args:
                    msg += " (unless inference_only=True)"
                return False, msg

        return True, None

    def convert(self, node_id):
        """
        Convert the given node of this layer to Nengo objects

        Parameters
        ----------
        node_id : int
            The index of the inbound node to be converted.

        Returns
        -------
        output : ``NengoObject``
            Nengo object whose output corresponds to the output of the Keras layer node.
        """
        raise NotImplementedError("Subclasses must implement convert")


@Converter.register(compat.Functional)
class ConvertModel(LayerConverter):
    """Convert ``tf.keras.Model`` to Nengo objects."""

    def convert(self, node_id):
        logger.info("=" * 30)
        logger.info("Converting model %s", self.layer.name)

        # functional models should already have been built when the model
        # was instantiated
        assert self.layer.built

        # trace the model to find all the nodes that need to be built into
        # the Nengo network
        ordered_nodes, _ = compat._build_map(self.layer.outputs)
        layer_ids = [
            (node.layer, node.layer.inbound_nodes.index(node)) for node in ordered_nodes
        ]

        with nengo.Network(
            label=self.layer.name + ("" if node_id is None else f".{node_id}")
        ) as net:
            # add the "trainable" attribute to all objects
            configure_settings(trainable=None)

            net.layer_map = collections.defaultdict(dict)

            for layer, layer_node_id in layer_ids:
                # should never be rebuilding the same node
                assert layer_node_id not in net.layer_map[layer]

                logger.info("-" * 30)
                logger.info("Converting layer %s node %d", layer.name, layer_node_id)

                # get the layerconverter object
                layer_converter = self.converter.get_converter(layer)

                # build the Nengo objects
                nengo_layer = layer_converter.convert(layer_node_id)
                assert isinstance(
                    nengo_layer,
                    (nengo.Node, nengo.ensemble.Neurons, TensorNode, nengo.Network),
                )

                # add output of layer_converter to layer_map
                net.layer_map[layer][layer_node_id] = (
                    list(nengo_layer.outputs.values())
                    if isinstance(nengo_layer, nengo.Network)
                    else [nengo_layer]
                )

            # data structures to track converted objects
            net.inputs = Converter.KerasTensorDict()
            for input in self.layer.inputs:
                input_layer, input_node_id, input_tensor_id = input._keras_history
                net.inputs[input] = net.layer_map[input_layer][input_node_id][
                    input_tensor_id
                ]

            net.outputs = Converter.KerasTensorDict()
            for output in self.layer.outputs:
                output_layer, output_node_id, output_tensor_id = output._keras_history
                net.outputs[output] = net.layer_map[output_layer][output_node_id][
                    output_tensor_id
                ]

            net.layers = Converter.KerasTensorDict()
            for layer in self.layer.layers:
                for layer_node_id, node_outputs in net.layer_map[layer].items():
                    for nengo_obj in node_outputs:
                        output_tensors = layer.inbound_nodes[
                            layer_node_id
                        ].output_tensors
                        net.layers[output_tensors] = nengo_obj

        if node_id is not None:
            # add incoming connections in parent network
            for i, inp in enumerate(net.inputs.values()):
                self.add_connection(node_id, inp, input_idx=i)

        logger.info("=" * 30)

        return net


@Converter.register(tf.keras.Sequential)
class ConvertSequential(ConvertModel):
    """Convert ``tf.keras.Sequential`` to Nengo objects."""

    def __init__(self, seq_model, converter):
        # convert sequential model to functional model
        warnings.warn(
            "Converting sequential model to functional model; "
            "use `Converter.model` to refer to the functional model (rather than the "
            "original sequential model) when working with the output of the Converter"
        )
        input_shape = seq_model.layers[0].input_shape

        inp = x = tf.keras.Input(batch_shape=input_shape)
        for layer in seq_model.layers:
            x = layer(x)

        func_model = tf.keras.Model(inp, x)

        super().__init__(func_model, converter)


@Converter.register(None)
class ConvertFallback(LayerConverter):
    """
    Convert layers which do not have a native Nengo equivalent into a
    `.TensorNode`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # copy layer, so that any changes to the layer (e.g., rebuilding it), do not
        # affect the source model
        layer = self.layer.__class__.from_config(self.layer.get_config())
        if self.layer.built:
            layer.build(self.input_shape(0, full_shape=True))
            layer.set_weights(self.layer.get_weights())

        self.tensor_layer = Layer(layer)

    def convert(self, node_id):
        logger.info("Using TensorNode %s", self.tensor_layer)

        input_obj = self.get_input_obj(node_id)
        output = self.tensor_layer(
            input_obj, shape_in=self.input_shape(node_id), label=self.layer.name
        )

        logger.info("Applying to %s, created %s", input_obj, output)

        return output


class ConvertAvgPool(LayerConverter):
    """Base class for converting average pooling layers to Nengo objects."""

    # "same" padding not supported because TensorFlow average pooling does not count
    # the padded cells in the average, which we don't have a way to do using Convolution
    unsupported_args = [("padding", "valid")]

    def convert(self, node_id, dimensions):
        output = self.add_nengo_obj(node_id)

        def to_tuple(val):
            if isinstance(val, int):
                return (val,) * dimensions

            return val

        spatial_shape = (
            self.input_shape(node_id)[:-1]
            if self.layer.data_format == "channels_last"
            else self.input_shape(node_id)[1:]
        )

        # the default values here are for GlobalAveragePooling (which doesn't have
        # these attributes)
        pool_size = to_tuple(getattr(self.layer, "pool_size", spatial_shape))
        padding = getattr(self.layer, "padding", "valid")
        strides = to_tuple(getattr(self.layer, "strides", 1))

        # the idea here is that we set up a convolutional transform with weights 1/n,
        # which will have the effect of implementing average pooling
        n_filters = (
            self.output_shape(node_id)[-1]
            if self.layer.data_format == "channels_last"
            else self.output_shape(node_id)[0]
        )
        n_pool = np.prod(pool_size)
        kernel = np.reshape(
            [np.eye(n_filters) / n_pool] * n_pool, pool_size + (n_filters, n_filters)
        )
        pool_conv = nengo.Convolution(
            n_filters=n_filters,
            input_shape=self.input_shape(node_id),
            padding=padding,
            strides=strides,
            kernel_size=pool_size,
            init=kernel,
            channels_last=self.layer.data_format == "channels_last",
        )
        self.add_connection(node_id, output, transform=pool_conv)

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if (
            isinstance(
                layer,
                (
                    tf.keras.layers.MaxPool1D,
                    tf.keras.layers.MaxPool2D,
                    tf.keras.layers.MaxPool3D,
                    tf.keras.layers.GlobalMaxPool1D,
                    tf.keras.layers.GlobalMaxPool2D,
                    tf.keras.layers.GlobalMaxPool3D,
                ),
            )
            and not converter.max_to_avg_pool
        ):
            msg = (
                "Cannot convert max pooling layers to native Nengo objects; consider "
                "setting max_to_avg_pool=True to use average pooling instead"
            )
            return False, msg

        unsupported = cls.unsupported_args
        if not hasattr(layer, "padding"):
            # global layers don't have this attribute, so we temporarily remove it
            # from the unsupported args
            cls.unsupported_args = []

        try:
            convertible = super().convertible(layer, converter)
        finally:
            # reset the unsupported args
            cls.unsupported_args = unsupported

        return convertible


@Converter.register(tf.keras.layers.Activation)
class ConvertActivation(LayerConverter):
    """Convert ``tf.keras.layers.Activation`` to Nengo objects."""

    def convert(self, node_id):
        output = self.add_nengo_obj(node_id, activation=self.layer.activation)

        self.add_connection(node_id, output)

        return output


@Converter.register(tf.keras.layers.Add)
class ConvertAdd(LayerConverter):
    """Convert ``tf.keras.layers.Add`` to Nengo objects."""

    def convert(self, node_id):
        output = self.add_nengo_obj(node_id)
        for i in range(len(self.layer.input)):
            self.add_connection(node_id, output, input_idx=i)

        return output


@Converter.register(tf.keras.layers.Average)
class ConvertAverage(LayerConverter):
    """Convert ``tf.keras.layers.Average`` to Nengo objects."""

    def convert(self, node_id):
        output = self.add_nengo_obj(node_id)
        for i in range(len(self.layer.input)):
            self.add_connection(
                node_id, output, input_idx=i, transform=1 / len(self.layer.input)
            )

        return output


@Converter.register(tf.keras.layers.AvgPool1D)
@Converter.register(tf.keras.layers.MaxPool1D)
@Converter.register(tf.keras.layers.GlobalAvgPool1D)
@Converter.register(tf.keras.layers.GlobalMaxPool1D)
class ConvertAvgPool1D(ConvertAvgPool):
    """
    Convert ``tf.keras.layers.AvgPool1D`` to Nengo objects.

    Also works for ``tf.keras.layers.GlobalAvgPool1D``, and
    ``tf.keras.layers.MaxPool1D``/``GlobalMaxPool1D`` (if ``max_to_avg_pool=True``).
    """

    def convert(self, node_id):
        return super().convert(node_id, dimensions=1)


@Converter.register(tf.keras.layers.AvgPool2D)
@Converter.register(tf.keras.layers.MaxPool2D)
@Converter.register(tf.keras.layers.GlobalAvgPool2D)
@Converter.register(tf.keras.layers.GlobalMaxPool2D)
class ConvertAvgPool2D(ConvertAvgPool):
    """
    Convert ``tf.keras.layers.AvgPool2D`` to Nengo objects.

    Also works for ``tf.keras.layers.GlobalAvgPool2D``, and
    ``tf.keras.layers.MaxPool2D``/``GlobalMaxPool2D`` (if ``max_to_avg_pool=True``).
    """

    def convert(self, node_id):
        return super().convert(node_id, dimensions=2)


@Converter.register(tf.keras.layers.AvgPool3D)
@Converter.register(tf.keras.layers.MaxPool3D)
@Converter.register(tf.keras.layers.GlobalAvgPool3D)
@Converter.register(tf.keras.layers.GlobalMaxPool3D)
class ConvertAvgPool3D(ConvertAvgPool):
    """
    Convert ``tf.keras.layers.AvgPool3D`` to Nengo objects.

    Also works for ``tf.keras.layers.GlobalAvgPool3D``, and
    ``tf.keras.layers.MaxPool3D``/``GlobalMaxPool3D`` (if ``max_to_avg_pool=True``).
    """

    def convert(self, node_id):
        return super().convert(node_id, dimensions=3)


@Converter.register(compat.BatchNormalizationV1)
@Converter.register(compat.BatchNormalizationV2)
class ConvertBatchNormalization(LayerConverter):
    """Convert ``tf.keras.layers.BatchNormalization`` to Nengo objects."""

    def convert(self, node_id):
        # look up the batch normalization parameters
        if self.layer.scale:
            gamma = tf.keras.backend.get_value(self.layer.gamma)
        else:
            gamma = 1

        if self.layer.center:
            beta = tf.keras.backend.get_value(self.layer.beta)
        else:
            beta = 0

        mean, variance = tf.keras.backend.batch_get_value(
            (self.layer.moving_mean, self.layer.moving_variance)
        )

        # compute the fixed affine transform values for this layer
        variance += self.layer.epsilon

        stddev = np.sqrt(variance)

        scale = gamma / stddev
        bias = beta - scale * mean

        # build output object
        output = self.add_nengo_obj(node_id)

        # the batch normalization parameters will be n-dimensional, where n is the
        # length of the axis specified in the batch normalization layer. so we need
        # to set up a connection structure so that all the elements of the output
        # corresponding to one of those axis elements will share the same parameter

        assert len(self.layer.axis) == 1
        assert self.layer.axis[0] > 0
        axis = self.layer.axis[0] - 1  # not counting batch dimension

        idxs = np.arange(output.size_in).reshape(self.output_shape(node_id))
        slices = [slice(None) for _ in range(len(idxs.shape))]

        # broadcast scale/bias along the non-axis dimensions, so that we can apply the
        # same scale/bias to all those elements
        broadcast_scale = np.zeros(self.output_shape(node_id))
        broadcast_bias = np.zeros(self.output_shape(node_id))
        for i in range(idxs.shape[axis]):
            slices[axis] = i
            broadcast_scale[tuple(slices)] = scale[i]
            broadcast_bias[tuple(slices)] = bias[i]
        broadcast_scale = np.ravel(broadcast_scale)
        broadcast_bias = np.ravel(broadcast_bias)

        # connect up bias node to output
        bias_node = nengo.Node(
            broadcast_bias, label=f"{self.layer.name}.{node_id}.bias"
        )
        conn = nengo.Connection(bias_node, output, synapse=None)
        self.set_trainable(conn, False)

        # connect input to output, scaled by the batch normalization scale
        self.add_connection(node_id, output, transform=broadcast_scale)

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if not converter.inference_only and layer.trainable:
            msg = (
                "Cannot convert BatchNormalization layer to native Nengo objects "
                "unless inference_only=True or layer.trainable=False"
            )
            return False, msg

        return super().convertible(layer, converter)


@Converter.register(tf.keras.layers.Concatenate)
class ConvertConcatenate(LayerConverter):
    """Convert ``tf.keras.layers.Concatenate`` to Nengo objects."""

    def convert(self, node_id):
        output = self.add_nengo_obj(node_id)

        # axis-1 because not counting batch dimension
        axis = self.layer.axis - 1 if self.layer.axis > 0 else self.layer.axis

        idxs = np.arange(np.prod(self.output_shape(node_id))).reshape(
            self.output_shape(node_id)
        )
        slices = [slice(None) for _ in range(idxs.ndim)]
        offsets = np.cumsum([shape[axis] for shape in self.input_shape(node_id)])
        offsets = np.concatenate(([0], offsets))

        for i in range(len(self.layer.input)):
            slices[axis] = slice(offsets[i], offsets[i + 1])
            self.add_connection(
                node_id, output[np.ravel(idxs[tuple(slices)])], input_idx=i
            )

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if layer.axis == 0:
            msg = "Cannot concatenate along batch dimension (axis 0)"
            return False, msg

        return super().convertible(layer, converter)


class ConvertConv(LayerConverter):
    """Base class for converting convolutional layers to Nengo objects."""

    has_weights = True

    def convert(self, node_id, dimensions):
        # look up parameter values from source layer
        if self.layer.use_bias:
            kernel, biases = tf.keras.backend.batch_get_value(
                (self.layer.kernel, self.layer.bias)
            )
        else:
            kernel = tf.keras.backend.get_value(self.layer.kernel)
            biases = None

        # create nengo object to implement activation function
        output = self.add_nengo_obj(node_id, activation=self.layer.activation)

        if self.layer.use_bias:
            # conv layer biases are per-output-channel, rather than per-output-element,
            # so we need to set up a nengo connection structure that will have one
            # bias parameter shared across all the spatial dimensions

            # add trainable bias weights
            bias_node = nengo.Node([1], label=f"{self.layer.name}.{node_id}.bias")
            bias_relay = nengo.Node(
                size_in=len(biases), label=f"{self.layer.name}.{node_id}.bias_relay"
            )
            nengo.Connection(
                bias_node, bias_relay, transform=biases[:, None], synapse=None
            )

            # use a non-trainable sparse transform to broadcast biases along all
            # non-channel dimensions
            broadcast_indices = []
            idxs = np.arange(np.prod(self.output_shape(node_id))).reshape(
                self.output_shape(node_id)
            )
            slices = [slice(None) for _ in range(len(self.output_shape(node_id)))]
            n_spatial = np.prod(
                self.output_shape(node_id)[:-1]
                if self.layer.data_format == "channels_last"
                else self.output_shape(node_id)[1:]
            )
            axis = -1 if self.layer.data_format == "channels_last" else 0
            for i in range(self.output_shape(node_id)[axis]):
                slices[axis] = i
                broadcast_indices.extend(
                    tuple(zip(np.ravel(idxs[tuple(slices)]), [i] * n_spatial))
                )
            conn = nengo.Connection(
                bias_relay,
                output,
                transform=nengo.Sparse(
                    (output.size_in, bias_relay.size_out), indices=broadcast_indices
                ),
                synapse=None,
            )
            self.set_trainable(conn, False)

        # set up a convolutional transform that matches the layer parameters
        transform = nengo.Convolution(
            n_filters=self.layer.filters,
            input_shape=self.input_shape(node_id),
            kernel_size=self.layer.kernel_size,
            strides=self.layer.strides,
            padding=self.layer.padding,
            channels_last=self.layer.data_format == "channels_last",
            init=kernel,
        )

        self.add_connection(node_id, output, transform=transform, trainable=True)

        return output


@Converter.register(tf.keras.layers.Conv1D)
class ConvertConv1D(ConvertConv):
    """Convert ``tf.keras.layers.Conv1D`` to Nengo objects."""

    unsupported_args = [
        ("dilation_rate", (1,)),
    ]
    unsupported_training_args = [
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]

    def convert(self, node_id):
        return super().convert(node_id, dimensions=1)


@Converter.register(tf.keras.layers.Conv2D)
class ConvertConv2D(ConvertConv):
    """Convert ``tf.keras.layers.Conv2D`` to Nengo objects."""

    unsupported_args = [
        ("dilation_rate", (1, 1)),
    ]
    unsupported_training_args = [
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]

    def convert(self, node_id):
        return super().convert(node_id, dimensions=2)


@Converter.register(tf.keras.layers.Conv3D)
class ConvertConv3D(ConvertConv):
    """Convert ``tf.keras.layers.Conv3D`` to Nengo objects."""

    unsupported_args = [
        ("dilation_rate", (1, 1, 1)),
    ]
    unsupported_training_args = [
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]

    def convert(self, node_id):
        return super().convert(node_id, dimensions=3)


@Converter.register(tf.keras.layers.Dense)
class ConvertDense(LayerConverter):
    """Convert ``tf.keras.layers.Dense`` to Nengo objects."""

    unsupported_training_args = [
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]

    has_weights = True

    def convert(self, node_id):
        # look up parameter values from source layer
        if self.layer.use_bias:
            weights, biases = tf.keras.backend.batch_get_value(
                (self.layer.kernel, self.layer.bias)
            )
        else:
            weights = tf.keras.backend.get_value(self.layer.kernel)
            biases = None

        # create nengo object to implement activation function and biases
        output = self.add_nengo_obj(
            node_id, activation=self.layer.activation, biases=biases
        )

        # add connection to implement the dense weights
        self.add_connection(node_id, output, transform=weights.T, trainable=True)

        return output


@Converter.register(tf.keras.layers.Flatten)
class ConvertFlatten(LayerConverter):
    """Convert ``tf.keras.layers.Flatten`` to Nengo objects."""

    def convert(self, node_id):
        # noop, same as reshape
        return self.get_input_obj(node_id)


@Converter.register(tf.keras.layers.InputLayer)
class ConvertInput(LayerConverter):
    """Convert ``tf.keras.layers.InputLayer`` to Nengo objects."""

    def convert(self, node_id):
        shape = self.output_shape(node_id)
        if any(x is None for x in shape):
            raise ValueError(
                f"Input shapes must be fully specified; got {shape}. If inputs contain "
                f"`None` in the first axis to indicate a variable number of timesteps, "
                f"set `temporal_model=True` on the `Converter`."
            )
        output = nengo.Node(size_in=np.prod(shape), label=self.layer.name)

        logger.info("Created %s", output)

        return output


@Converter.register(tf.keras.layers.ReLU)
class ConvertReLU(LayerConverter):
    """Convert ``tf.keras.layers.ReLU`` to Nengo objects."""

    unsupported_args = ["max_value", ("threshold", 0)]

    def convert(self, node_id):
        if self.layer.negative_slope == 0:
            activation = tf.nn.relu
        else:
            activation = LeakyReLU(negative_slope=self.layer.negative_slope)

        output = self.add_nengo_obj(node_id, biases=None, activation=activation)

        self.add_connection(node_id, output)

        return output


@Converter.register(tf.keras.layers.LeakyReLU)
class ConvertLeakyReLU(LayerConverter):
    """Convert ``tf.keras.layers.LeakyReLU`` to Nengo objects."""

    def convert(self, node_id):
        output = self.add_nengo_obj(
            node_id, biases=None, activation=LeakyReLU(negative_slope=self.layer.alpha)
        )

        self.add_connection(node_id, output)

        return output


@Converter.register(tf.keras.layers.Reshape)
class ConvertReshape(LayerConverter):
    """Convert ``tf.keras.layers.Reshape`` to Nengo objects."""

    def convert(self, node_id):
        # nengo doesn't pass shape information between objects (everything is just a
        # vector), so we don't actually need to do anything here, we just return
        # the input layer. layers that require shape information can look it up from
        # the input_shape attribute of their layer
        return self.get_input_obj(node_id)


class ConvertUpSampling(LayerConverter):
    """Base class for converting upsampling layers to Nengo objects."""

    def convert(self, node_id, dimensions):
        output = self.add_nengo_obj(node_id)

        channels_last = (
            getattr(self.layer, "data_format", "channels_last") == "channels_last"
        )
        reps = np.atleast_1d(self.layer.size)

        input_shape = self.input_shape(node_id)
        input_shape = input_shape[:-1] if channels_last else input_shape[1:]

        output_shape = self.output_shape(node_id)
        output_shape = output_shape[:-1] if channels_last else output_shape[1:]

        # figure out nearest neighbour (in the source array) for each point in the
        # upsampled array
        input_pts = np.stack(
            np.meshgrid(
                *[
                    np.arange(0, x * reps[i], reps[i], dtype=np.float32)
                    for i, x in enumerate(input_shape)
                ],
                indexing="ij",
            ),
            axis=-1,
        )
        input_pts += (reps - 1) / 2  # shift to centre of upsampled block
        input_pts = np.reshape(input_pts, (-1, dimensions))

        upsampled_pts = np.stack(
            np.meshgrid(
                *[np.arange(x) for x in output_shape],
                indexing="ij",
            ),
            axis=-1,
        )
        upsampled_pts = np.reshape(upsampled_pts, (-1, dimensions))

        dists = np.linalg.norm(
            input_pts[..., None] - upsampled_pts.T[None, ...], axis=1
        )
        nearest = np.argmin(dists, axis=0)

        # duplicate along channel axis
        idxs = np.arange(np.prod(self.input_shape(node_id))).reshape(
            self.input_shape(node_id)
        )
        if channels_last:
            idxs = np.reshape(idxs, (-1, idxs.shape[-1]))
            idxs = idxs[nearest]
        else:
            idxs = np.reshape(idxs, (idxs.shape[0], -1))
            idxs = idxs[:, nearest]

        # connect from pre, using idxs to perform upsampling
        pre = self.get_input_obj(node_id, tensor_idx=0)[np.ravel(idxs)]
        conn = nengo.Connection(
            pre,
            output,
            synapse=self.converter.synapse
            if isinstance(pre, nengo.ensemble.Neurons)
            else None,
        )
        self.set_trainable(conn, False)

        return output


@Converter.register(tf.keras.layers.UpSampling1D)
class ConvertUpSampling1D(ConvertUpSampling):
    """Convert ``tf.keras.layers.UpSampling1D`` to Nengo objects."""

    def convert(self, node_id):
        return super().convert(node_id, dimensions=1)


@Converter.register(tf.keras.layers.UpSampling2D)
class ConvertUpSampling2D(ConvertUpSampling):
    """Convert ``tf.keras.layers.UpSampling2D`` to Nengo objects."""

    unsupported_args = [("interpolation", "nearest")]

    def convert(self, node_id):
        return super().convert(node_id, dimensions=2)


@Converter.register(tf.keras.layers.UpSampling3D)
class ConvertUpSampling3D(ConvertUpSampling):
    """Convert ``tf.keras.layers.UpSampling3D`` to Nengo objects."""

    def convert(self, node_id):
        return super().convert(node_id, dimensions=3)


class ConvertZeroPadding(LayerConverter):
    """Base class for converting zero-padding layers to Nengo objects."""

    def convert(self, node_id, dimensions):
        output = self.add_nengo_obj(node_id)

        # zeropadding1d doesn't doesn't have data_format, assumes channels_last
        channels_first = (
            getattr(self.layer, "data_format", "channels_last") == "channels_first"
        )

        # the strategy here is that we'll create a nengo node of the full padded size,
        # and then connect up the input to the subset of those node elements
        # corresponding to the inner, non-padded elements. so we need to figure out
        # what the indices are that we need to connect to.

        # build slices representing the non-padded elements within the output shape
        slices = []
        if channels_first:
            slices.append(slice(None))
        for i in range(dimensions):
            if dimensions == 1:
                top_pad, bottom_pad = self.layer.padding
            else:
                top_pad, bottom_pad = self.layer.padding[i]
            length = self.output_shape(node_id)[i + channels_first]
            slices.append(slice(top_pad, length - bottom_pad))

        # apply slices to index array to get the list of indices we want to connect to
        idxs = np.arange(output.size_in).reshape(self.output_shape(node_id))
        idxs = np.ravel(idxs[tuple(slices)])

        # connect up the input to the appropriate indices
        self.add_connection(node_id, output[idxs])

        return output


@Converter.register(tf.keras.layers.ZeroPadding1D)
class ConvertZeroPadding1D(ConvertZeroPadding):
    """Convert ``tf.keras.layers.ZeroPadding1D`` to Nengo objects."""

    def convert(self, node_id):
        return super().convert(node_id, dimensions=1)


@Converter.register(tf.keras.layers.ZeroPadding2D)
class ConvertZeroPadding2D(ConvertZeroPadding):
    """Convert ``tf.keras.layers.ZeroPadding2D`` to Nengo objects."""

    def convert(self, node_id):
        return super().convert(node_id, dimensions=2)


@Converter.register(tf.keras.layers.ZeroPadding3D)
class ConvertZeroPadding3D(ConvertZeroPadding):
    """Convert ``tf.keras.layers.ZeroPadding3D`` to Nengo objects."""

    def convert(self, node_id):
        return super().convert(node_id, dimensions=3)


class ConvertKerasSpiking(LayerConverter):
    """Base class for converting KerasSpiking layers."""

    unsupported_args = [
        ("return_state", False),
        ("return_sequences", True),
        ("time_major", False),
    ]

    allow_fallback = False

    def convert(self, node_id):
        if self.layer.dt != 0.001:
            # this is the default keras-spiking dt; warn if it has been changed, as
            # those changes will be overridden by sim.dt. we don't want to error,
            # because this isn't necessarily a problem (they could set sim.dt to
            # match the layer dt).
            # TODO: add some kind of callback to check that sim.dt matches layer.dt?
            warnings.warn(
                f"Ignoring {type(self.layer).__name__}.dt={self.layer.dt:f} parameter; "
                f"dt will be controlled by Simulator.dt"
            )
        if self.layer.stateful:
            warnings.warn(
                f"Ignoring {type(self.layer).__name__}.stateful=True parameter; "
                f"statefulness will be controlled by Simulator"
            )


@Converter.register(compat.SpikingActivation)
class ConvertSpikingActivation(ConvertKerasSpiking):
    """Convert ``keras_spiking.SpikingActivation`` to Nengo objects."""

    unsupported_training_args = [("spiking_aware_training", False)]

    def convert(self, node_id):
        super().convert(node_id)

        # note: not applying swap_activations to these activations, because that's
        # likely to be confusing (e.g. swapping relu to spiking relu, and then wrapping
        # that in regularspiking by accident)
        activation = self.activation_map.get(self.layer.activation, None)

        if activation is None:
            # TODO: allow fallback within SpikingActivation?
            raise TypeError(
                f"SpikingActivation activation type ({self.layer.activation}) does not "
                f"have a native Nengo equivalent"
            )

        initial_state = tf.keras.backend.get_value(
            self.layer.layer.cell.get_initial_state(batch_size=1, dtype="float32")
        )[0]

        activation = nengo.RegularSpiking(
            activation, initial_state={"voltage": initial_state}
        )

        output = self.add_nengo_obj(node_id, biases=None, activation=activation)

        self.add_connection(node_id, output)

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if version.parse(nengo.__version__) <= version.parse("3.0.0"):
            return False, "Converting SpikingActivation layers requires Nengo>=3.1.0"

        return super().convertible(layer, converter)


@Converter.register(compat.Lowpass)
@Converter.register(compat.Alpha)
class ConvertLowpassAlpha(ConvertKerasSpiking):
    """Convert ``keras_spiking.Lowpass`` to Nengo objects."""

    def convert(self, node_id):
        super().convert(node_id)

        output = self.add_nengo_obj(node_id)

        tau = np.ravel(tf.keras.backend.get_value(self.layer.layer.cell.tau_var))[0]

        synapse = (
            nengo.Lowpass(tau)
            if isinstance(self.layer, compat.Lowpass)
            else nengo.Alpha(tau)
        )

        self.add_connection(node_id, output, synapse=synapse)

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if not converter.inference_only and layer.trainable:
            msg = (
                f"Cannot convert a {type(layer).__name__} layer to native Nengo "
                f"objects unless inference_only=True or layer.trainable=False"
            )
            return False, msg

        if not np.allclose(
            tf.keras.backend.get_value(layer.layer.cell.initial_level), 0
        ):
            msg = (
                f"Cannot convert a {type(layer).__name__} layer to native Nengo "
                f"objects with initial_level != 0 (this probably means that training "
                f"has been applied to the layer before conversion)"
            )
            return False, msg

        tau = tf.keras.backend.get_value(layer.layer.cell.tau_var)
        if not np.allclose(tau, np.ravel(tau)[0]):
            msg = (
                f"Cannot convert a {type(layer).__name__} layer to native Nengo "
                f"objects with different tau values for each element (this probably "
                f"means that training has been applied to the layer before conversion)"
            )
            return False, msg

        return super().convertible(layer, converter)


@Converter.register(tf.keras.layers.TimeDistributed)
class ConvertTimeDistributed(LayerConverter):
    """Convert ``tf.keras.layers.TimeDistributed`` to Keras objects."""

    allow_fallback = False

    def convert(self, node_id):
        # nengo models are already temporal, so we don't need to do anything special
        # to make the wrapped layer temporal, we just convert the wrapped layer

        # for some reason, using a layer inside a TimeDistributed wrapper doesn't
        # update it's inbound_nodes. so we'll call the wrapped layer on the wrapper's
        # input. note that the wrapper's input has the extra time dimension, so we slice
        # out just the first timestep (it doesn't matter what the values are)
        input_node = self.layer.inbound_nodes[node_id]
        sliced_input = input_node.input_tensors[:, 0]
        _ = self.layer.layer(sliced_input)

        # add the sliced input we created above to the layer map so that when
        # we look up the input nengo objects of the wrapped layer we get the input of
        # the timedistributed layer
        (
            input_layer,
            input_node_id,
            _,
        ) = input_node.input_tensors._keras_history
        (
            sliced_input_layer,
            sliced_input_node_id,
            _,
        ) = sliced_input._keras_history
        layer_map = nengo.Network.context[-1].layer_map
        layer_map[sliced_input_layer][sliced_input_node_id] = layer_map[input_layer][
            input_node_id
        ]

        # temporarily set temporal_model=False so that the wrapped layer builds
        # correctly
        self.converter.temporal_model = False
        try:
            result = self.converter.get_converter(self.layer.layer).convert(
                # use the nodeid of the just created node
                len(self.layer.layer.inbound_nodes)
                - 1
            )
        finally:
            self.converter.temporal_model = True

        return result

    @classmethod
    def convertible(cls, layer, converter):
        if not converter.temporal_model:
            return (
                False,
                "TimeDistributed layers can only be converted when temporal_model=True",
            )

        return super().convertible(layer, converter)
