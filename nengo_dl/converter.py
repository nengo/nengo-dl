"""Tools for automatically converting a Keras model to a Nengo network."""

import collections
import logging
import warnings

import nengo
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization, BatchNormalizationV2
from tensorflow.python.util import nest

from nengo_dl.config import configure_settings
from nengo_dl.tensor_node import Layer, TensorNode
from nengo_dl.simulator import Simulator

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

    Attributes
    ----------
    model : ``tf.keras.Model``
        The input Keras model (if input was a Sequential model then this will be the
        equivalent Functional model).
    net : `nengo.Network`
        The converted Nengo network.
    inputs : `.Converter.TensorDict`
        Maps from Keras model inputs to input Nodes in the converted Nengo network.
        For example, ``my_node = Converter(my_model).inputs[my_model.input]``.
    outputs : `.Converter.TensorDict`
        Maps from Keras model outputs to output Probes in the converted Nengo network.
        For example, ``my_probe = Converter(my_model).outputs[my_model.output]``.
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
    ):
        self.allow_fallback = allow_fallback
        self.inference_only = inference_only
        self.max_to_avg_pool = max_to_avg_pool
        self.split_shared_weights = split_shared_weights
        self.swap_activations = swap_activations or {}
        self.layer_map = collections.defaultdict(dict)
        self._layer_converters = {}

        with nengo.Network(label=model.name) as self.net:
            # add the "trainable" attribute to all objects
            configure_settings(trainable=None, inference_only=self.inference_only)

            # convert model
            self.get_converter(model).convert(None)

            if isinstance(model, tf.keras.Sequential):
                # if someone passes a sequential model we convert it to a functional
                # model and then convert that to a nengo model, so make the functional
                # model accessible here
                warnings.warn("Converting sequential model to functional model")
                self.model = self.get_converter(model).layer
            else:
                self.model = model

            # track inputs/outputs of model on network object
            self.inputs = Converter.TensorDict()
            for input in self.model.inputs:
                (
                    input_layer,
                    input_node_id,
                    input_tensor_id,
                ) = LayerConverter.get_history(input)
                self.inputs[input] = self.layer_map[input_layer][input_node_id][
                    input_tensor_id
                ]

            self.outputs = Converter.TensorDict()
            for output in self.model.outputs:
                (
                    output_layer,
                    output_node_id,
                    output_tensor_id,
                ) = LayerConverter.get_history(output)
                output_obj = self.layer_map[output_layer][output_node_id][
                    output_tensor_id
                ]

                # add probes to outputs
                logger.info("Probing %s (%s)", output_obj, output)
                self.outputs[output] = nengo.Probe(output_obj)

    def verify(self, training=False, inputs=None):
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

        if inputs is None:
            batch_size = 2
            inp_vals = [np.ones((batch_size,) + x.shape[1:]) for x in self.model.inputs]
        else:
            batch_size = inputs[0].shape[0]
            inp_vals = inputs

        # get keras model output
        if training:
            out_vals = [
                np.ones((batch_size,) + x.shape[1:]) for x in self.model.outputs
            ]
            self.model.compile(optimizer=tf.optimizers.SGD(0.1), loss=tf.losses.mse)
            self.model.fit(inp_vals, out_vals, epochs=epochs)

        keras_out = self.model.predict(inp_vals)

        if not isinstance(keras_out, (list, tuple)):
            keras_out = [keras_out]

        # get nengo sim output
        inp_vals = [np.reshape(x, (batch_size, 1, -1)) for x in inp_vals]

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
                        "Number of trainable parameters in Nengo network (%d) does not "
                        "match number of trainable parameters in Keras model (%d)"
                        % (nengo_params, keras_params)
                    )

                out_vals = [np.reshape(x, (batch_size, 1, -1)) for x in out_vals]
                sim.compile(optimizer=tf.optimizers.SGD(0.1), loss=tf.losses.mse)
                sim.fit(inp_vals, out_vals, epochs=epochs)

            sim_out = sim.predict(inp_vals)

        for i, out in enumerate(self.model.outputs):
            keras_vals = np.ravel(keras_out[i])
            nengo_vals = np.ravel(sim_out[self.outputs[out]])
            if not np.allclose(keras_vals, nengo_vals):
                logger.info("Verification failure")
                logger.info("Keras:\n%s", keras_vals)
                logger.info("Nengo:\n%s", nengo_vals)
                raise ValueError(
                    "Output of Keras model does not match output of converted "
                    "Nengo network"
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
                    "Multiple applications of layer %s detected; this is not supported "
                    "unless split_shared_weights=True" % layer
                )
            return converter

        # check if there is a registered builder for this layer type
        ConverterClass = self.converters.get(type(layer), None)

        # perform custom checks in layer converters
        if ConverterClass is None:
            error_msg = "Layer type %s does not have a registered converter" % type(
                layer
            )
        else:
            convertible, error_msg = ConverterClass.convertible(layer, self)
            if not convertible:
                ConverterClass = None

        if ConverterClass is None:
            # this means that there is no LayerConverter compatible with this layer
            # (either because it has an unknown type, or it failed the ``.convertible``
            # check due to its internal parameterization)
            if self.allow_fallback:
                warnings.warn(
                    "%sFalling back to TensorNode."
                    % (error_msg + ". " if error_msg else "")
                )
                ConverterClass = self.converters[None]
            else:
                raise TypeError(
                    "%sUnable to convert layer %s to native Nengo objects; set "
                    "allow_fallback=True if you would like to use a TensorNode "
                    "instead, or consider registering a custom LayerConverter for this "
                    "layer type." % (error_msg + ". " if error_msg else "", layer.name)
                )

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
                    "Layer '%s' already has a converter. Overwriting." % keras_layer
                )

            cls.converters[keras_layer] = convert_cls

            return convert_cls

        return register_converter

    class TensorDict:
        """A dictionary-like object that works with TensorFlow Tensors."""

        def __init__(self):
            self.dict = collections.OrderedDict()

        def __setitem__(self, key, val):
            if isinstance(key, tf.Tensor):
                key = key.experimental_ref()

            self.dict[key] = val

        def __getitem__(self, key):
            if isinstance(key, tf.Tensor):
                key = key.experimental_ref()

            return self.dict[key]


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

    # attributes of the Keras layer that are not supported for non-default values.
    # the default value is assumed to be None, or a tuple of
    # ("attr_name", default_value) can be specified
    unsupported_args = []
    # attributes that are supported in inference_only mode but otherwise not
    unsupported_training_args = []

    # whether or not this layer contains trainable weights (this indicates whether
    # this layer is affected by split_shared_weights)
    has_weights = False

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
        name = self.layer.name + ".%d" % node_id

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
                    size_in=np.prod(self.output_shape(node_id)), label=name,
                )
                if biases is not None:
                    # use a connection from a constant node (so that the bias
                    # values will be trainable)
                    bias_node = nengo.Node([1], label="%s.bias_node" % name)
                    nengo.Connection(
                        bias_node, obj, transform=biases[:, None], synapse=None
                    )
            else:
                # use ensemble to implement the appropriate neuron type
                obj = nengo.Ensemble(
                    np.prod(self.output_shape(node_id)),
                    1,
                    neuron_type=activation,
                    gain=nengo.dists.Choice([1]),
                    bias=nengo.dists.Choice([0]) if biases is None else biases,
                    label=name,
                ).neurons
                if biases is None:
                    # ensembles always have biases, so if biases=None we just use
                    # all-zero biases and mark them as non-trainable
                    self.converter.net.config[obj].trainable = False
        elif self.converter.allow_fallback:
            warnings.warn(
                "Activation type %s does not have a native Nengo equivalent; "
                "falling back to a TensorNode" % activation
            )
            obj = TensorNode(
                activation,
                shape_in=self.input_shape(node_id),
                pass_time=False,
                label=name,
            )
        else:
            raise TypeError("Unsupported activation type (%s)" % self.layer.activation)

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
        conn = nengo.Connection(
            self.get_input_obj(node_id, tensor_idx=input_idx),
            obj,
            synapse=None,
            **kwargs,
        )
        self.converter.net.config[conn].trainable = trainable

        logger.info(
            "Connected %s to %s (trainable=%s)", conn.pre, conn.post, trainable,
        )

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

        input_layer, input_node_id, input_tensor_id = self.get_history(tensor)

        if input_node_id in self.converter.layer_map[input_layer]:
            return self.converter.layer_map[input_layer][input_node_id][input_tensor_id]
        else:
            return None

    def _get_shape(self, input_output, node_id, include_batch=False):
        """
        Looks up the input or output shape of this Node.

        Parameters
        ----------
        input_output : "input" or "output"
            Whether we want the input or output shape.
        node_id : int
            The node whose shape we want to look up.
        include_batch : bool
            Whether or not the returned shape should include the batch dimension.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one input/output, or a list of shapes
            if the node as multiple inputs/outputs.
        """

        # note: layer.get_input/output_shape_at is generally equivalent to
        # layer.input/output_shape, except when the layer is called multiple times
        # with different shapes, in which case input/output_shape is not well defined
        func = getattr(self.layer, "get_%s_shape_at" % input_output)

        # get the shape
        shape = func(node_id)

        if not include_batch:
            if isinstance(shape, list):
                # multiple inputs/outputs; trim the batch from each one
                shape = [s[1:] for s in shape]
            else:
                shape = shape[1:]

        return shape

    def input_shape(self, node_id, include_batch=False):
        """
        Returns the input shape of the given node.

        Parameters
        ----------
        node_id : int
            The node whose shape we want to look up.
        include_batch : bool
            Whether or not the returned shape should include the batch dimension.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one input, or a list of shapes
            if the node as multiple inputs.
        """
        return self._get_shape("input", node_id, include_batch=include_batch)

    def output_shape(self, node_id, include_batch=False):
        """
        Returns the output shape of the given node.

        Parameters
        ----------
        node_id : int
            The node whose shape we want to look up.
        include_batch : bool
            Whether or not the returned shape should include the batch dimension.

        Returns
        -------
        shape : (list of) tuple of int
            A single tuple shape if the node has one output, or a list of shapes
            if the node as multiple outputs.
        """
        return self._get_shape("output", node_id, include_batch=include_batch)

    @staticmethod
    def get_history(tensor):
        """
        Returns the Keras history (layer/node_idx/tensor_idx) that defined this tensor.

        This function contains additional logic so that if ``tensor`` is the output of
        a Model then the history will trace into the internal layers of that Model
        (rather than skipping to the input of that Model, which is the default Keras
        history).

        Parameters
        ----------
        tensor : ``tf.Tensor``
            The tensor whose Keras history we want to look up.

        Returns
        -------
        layer : ``tf.keras.layers.Layer``
            The Layer object that created this Tensor.
        node_index : int
            The index of the outbound node of ``layer`` that created this Tensor.
        tensor_index : int
            The index in the output of the Node corresponding to this Tensor (for
            Nodes with multiple outputs).
        """
        layer, node_index, tensor_index = tensor._keras_history

        while isinstance(layer, tf.keras.Model):
            # models have an output Identity transform that stores the history that
            # "skips" the internals of the model; we want to traverse into the internals
            # of the model, so we go back to the input of that identity op (which
            # is the real output tensor from the model)
            assert tensor.op.type == "Identity"
            tensor = tensor.op.inputs[0]
            layer, node_index, tensor_index = tensor._keras_history

        return layer, node_index, tensor_index

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
                msg = "%s.%s has value %s != %s, which is not supported" % (
                    layer.name,
                    arg,
                    val,
                    default,
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


@Converter.register(tf.keras.Model)
class ConvertModel(LayerConverter):
    """Convert ``tf.keras.Model`` to Nengo objects."""

    def convert(self, node_id):
        # should never be building a model except in the top-level converter
        assert node_id is None

        logger.info("=" * 30)
        logger.info("Converting model %s", self.layer.name)

        # functional models should already have been built when the model
        # was instantiated
        assert self.layer.built

        # trace the model to find all the tensors (which correspond to layers/nodes)
        # that need to be built into the Nengo network
        source_tensors = self.trace_tensors(self.layer.outputs)

        def sort_key(x):
            # sort tensors so that order of model inputs/outputs is preserved
            for i, y in enumerate(self.layer.inputs):
                if x is y:
                    return -(len(self.layer.inputs) - i)
            for i, y in enumerate(self.layer.outputs):
                if x is y:
                    return i + 1
            return 0

        source_tensors = sorted(source_tensors, key=sort_key)

        for tensor in source_tensors:
            # look up the layer/node to be converted
            model_layer, model_node_id, _ = self.get_history(tensor)
            if model_node_id in self.converter.layer_map[model_layer]:
                # already built this node
                continue

            logger.info("-" * 30)
            logger.info("Converting layer %s node %d", model_layer.name, model_node_id)

            # get the layerconverter object
            layer_converter = self.converter.get_converter(model_layer)

            # build the Nengo objects
            nengo_layer = layer_converter.convert(model_node_id)
            assert isinstance(
                nengo_layer, (nengo.Node, nengo.ensemble.Neurons, TensorNode),
            )

            # add output of layer_converter to layer_map
            self.converter.layer_map[model_layer][model_node_id] = [nengo_layer]

        logger.info("=" * 30)

        # note: not returning anything, because we don't need to store anything in
        # the layer map (this network is only used by the top-level converter class)

    def trace_tensors(self, tensors, results=None):
        """
        Recursively trace all the upstream layer tensors, starting from ``tensors``.

        Parameters
        ----------
        tensors : list of ``tf.Tensor``
            Tensors representing the output of some layers.
        results : list of ``tf.Tensor``
            Output tensors for all the layers leading up to and including ``tensors``.
            This will be populated in-place during the recursive execution.

        Returns
        -------
        results : list of ``tf.Tensor``
            The same as the ``results`` parameter (returned so that the top-level call,
            which may not have a reference to the ``results`` list can get the results).
        """
        # brief intro to the keras functional graph structure:
        # - a node represents the application of some layer to an input tensor
        # - whenever a layer B is applied to the output of layer A a new Node is
        #   created; this Node is added to A.outbound_nodes and B.inbound_nodes
        # - every Node has input tensors x (which will be the input to B) and output
        #   tensors y (the output of B)
        # - every tensor tracks the layer/node that created it in _keras_history
        #   (so the _keras_history of y would be (B, 0) (where 0 is the index within
        #   B.inbound_nodes corresponding to the node that created y); note that x was
        #   created whenever A was applied to some other layer, so its _keras_history is
        #   unrelated to the application of B

        # for example, if we apply multiple layers B/C/D to the output of some
        # layer A:
        #  b = B(a)
        #  c = C(a)
        #  d = D(a)
        # each time will create a new Node. so we will have 3 nodes total;
        # A will have 3 outbound nodes, and B/C/D will each have one inbound node.
        # every node will have the same input tensor (a), but a different
        # output tensor (the output of B/C/D) with keras_history (B, 0), (C, 0), and
        # (D, 0).

        # on the other hand, if we take one layer and apply it to multiple inputs:
        #  d0 = D(a)
        #  d1 = D(b)
        #  d2 = D(c)
        # again we will have 3 nodes total. D will 3 inbound nodes, and A/B/C will each
        # have one outbound node. each node will have a different input tensor (the
        # output of A/B/C), but _also a different output tensor_ (the result of applying
        # D to each one of those inputs) with keras history (D, 0), (D, 1), (D, 2)

        if results is None:
            results = []

        logger.debug("===starting trace_tensors===")
        logger.debug("Tracing tensors %s", tensors)

        for tensor in tensors:
            if any(tensor is y for y in results):
                # already traced this tensor
                continue

            layer, node_index, _ = self.get_history(tensor)

            logger.debug("---")
            logger.debug("Layer %s node %s", layer.name, node_index)

            if layer.inbound_nodes:
                node = layer.inbound_nodes[node_index]
                if node.inbound_layers:
                    logger.debug("Input layers %s", node.inbound_layers)
                    logger.debug("Input tensors %s", node.input_tensors)

                    # not an input layer, so continue recursion
                    self.trace_tensors(
                        nest.flatten(node.input_tensors), results=results
                    )

            results.append(tensor)

        logger.debug("===done trace_tensors===")

        return results


@Converter.register(tf.keras.Sequential)
class ConvertSequential(ConvertModel):
    """Convert ``tf.keras.Sequential`` to Nengo objects."""

    def __init__(self, seq_model, converter):
        # convert sequential model to functional model
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
            layer.build(self.input_shape(0, include_batch=True))
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
            [np.eye(n_filters) / n_pool] * n_pool, pool_size + (n_filters, n_filters),
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


@Converter.register(BatchNormalization)
@Converter.register(BatchNormalizationV2)
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
        bias = beta - gamma * mean / stddev

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
            broadcast_scale[slices] = scale[i]
            broadcast_bias[slices] = bias[i]
        broadcast_scale = np.ravel(broadcast_scale)
        broadcast_bias = np.ravel(broadcast_bias)

        # connect up bias node to output
        bias_node = nengo.Node(broadcast_bias)
        conn = nengo.Connection(bias_node, output, synapse=None)
        self.converter.net.config[conn].trainable = False

        # connect input to output, scaled by the batch normalization scale
        conn = nengo.Connection(
            self.get_input_obj(node_id),
            output,
            synapse=None,
            transform=broadcast_scale,
        )
        self.converter.net.config[conn].trainable = False

        # this is an alternate approach, where rather than broadcasting scale/bias,
        # we create individual connections for each element in the batch normalization
        # axis. this will result in smaller weight matrices, but more Connections
        # TODO: figure out where the tradeoffs lie between these two approaches
        # bias_node = nengo.Node(np.ones(idxs[slices].size))
        #
        # # for each element in the batch normalization axis
        # for i in range(idxs.shape[axis]):
        #     # slice out one element of the output along the axis
        #     slices[axis] = i
        #     slice_idxs = np.ravel(idxs[slices])
        #     sliced_output = output[slice_idxs]
        #
        #     # connect up bias
        #     conn = nengo.Connection(
        #         bias_node, sliced_output, synapse=None, transform=bias[i],
        #     )
        #     self.converter.net.config[conn].trainable = False
        #
        #     # connect up input with scale applied
        #     conn = nengo.Connection(
        #         self.get_input_obj(node_id)[slice_idxs],
        #         sliced_output,
        #         synapse=None,
        #         transform=scale[i],
        #     )
        #     self.converter.net.config[conn].trainable = False

        return output

    @classmethod
    def convertible(cls, layer, converter):
        if not converter.inference_only:
            msg = (
                "Cannot convert BatchNormalization layer to native Nengo objects "
                "unless inference_only=True"
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
            if self.layer.data_format == "channels_first":
                spatial_size = np.prod(self.output_shape(node_id)[1:])
                bias_node = nengo.Node(np.ones(spatial_size), label="conv_bias")
                offset = 0
                for i in range(self.output_shape(node_id)[0]):
                    nengo.Connection(
                        bias_node,
                        output[offset : offset + spatial_size],
                        transform=biases[i],
                        synapse=None,
                    )
                    offset += spatial_size
            else:
                spatial_size = np.prod(self.output_shape(node_id)[:-1])
                bias_node = nengo.Node(np.ones(spatial_size), label="conv_bias")
                idxs = np.arange(np.prod(self.output_shape(node_id))).reshape(
                    (-1, self.output_shape(node_id)[-1])
                )
                for i in range(self.output_shape(node_id)[-1]):
                    nengo.Connection(
                        bias_node,
                        output[idxs[:, i]],
                        transform=biases[i],
                        synapse=None,
                    )

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


@Converter.register(tf.keras.layers.Conv1D,)
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


@Converter.register(tf.keras.layers.Conv2D,)
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


@Converter.register(tf.keras.layers.Conv3D,)
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
        # if this input layer has an input obj, that means it is a passthrough
        # (so we just return the input)
        output = self.get_input_obj(node_id)

        if output is None:
            # not a passthrough input, so create input node
            shape = self.output_shape(node_id)
            if any(x is None for x in shape):
                raise ValueError(
                    "Input shapes must be fully specified; got %s" % (shape,)
                )
            output = nengo.Node(np.zeros(np.prod(shape)), label=self.layer.name)

            logger.info("Created %s", output)

        return output


@Converter.register(tf.keras.layers.ReLU)
class ConvertReLU(LayerConverter):
    """Convert ``tf.keras.layers.ReLU`` to Nengo objects."""

    unsupported_args = [("negative_slope", 0), "max_value", ("threshold", 0)]

    def convert(self, node_id):
        output = self.add_nengo_obj(node_id, biases=None, activation=tf.nn.relu)

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
