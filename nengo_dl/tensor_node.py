"""
TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.

See `the documentation <https://www.nengo.ai/nengo-dl/tensor-node.html>`_ for more
details.
"""

import contextlib
import warnings

import numpy as np
import tensorflow as tf
from nengo import Connection, Ensemble, Node, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.config import Config
from nengo.exceptions import SimulationError, ValidationError
from nengo.neurons import NeuronType
from nengo.params import BoolParam, Default, Parameter, ShapeParam
from tensorflow.python.eager import context

from nengo_dl.builder import Builder, NengoBuilder, OpBuilder
from nengo_dl.compat import default_transform, eager_enabled
from nengo_dl.config import configure_settings


def validate_output(output, minibatch_size=None, output_d=None, dtype=None):
    """
    Performs validation on the output of a TensorNode ``tensor_func``.

    Parameters
    ----------
    output : ``tf.Tensor`` or ``tf.TensorSpec``
        Output from the ``tensor_func``.
    minibatch_size : int
        Expected minibatch size for the simulation.
    output_d
        Expected output dimensionality for the function.
    dtype
        Expected dtype of the function output.
    """

    if not isinstance(output, tf.TensorSpec) and not tf.is_tensor(output):
        raise ValidationError(
            f"TensorNode function must return a Tensor (got {type(output)})",
            attr="tensor_func",
        )

    if minibatch_size is not None and output.shape[0] != minibatch_size:
        raise ValidationError(
            f"TensorNode output should have batch size {minibatch_size} (got "
            f"{output.shape[0]})",
            attr="tensor_func",
        )

    if output_d is not None and np.prod(output.shape[1:]) != output_d:
        raise ValidationError(
            f"TensorNode output should have size {output_d} (got shape "
            f"{output.shape[1:]} with size {np.prod(output.shape[1:])})",
            attr="tensor_func",
        )

    if dtype is not None and output.dtype != dtype:
        raise ValidationError(
            f"TensorNode output should have dtype {dtype} (got {output.dtype})",
            attr="tensor_func",
        )


class TensorFuncParam(Parameter):
    """Parameter for the ``tensor_func`` parameter of a `.TensorNode`."""

    def __init__(self, name, readonly=False):
        super().__init__(name, optional=False, readonly=readonly)

    def coerce(self, instance, value):
        """
        Performs validation on the function passed to TensorNode, and sets
        ``shape_out`` if necessary.

        Parameters
        ----------
        instance : `.TensorNode`
            The node whose ``tensor_func`` parameter is being set.
        value : callable
            The function being assigned to the TensorNode.

        Returns
        -------
        output : callable
            The function after validation is applied.
        """

        output = super().coerce(instance, value)

        if not callable(value):
            raise ValidationError(
                "TensorNode output must be a function or Keras Layer",
                attr=self.name,
                obj=instance,
            )

        if instance.shape_out is None:
            if isinstance(value, tf.keras.layers.Layer):
                # we can use Keras' static shape inference to get the
                # output shape, which avoids having to build/call the layer
                if instance.pass_time:
                    input_spec = [tf.TensorSpec(())]
                else:
                    input_spec = []
                if instance.shape_in is not None:
                    input_spec += [tf.TensorSpec((1,) + instance.shape_in)]

                if len(input_spec) == 1:
                    input_spec = input_spec[0]

                ctx = contextlib.suppress() if eager_enabled() else context.eager_mode()

                try:
                    with ctx:
                        result = value.compute_output_signature(input_spec)
                except Exception as e:
                    raise ValidationError(
                        "Attempting to automatically determine TensorNode output shape "
                        "by calling Layer.compute_output_signature produced an error. "
                        "If you would like to avoid this step, try manually setting "
                        "`TensorNode(..., shape_out=x)`.",
                        attr=self.name,
                        obj=instance,
                    ) from e

            else:
                if instance.pass_time:
                    args = (tf.constant(0.0),)
                else:
                    args = ()
                if instance.shape_in is not None:
                    args += (tf.zeros((1,) + instance.shape_in),)

                try:
                    result = value(*args)
                except Exception as e:
                    raise ValidationError(
                        "Attempting to automatically determine TensorNode output "
                        "shape by calling TensorNode function produced an error. If "
                        "you would like to avoid this step, try manually setting "
                        "`TensorNode(..., shape_out=x)`.",
                        attr=self.name,
                        obj=instance,
                    ) from e

            validate_output(result)

            instance.shape_out = result.shape[1:]

        return output


class TensorNode(Node):
    """
    Inserts TensorFlow code into a Nengo model.

    Parameters
    ----------
    tensor_func : callable
        A function that maps node inputs to outputs
    shape_in : tuple of int
        Shape of TensorNode input signal (not including batch dimension).
    shape_out : tuple of int
        Shape of TensorNode output signal (not including batch dimension).
        If None, value will be inferred by calling ``tensor_func``.
    pass_time : bool
        If True, pass current simulation time to TensorNode function (in addition
        to the standard input).
    label : str (Default: None)
        A name for the node, used for debugging and visualization
    """

    tensor_func = TensorFuncParam("tensor_func")
    shape_in = ShapeParam("shape_in", default=None, low=1, optional=True)
    shape_out = ShapeParam("shape_out", default=None, low=1, optional=True)
    pass_time = BoolParam("pass_time", default=True)

    def __init__(
        self,
        tensor_func,
        shape_in=Default,
        shape_out=Default,
        pass_time=Default,
        label=Default,
    ):
        # pylint: disable=non-parent-init-called,super-init-not-called
        # note: we bypass the Node constructor, because we don't want to
        # perform validation on `output`
        NengoObject.__init__(self, label=label, seed=None)

        self.shape_in = shape_in
        self.shape_out = shape_out
        self.pass_time = pass_time

        if not (self.shape_in or self.pass_time):
            raise ValidationError(
                "Must specify either shape_in or pass_time", "TensorNode"
            )

        self.tensor_func = tensor_func

    @property
    def output(self):
        """
        Ensures that nothing tries to evaluate the `output` attribute
        (indicating that something is trying to simulate this as a regular
        `nengo.Node` rather than a TensorNode).
        """

        def output_func(*_):
            raise SimulationError(
                "Cannot call TensorNode output function (this probably means "
                "you are trying to use a TensorNode inside a Simulator other "
                "than NengoDL)"
            )

        return output_func

    @property
    def size_in(self):
        """Number of input elements (flattened)."""

        return 0 if self.shape_in is None else np.prod(self.shape_in)

    @property
    def size_out(self):
        """Number of output elements (flattened)."""

        return 0 if self.shape_out is None else np.prod(self.shape_out)


@NengoBuilder.register(TensorNode)
def build_tensor_node(model, node):
    """This is the Nengo build function, so that Nengo knows what to do with
    TensorNodes."""

    # time signal
    if node.pass_time:
        time_in = model.time
    else:
        time_in = None

    # input signal
    if node.shape_in is not None:
        sig_in = builder.Signal(shape=(node.size_in,), name=f"{node}.in")
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    sig_out = builder.Signal(shape=(node.size_out,), name=f"{node}.out")

    model.sig[node]["in"] = sig_in
    model.sig[node]["out"] = sig_out
    model.params[node] = None

    model.operators.append(
        SimTensorNode(node.tensor_func, time_in, sig_in, sig_out, node.shape_in)
    )


class SimTensorNode(builder.Operator):  # pylint: disable=abstract-method
    """Operator for TensorNodes (constructed by `.build_tensor_node`).

    Parameters
    ----------
    func : callable
        The TensorNode function (``tensor_func``).
    time : `~nengo.builder.Signal` or None
        Signal representing the current simulation time (or None if ``pass_time`` is
        False).
    input : `~nengo.builder.Signal` or None
        Input Signal for the TensorNode (or None if no inputs).
    output : `~nengo.builder.Signal`
        Output Signal for the TensorNode.
    shape_in : tuple of int or None
        Shape of input to TensorNode (if None, will leave the shape of input signal
        unchanged).
    tag : str
        A label associated with the operator, for debugging

    Notes
    -----
    1. sets ``[output]``
    2. incs ``[]``
    3. reads ``[time]`` (if ``pass_time=True``) + ``[input]`` (if ``input`` is not None)
    4. updates ``[]``
    """

    def __init__(self, func, time, input, output, shape_in, tag=None):
        super().__init__(tag=tag)

        self.func = func
        self.time = time
        self.input = input
        self.output = output
        self.shape_in = shape_in

        self.sets = [output]
        self.incs = []
        if time is None:
            self.reads = []
        else:
            self.reads = [time]
        if input is not None:
            self.reads += [input]
        self.updates = []


@Builder.register(SimTensorNode)
class SimTensorNodeBuilder(OpBuilder):
    """Builds a `~.tensor_node.SimTensorNode` operator into a NengoDL
    model."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        # SimTensorNodes should never be merged
        assert len(self.ops) == 1
        op = self.ops[0]

        if op.time is None:
            self.time_data = None
        else:
            self.time_data = signals[op.time].reshape(())

        if op.input is None:
            self.src_data = None
        else:
            self.src_data = signals[op.input]
            assert self.src_data.ndim == 1
            if op.shape_in is not None:
                self.src_data = self.src_data.reshape(op.shape_in)

        self.dst_data = signals[op.output]

        self.func = op.func

    def build_step(self, signals):
        if self.time_data is None:
            inputs = []
        else:
            inputs = [signals.gather(self.time_data)]

        if self.src_data is not None:
            inputs += [signals.gather(self.src_data)]

        if isinstance(self.func, tf.keras.layers.Layer):
            if len(inputs) == 1:
                inputs = inputs[0]
            kwargs = (
                dict(training=self.config.training)
                if self.func._expects_training_arg
                else {}
            )
            output = self.func.call(inputs, **kwargs)
        else:
            output = self.func(*inputs)

        validate_output(
            output,
            minibatch_size=signals.minibatch_size,
            output_d=self.dst_data.shape[0],
            dtype=signals.dtype,
        )

        signals.scatter(self.dst_data, output)


class Layer:
    """
    A wrapper for constructing TensorNodes.

    This is designed to mimic and integrate with the ``tf.keras.layers.Layer`` API, e.g.

    .. testcode::

        with nengo.Network():
            a = nengo.Ensemble(10, 1)
            b = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(a)
            c = nengo_dl.Layer(lambda x: x + 1)(b)
            d = nengo_dl.Layer(nengo.LIF())(c)

    Parameters
    ----------
    layer_func : callable or ``tf.keras.Layer`` or `~nengo.neurons.NeuronType`
        A function or Keras Layer that takes the value from an input (represented
        as a ``tf.Tensor``) and maps it to some output value, or a Nengo neuron type
        (which will be instantiated in a Nengo Ensemble and applied to the input).
    """

    def __init__(self, layer_func):
        self.layer_func = layer_func

    def __call__(
        self,
        input,
        transform=default_transform,
        shape_in=None,
        synapse=None,
        return_conn=False,
        **layer_args,
    ):
        """
        Apply the TensorNode layer to the given input object.

        Parameters
        ----------
        input : ``NengoObject``
            Object providing input to the layer.
        transform : `~numpy.ndarray`
            Transform matrix to apply on connection from ``input`` to this layer.
        shape_in : tuple of int
            If not None, reshape the input to the given shape.
        synapse : float or `~nengo.synapses.Synapse`
            Synapse to apply on connection from ``input`` to this layer.
        return_conn : bool
            If True, also return the connection linking this layer to ``input``.
        layer_args : dict
            These arguments will be passed to `.TensorNode` if ``layer_func`` is a
            callable or Keras Layer, or `~nengo.Ensemble` if ``layer_func`` is a
            `~nengo.neurons.NeuronType`.

        Returns
        -------
        obj : `.TensorNode` or `~nengo.ensemble.Neurons`
            A TensorNode that implements the given layer function (if
            ``layer_func`` was a callable/Keras layer), or a Neuron object with the
            given neuron type, connected to ``input``.
        conn : `~nengo.Connection`
            If ``return_conn`` is True, also returns the connection object linking
            ``input`` and ``obj``.

        Notes
        -----
        The input connection created for the new TensorNode will be marked as
        non-trainable by default.
        """

        if shape_in is not None and all(x is not None for x in shape_in):
            size_in = np.prod(shape_in)
        elif isinstance(transform, np.ndarray) and transform.ndim == 2:
            size_in = transform.shape[0]
        else:
            size_in = input.size_out

        if isinstance(self.layer_func, NeuronType):
            obj = Ensemble(
                size_in, 1, neuron_type=self.layer_func, **layer_args
            ).neurons
        else:
            obj = TensorNode(
                self.layer_func,
                shape_in=(size_in,) if shape_in is None else shape_in,
                pass_time=False,
                **layer_args,
            )

        conn = Connection(input, obj, synapse=synapse, transform=transform)

        # set connection to non-trainable
        cfg = Config.context[0][conn]
        if not hasattr(cfg, "trainable"):
            configure_settings(trainable=None)
        cfg.trainable = False

        return (obj, conn) if return_conn else obj

    def __str__(self):
        name = getattr(
            self.layer_func,
            "name",
            getattr(self.layer_func, "__name__", self.layer_func),
        )
        return f"Layer({name})"


def tensor_layer(input, layer_func, **kwargs):
    """Deprecated, use `.Layer` instead."""

    warnings.warn(
        "nengo_dl.tensor_layer is deprecated; use nengo_dl.Layer instead",
        DeprecationWarning,
    )
    return Layer(layer_func)(input, **kwargs)
