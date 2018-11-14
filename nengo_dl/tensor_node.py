"""
TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.
"""

from nengo import Node, Connection, Ensemble, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError, SimulationError
from nengo.neurons import NeuronType
from nengo.params import Default, IntParam, Parameter
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


def validate_output(output, minibatch_size=None, output_d=None, dtype=None):
    """
    Performs validation on the output of a TensorNode ``tensor_func``.

    Parameters
    ----------
    output : ``tf.Tensor``
        Output from the ``tensor_func``.
    minibatch_size : int
        Expected minibatch size for the simulation.
    output_d
        Expected output dimensionality for the function.
    dtype
        Expected dtype of the function output.
    """

    if not isinstance(output, tf.Tensor):
        raise ValidationError("TensorNode function must return a Tensor "
                              "(got %s)" % type(output), attr="tensor_func")

    shape = output.get_shape()
    if (shape.ndims != 2 or
            (minibatch_size is not None and shape[0] != minibatch_size) or
            (output_d is not None and shape[1] != output_d)):
        raise ValidationError("TensorNode output should have shape (%s, %s) "
                              "(got shape %s)" % (minibatch_size, output_d,
                                                  output.get_shape()),
                              attr="tensor_func")

    if dtype is not None and output.dtype != dtype:
        raise ValidationError("TensorNode output should have dtype %s "
                              "(got %s)" % (dtype, output.dtype),
                              attr="tensor_func")


class TensorFuncParam(Parameter):
    """Performs validation on the function passed to TensorNode, and sets
    ``size_out`` if necessary."""

    def __init__(self, name, readonly=False):
        super(TensorFuncParam, self).__init__(
            name, optional=False, readonly=readonly)

    def coerce(self, node, func):
        output = super(TensorFuncParam, self).coerce(node, func)

        if node.size_out is None:
            if not callable(func):
                raise ValidationError("TensorNode output must be a function",
                                      attr=self.name, obj=node)

            with tf.Graph().as_default():
                t, x = tf.constant(0.0), tf.zeros((1, node.size_in))
                args = (t, x) if node.size_in > 0 else (t,)
                try:
                    result = func(*args)
                except Exception as e:
                    raise ValidationError(
                        "Calling TensorNode function with arguments %s "
                        "produced an error:\n%s" % (args, e),
                        attr=self.name, obj=node)

            validate_output(result)

            node.size_out = result.get_shape()[1].value

        return output


class TensorNode(Node):
    """
    Inserts TensorFlow code into a Nengo model.

    Parameters
    ----------
    tensor_func : callable
        A function that maps node inputs to outputs
    size_in : int (Default: 0)
        The number of elements in the input vector
    size_out : int (Default: None)
        The number of elements in the output vector (if None, value will be
        inferred by calling ``tensor_func``)
    label : str (Default: None)
        A name for the node, used for debugging and visualization
    """

    tensor_func = TensorFuncParam('tensor_func')
    size_in = IntParam('size_in', default=0, low=0, optional=True)
    size_out = IntParam('size_out', default=None, low=1, optional=True)

    def __init__(self, tensor_func, size_in=Default, size_out=Default,
                 label=Default):
        # pylint: disable=non-parent-init-called,super-init-not-called
        # note: we bypass the Node constructor, because we don't want to
        # perform validation on `output`
        NengoObject.__init__(self, label=label, seed=None)

        self.size_in = size_in
        self.size_out = size_out
        self.tensor_func = tensor_func

    @property
    def output(self):
        """
        Ensure that nothing tries to evaluate the `output` attribute
        (indicating that something is trying to simulate this as a regular
        `nengo.Node` rather than a TensorNode.
        """
        def output_func(*_):
            raise SimulationError(
                "Cannot call TensorNode output function (this probably means "
                "you are trying to use a TensorNode inside a Simulator other "
                "than NengoDL)")

        return output_func


@NengoBuilder.register(TensorNode)
def build_tensor_node(model, node):
    """This is the Nengo build function, so that Nengo knows what to do with
    TensorNodes."""

    # input signal
    if node.size_in > 0:
        sig_in = builder.Signal(np.zeros(node.size_in), name="%s.in" % node)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    sig_out = builder.Signal(np.zeros(node.size_out), name="%s.out" % node)

    model.sig[node]['in'] = sig_in
    model.sig[node]['out'] = sig_out
    model.params[node] = None

    model.operators.append(SimTensorNode(node.tensor_func, model.time, sig_in,
                                         sig_out))


class SimTensorNode(builder.Operator):  # pylint: disable=abstract-method
    """Operator for TensorNodes (constructed by `.build_tensor_node`).

    Parameters
    ----------
    func : callable
        The TensorNode function (``tensor_func``)
    time : `~nengo.builder.Signal`
        Signal representing the current simulation time
    input : `~nengo.builder.Signal` or None
        Input Signal for the TensorNode (or None if size_in==0)
    output : `~nengo.builder.Signal`
        Output Signal for the TensorNode
    tag : str
        A label associated with the operator, for debugging

    Notes
    -----
    1. sets ``[output]``
    2. incs ``[]``
    3. reads ``[time] if input is None else [time, input]``
    4. updates ``[]``
    """

    def __init__(self, func, time, input, output, tag=None):
        super(SimTensorNode, self).__init__(tag=tag)

        self.func = func
        self.input = input
        self.output = output

        self.sets = [output]
        self.incs = []
        self.reads = [time] if input is None else [time, input]
        self.updates = []


@Builder.register(SimTensorNode)
class SimTensorNodeBuilder(OpBuilder):
    """Builds a `~.tensor_node.SimTensorNode` operator into a NengoDL
    model."""

    def __init__(self, ops, signals, config):
        super(SimTensorNodeBuilder, self).__init__(ops, signals, config)

        # SimTensorNodes should never be merged
        assert len(ops) == 1
        op = ops[0]

        if op.input is None:
            self.src_data = None
        else:
            self.src_data = signals[op.input]
            assert self.src_data.ndim == 1

        self.dst_data = signals[op.output]

        self.func = op.func

        if hasattr(self.func, "pre_build"):
            self.func.pre_build(
                None if self.src_data is None else ((signals.minibatch_size,) +
                                                    self.src_data.shape),
                (signals.minibatch_size,) + self.dst_data.shape)

    def build_step(self, signals):
        if self.src_data is None:
            output = self.func(signals.time)
        else:
            input = signals.gather(self.src_data)

            # move minibatch dimension to front
            input = tf.transpose(input, (1, 0))

            output = self.func(signals.time, input)

        validate_output(output, minibatch_size=signals.minibatch_size,
                        output_d=self.dst_data.shape[0], dtype=signals.dtype)

        # move minibatch dimension back to end
        output = tf.transpose(output, (1, 0))

        signals.scatter(self.dst_data, output)

    def build_post(self, ops, signals, sess, rng):
        if hasattr(self.func, "post_build"):
            self.func.post_build(sess, rng)


def reshaped(shape_in):
    """A decorator to reshape the inputs to a function into non-vector shapes.

    The output of the function will be flatten back into (batched) vectors.

    Parameters
    ----------
    shape_in : tuple of int
        The desired shape for inputs to the function (not including the first
        dimension, which corresponds to the batch axis)

    Returns
    -------
    reshaper : callable
        The decorated function
    """

    def reshape_dec(func):
        def reshaped_func(t, x):
            batch_size = x.get_shape()[0].value
            x = tf.reshape(x, (batch_size,) + shape_in)
            x = func(t, x)
            x = tf.reshape(x, (batch_size, -1))
            return x

        return reshaped_func

    return reshape_dec


def tensor_layer(input, layer_func, shape_in=None, synapse=None,
                 transform=1, return_conn=False, **layer_args):
    """A utility function to construct TensorNodes that apply some function
    to their input (analogous to the ``tf.layers`` syntax).

    Parameters
    ----------
    input : ``NengoObject``
        Object providing input to the layer
    layer_func : callable or `~nengo.neurons.NeuronType`
        A function that takes the value from ``input`` (represented as a
        ``tf.Tensor``) and maps it to some output value, or a Nengo neuron
        type, defining a nonlinearity that will be applied to ``input``.
    shape_in : tuple of int
        If not None, reshape the input to the given shape
    synapse : float or `~nengo.synapses.Synapse`
        Synapse to apply on connection from ``input`` to this layer
    transform : `~numpy.ndarray`
        Transform matrix to apply on connection from ``input`` to this layer
    return_conn : bool
        If True, also return the connection linking this layer to ``input``
    layer_args : dict
        These arguments will be passed to ``layer_func`` if it is callable, or
        `~nengo.Ensemble` if ``layer_func`` is a `~nengo.neurons.NeuronType`

    Returns
    -------
    node : `.TensorNode` or `~nengo.ensemble.Neurons`
        A TensorNode that implements the given layer function (if
        ``layer_func`` was a callable), or a Neuron object with the given
        neuron type, connected to ``input``
    conn : `~nengo.Connection`
        If ``return_conn`` is True, also returns the connection object linking
        ``input`` and ``node``.
    """

    if isinstance(transform, np.ndarray) and transform.ndim == 2:
        size_in = transform.shape[0]
    elif shape_in is not None:
        size_in = np.prod(shape_in)
    else:
        size_in = input.size_out

    if isinstance(layer_func, NeuronType):
        node = Ensemble(size_in, 1, neuron_type=layer_func,
                        **layer_args).neurons
    else:
        # add (ignored) time input and pass kwargs
        def node_func(_, x):
            return layer_func(x, **layer_args)

        # reshape input if necessary
        if shape_in is not None:
            node_func = reshaped(shape_in)(node_func)

        node = TensorNode(node_func, size_in=size_in)

    conn = Connection(input, node, synapse=synapse, transform=transform)

    return (node, conn) if return_conn else node
