from nengo import Node, Connection, Ensemble, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError, BuildError
from nengo.neurons import NeuronType
from nengo.params import Default, IntParam, Parameter
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


class TensorFuncParam(Parameter):
    """Performs validation on the function passed to TensorNode, and sets
    ``size_out`` if necessary."""

    def __init__(self, name, readonly=False):
        super(TensorFuncParam, self).__init__(
            name, optional=False, readonly=readonly)

    def coerce(self, node, func):
        output = super(TensorFuncParam, self).coerce(node, func)

        if node.size_out is None:
            node.size_out = self.check_size_out(node, func)

        return output

    def check_size_out(self, node, func):
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
                    "Calling TensorNode function with arguments %s produced "
                    "an error:\n%s" % (args, e), attr=self.name, obj=node)

        if not isinstance(result, tf.Tensor):
            raise ValidationError("TensorNode function must return a Tensor",
                                  attr=self.name, obj=node)

        if result.get_shape().ndims != 2:
            raise ValidationError("Node output must be a minibatched vector "
                                  "(got shape %s)" % result.get_shape(),
                                  attr=self.name, obj=node)

        return result.get_shape()[1].value


class TensorNode(Node):
    """Inserts TensorFlow code into a Nengo model.  A TensorNode operates in
    much the same way as a :class:`~nengo:nengo.Node`, except its inputs and
    outputs are defined using TensorFlow operations.

    The TensorFlow code is defined in a function or callable class
    (``tensor_func``).  This function accepts the current simulation time as
    input, or the current simulation time and a Tensor ``x`` if
    ``node.size_in > 0``.  ``x`` will have shape
    ``(sim.minibatch_size, node.size_in``), and the function should return a
    Tensor with shape ``(sim.minibatch_size, node.size_out)``.
    ``node.size_out`` will be inferred by calling the function once and
    checking the output, if it isn't set when the Node is created.

    If ``tensor_func`` has a ``pre_build`` attribute, that function will be
    called once when the model is constructed.  This can be used to compute any
    constant values or set up variables -- things that don't need to
    execute every simulation timestep.

    .. code-block:: python

        def pre_build(shape_in, shape_out):
            print(shape_in)  # (minibatch_size, node.size_in)
            print(shape_out)  # (minibatch_size, node.size_out)

    If ``tensor_func`` has a ``post_build`` attribute, that function will be
    called after the simulator is created and whenever it is reset.  This can
    be used to set any random elements in the TensorNode or perform any
    post-initialization setup required by the node (e.g., loading pretrained
    weights).

    .. code-block:: python

        def post_build(sess, rng):
            print(sess)  # the TensorFlow simulation session object
            print(rng)  # random number generator (np.random.RandomState)

    Parameters
    ----------
    tensor_func : callable
        A function that maps node inputs to outputs
    size_in : int, optional (Default: 0)
        The number of elements in the input vector
    size_out : int, optional (Default: None)
        The number of elements in the output vector (if None, value will be
        inferred by calling ``tensor_func``)
    label : str, optional (Default: None)
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
        raise BuildError(
            "TensorNode does not have an `output` attribute (this probably "
            "means you are trying to use a TensorNode inside a Simulator "
            "other than Nengo DL)")


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
    """Operator for TensorNodes (constructed by :func:`.build_tensor_node`).

    Parameters
    ----------
    func : callable
        The TensorNode function (``tensor_func``)
    time : :class:`~nengo:nengo.builder.Signal`
        Signal representing the current simulation time
    input : :class:`~nengo:nengo.builder.Signal` or None
        Input Signal for the TensorNode (or None if size_in==0)
    output : :class:`~nengo:nengo.builder.Signal`
        Output Signal for the TensorNode
    tag : str, optional
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
    """Builds a :class:`~.tensor_node.SimTensorNode` operator into a NengoDL
    model."""

    def __init__(self, ops, signals):
        super(SimTensorNodeBuilder, self).__init__(ops, signals)

        # SimTensorNodes should never be merged
        assert len(ops) == 1
        op = ops[0]

        if op.input is None:
            self.src_data = None
        else:
            self.src_data = signals.sig_map[op.input]
            self.src_data.load_indices(constant=signals.constant)
            assert self.src_data.ndim == 1

        self.dst_data = signals.sig_map[op.output]
        self.dst_data.load_indices(constant=signals.constant)

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
    callable
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
    layer_func : callable or :class:`~nengo:nengo.neurons.NeuronType`
        A function that takes the value from ``input`` (represented as a
        ``tf.Tensor``) and maps it to some output value, or a Nengo neuron
        type, defining a nonlinearity that will be applied to ``input``.
    shape_in : tuple of int, optional
        If not None, reshape the input to the given shape
    synapse : float or :class:`~nengo:nengo.synapses.Synapse`, optional
        Synapse to apply on connection from ``input`` to this layer
    transform : :class:`~numpy:numpy.ndarray`, optional
        Transform matrix to apply on connection from ``input`` to this layer
    return_conn : bool, optional
        If True, also return the connection linking this layer to ``input``
    layer_args : dict, optional
        These arguments will be passed to ``layer_func`` if it is callable, or
        :class:`~nengo:nengo.Ensemble` if ``layer_func`` is a
        :class:`~nengo:nengo.neurons.NeuronType`

    Returns
    -------
    :class:`.TensorNode` or :class:`~nengo:nengo.ensemble.Neurons`
        A TensorNode that implements the given layer function (if
        ``layer_func`` was a callable), or a Neuron object with the given
        neuron type, connected to ``input``
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
