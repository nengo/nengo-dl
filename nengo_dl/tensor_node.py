"""
TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.
"""

from nengo import Node, Connection, Ensemble, builder
from nengo.base import NengoObject
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError, SimulationError
from nengo.neurons import NeuronType
from nengo.params import Default, ShapeParam, Parameter, BoolParam
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


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

    if not isinstance(output, (tf.Tensor, tf.TensorSpec)):
        raise ValidationError(
            "TensorNode function must return a Tensor (got %s)" % type(output),
            attr="tensor_func",
        )

    if minibatch_size is not None and output.shape[0] != minibatch_size:
        raise ValidationError(
            "TensorNode output should have batch size %d (got %d)"
            % (minibatch_size, output.shape[0]),
            attr="tensor_func",
        )

    if output_d is not None and np.prod(output.shape[1:]) != output_d:
        raise ValidationError(
            "TensorNode output should have size %d (got shape %s with size %d)"
            % (minibatch_size, output.shape[1:], np.prod(output.shape[1:])),
            attr="tensor_func",
        )

    if dtype is not None and output.dtype != dtype:
        raise ValidationError(
            "TensorNode output should have dtype %s "
            "(got %s)" % (dtype, output.dtype),
            attr="tensor_func",
        )


class TensorFuncParam(Parameter):
    """Parameter for the ``tensor_func`` parameter of a `.TensorNode`."""

    def __init__(self, name, readonly=False):
        super(TensorFuncParam, self).__init__(name, optional=False, readonly=readonly)

    def coerce(self, node, func):
        """
        Performs validation on the function passed to TensorNode, and sets
        ``shape_out`` if necessary.

        Parameters
        ----------
        node : `.TensorNode`
            The node whose ``tensor_func`` parameter is being set.
        func : callable
            The function being assigned to the TensorNode.

        Returns
        -------
        output : callable
            The function after validation is applied.
        """

        output = super(TensorFuncParam, self).coerce(node, func)

        if not callable(func):
            raise ValidationError(
                "TensorNode output must be a function or Keras Layer",
                attr=self.name,
                obj=node,
            )

        if node.shape_out is None:
            if isinstance(func, tf.keras.layers.Layer):
                # we can use Keras' static shape inference to get the
                # output shape, which avoids having to build/call the layer
                if node.pass_time:
                    input_spec = [tf.TensorSpec(())]
                else:
                    input_spec = []
                if node.shape_in is not None:
                    input_spec += [tf.TensorSpec((1,) + node.shape_in)]
                if len(input_spec) == 1:
                    input_spec = input_spec[0]

                try:
                    result = func.compute_output_signature(input_spec)
                except Exception as e:
                    raise ValidationError(
                        "Attempting to automatically determine TensorNode output shape "
                        "by calling Layer.compute_output_signature produced an error. "
                        "If you would like to avoid this step, try manually setting "
                        "`TensorNode(..., shape_out=x)`. The error is shown below:\n%s"
                        % e,
                        attr=self.name,
                        obj=node,
                    )

            else:
                if node.pass_time:
                    args = (tf.constant(0.0),)
                else:
                    args = ()
                if node.shape_in is not None:
                    args += (tf.zeros((1,) + node.shape_in),)

                try:
                    result = func(*args)
                except Exception as e:
                    raise ValidationError(
                        "Attempting to automatically determine TensorNode output shape "
                        "by calling TensorNode function produced an error. "
                        "If you would like to avoid this step, try manually setting "
                        "`TensorNode(..., shape_out=x)`. The error is shown below:\n%s"
                        % e,
                        attr=self.name,
                        obj=node,
                    )

            validate_output(result)

            node.shape_out = result.shape[1:]

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
        sig_in = builder.Signal(np.zeros(node.size_in), name="%s.in" % node)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    sig_out = builder.Signal(np.zeros(node.size_out), name="%s.out" % node)

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
        The TensorNode function (``tensor_func``)
    time : `~nengo.builder.Signal`
        Signal representing the current simulation time (or None if pass_time is False)
    input : `~nengo.builder.Signal` or None
        Input Signal for the TensorNode (or None if no inputs)
    output : `~nengo.builder.Signal`
        Output Signal for the TensorNode
    shape_in : tuple of int or None
        Shape of input to TensorNode (if None, will leave the shape of input signal
        unchanged).
    tag : str
        A label associated with the operator, for debugging

    Notes
    -----
    1. sets ``[output]``
    2. incs ``[]``
    3. reads ``[time] if input is None else [time, input]``
    4. updates ``[]``
    """

    def __init__(self, func, time, input, output, shape_in, tag=None):
        super(SimTensorNode, self).__init__(tag=tag)

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

    def __init__(self, ops, signals, config):
        super(SimTensorNodeBuilder, self).__init__(ops, signals, config)

        # SimTensorNodes should never be merged
        assert len(ops) == 1
        op = ops[0]

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
            output = self.func.call(inputs)
        else:
            output = self.func(*inputs)

        validate_output(
            output,
            minibatch_size=signals.minibatch_size,
            output_d=self.dst_data.shape[0],
            dtype=signals.dtype,
        )

        signals.scatter(self.dst_data, output)


def tensor_layer(
    input,
    layer_func,
    shape_in=None,
    synapse=None,
    transform=1,
    return_conn=False,
    # TODO: remove once there is a black release with this
    #  bugfix https://github.com/psf/black/pull/763
    # fmt: off
    **layer_args
    # fmt: on
):
    """A utility function to construct TensorNodes that apply some function
    to their input (analogous to the ``tf.layers`` syntax).

    Parameters
    ----------
    input : ``NengoObject``
        Object providing input to the layer.
    layer_func : callable or ``keras.Layer`` or `~nengo.neurons.NeuronType`
        A function that takes the value from ``input`` (represented as a
        ``tf.Tensor``) and maps it to some output value,
        or a Keras layer type (which will be instantiated and applied
        to ``input``), or a Nengo neuron type (which will be instantiated in
        a Nengo Ensemble and applied to ``input``).
    shape_in : tuple of int
        If not None, reshape the input to the given shape.
    synapse : float or `~nengo.synapses.Synapse`
        Synapse to apply on connection from ``input`` to this layer.
    transform : `~numpy.ndarray`
        Transform matrix to apply on connection from ``input`` to this layer.
    return_conn : bool
        If True, also return the connection linking this layer to ``input``.
    layer_args : dict
        These arguments will be passed to `.TensorNode` if ``layer_func`` is a callable
        or Keras Layer, or `~nengo.Ensemble` if ``layer_func`` is a
        `~nengo.neurons.NeuronType`.

    Returns
    -------
    node : `.TensorNode` or `~nengo.ensemble.Neurons`
        A TensorNode that implements the given layer function (if
        ``layer_func`` was a callable), or a Neuron object with the given
        neuron type, connected to ``input``.
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
        node = Ensemble(size_in, 1, neuron_type=layer_func, **layer_args).neurons
    else:
        node = TensorNode(
            layer_func,
            shape_in=(size_in,) if shape_in is None else shape_in,
            pass_time=False,
            **layer_args,
        )

    conn = Connection(input, node, synapse=synapse, transform=transform)

    return (node, conn) if return_conn else node
