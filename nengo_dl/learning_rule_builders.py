"""
Build classes for Nengo learning rule operators.
"""

from distutils.version import LooseVersion

from nengo import Lowpass
from nengo.builder import Signal
from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja, get_post_ens
from nengo.builder.operator import Operator, Reset, DotInc, Copy
from nengo.learning_rules import PES
from nengo.version import version as nengo_version
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


@Builder.register(SimBCM)
class SimBCMBuilder(OpBuilder):
    """Build a group of `~nengo.builder.learning_rules.SimBCM`
    operators."""

    def __init__(self, ops, signals, config):
        super(SimBCMBuilder, self).__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.theta_data = signals.combine([op.theta for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))

        self.learning_rate = signals.op_constant(
            ops, [op.post_filtered.shape[0] for op in ops], "learning_rate",
            signals.dtype, ndims=3)

        self.output_data = signals.combine([op.delta for op in ops])

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        theta = signals.gather(self.theta_data)

        post = self.learning_rate * signals.dt * post * (post - theta)

        post = tf.expand_dims(post, 1)

        signals.scatter(self.output_data, post * pre)

    @staticmethod
    def mergeable(x, y):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product
        return x.pre_filtered.shape[0] == y.pre_filtered.shape[0]


@Builder.register(SimOja)
class SimOjaBuilder(OpBuilder):
    """Build a group of `~nengo.builder.learning_rules.SimOja`
        operators."""

    def __init__(self, ops, signals, config):
        super(SimOjaBuilder, self).__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))

        self.weights_data = signals.combine([op.weights for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.learning_rate = signals.op_constant(
            ops, [op.post_filtered.shape[0] for op in ops], "learning_rate",
            signals.dtype, ndims=3)

        self.beta = signals.op_constant(
            ops, [op.post_filtered.shape[0] for op in ops], "beta",
            signals.dtype, ndims=3)

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        weights = signals.gather(self.weights_data)

        post = tf.expand_dims(post, 1)

        alpha = self.learning_rate * signals.dt

        update = alpha * post ** 2
        update *= -self.beta * weights
        update += alpha * post * pre

        signals.scatter(self.output_data, update)

    @staticmethod
    def mergeable(x, y):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product
        return x.pre_filtered.shape[0] == y.pre_filtered.shape[0]


@Builder.register(SimVoja)
class SimVojaBuilder(OpBuilder):
    """Build a group of `~nengo.builder.learning_rules.SimVoja`
        operators."""

    def __init__(self, ops, signals, config):
        super(SimVojaBuilder, self).__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_decoded for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_decoded.shape[0]))

        self.learning_data = signals.combine(
            [op.learning_signal for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.encoder_data = signals.combine([op.scaled_encoders for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.scale = signals.constant(
            np.concatenate([op.scale[:, None, None] for op in ops], axis=0),
            dtype=signals.dtype)

        self.learning_rate = signals.op_constant(
            ops, [op.post_filtered.shape[0] for op in ops], "learning_rate",
            signals.dtype)

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        learning_signal = signals.gather(self.learning_data)
        scaled_encoders = signals.gather(self.encoder_data)

        alpha = tf.expand_dims(
            self.learning_rate * signals.dt * learning_signal, 1)
        post = tf.expand_dims(post, 1)

        update = alpha * (self.scale * post * pre - post * scaled_encoders)

        signals.scatter(self.output_data, update)

    @staticmethod
    def mergeable(x, y):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product
        return x.pre_decoded.shape[0] == y.pre_decoded.shape[0]


class SimPES(Operator):  # pylint: disable=abstract-method
    r"""
    Calculate connection weight change according to the PES rule.

    Implements the PES learning rule of the form

    .. math:: \Delta \omega_{ij} = \frac{\kappa}{n} e_j a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`n` is the number of presynaptic neurons
    * :math:`e_j` is the error for the jth output dimension, and
    * :math:`a_i` is the activity of a presynaptic neuron.

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str (Default: None)
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[delta]``
    2. incs ``[]``
    3. reads ``[pre_filtered, error]``
    4. updates ``[]``
    """

    def __init__(self, pre_filtered, error, delta, learning_rate, tag=None):
        super(SimPES, self).__init__(tag=tag)

        self.pre_filtered = pre_filtered
        self.error = error
        self.delta = delta
        self.learning_rate = learning_rate

        # TODO: change this to an update when (if) we make that change in nengo
        self.sets = [delta]
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = []

    def _descstr(self):
        return 'pre=%s, error=%s -> %s' % (
            self.pre_filtered, self.error, self.delta)


@NengoBuilder.register(PES)
def build_pes(model, pes, rule):
    """
    Builds a `nengo.PES` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    pes : PES
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `nengo.PES` instance.
    """

    conn = rule.connection

    # Create input error signal
    error = Signal(np.zeros(rule.size_in), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error  # error connection will attach here

    if LooseVersion(nengo_version) < "2.7.1":
        acts = model.build(
            Lowpass(pes.pre_tau), model.sig[conn.pre_obj]["out"])
    else:
        acts = model.build(pes.pre_synapse, model.sig[conn.pre_obj]["out"])

    if not conn.is_decoded:
        # multiply error by post encoders to get a per-neuron error

        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"]

        if conn.post_obj is not conn.post:
            # in order to avoid slicing encoders along an axis > 0, we pad
            # `error` out to the full base dimensionality and then do the
            # dotinc with the full encoder matrix
            padded_error = Signal(np.zeros(encoders.shape[1]))
            model.add_op(Copy(error, padded_error,
                              dst_slice=conn.post_slice))
        else:
            padded_error = error

        # error = dot(encoders, error)
        local_error = Signal(np.zeros(post.n_neurons), name="PES:encoded")
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, padded_error, local_error,
                            tag="PES:encode"))
    else:
        local_error = error

    model.operators.append(SimPES(acts, local_error, model.sig[rule]["delta"],
                                  pes.learning_rate))

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts


# remove 'correction' from probeable attributes
PES.probeable = ("error", "activities", "delta")


@Builder.register(SimPES)
class SimPESBuilder(OpBuilder):
    """Build a group of `.SimPES` operators."""

    def __init__(self, ops, signals, config):
        super(SimPESBuilder, self).__init__(ops, signals, config)

        self.error_data = signals.combine([op.error for op in ops])
        self.error_data = self.error_data.reshape(
            (len(ops), ops[0].error.shape[0], 1))

        self.pre_data = signals.combine([op.pre_filtered for op in ops])
        self.pre_data = self.pre_data.reshape(
            (len(ops), 1, ops[0].pre_filtered.shape[0]))

        self.alpha = signals.op_constant(
            ops, [1 for _ in ops], "learning_rate", signals.dtype, ndims=4) * (
                -signals.dt_val / ops[0].pre_filtered.shape[0])

        self.output_data = signals.combine([op.delta for op in ops])

    def build_step(self, signals):
        pre_filtered = signals.gather(self.pre_data)
        error = signals.gather(self.error_data)

        error *= self.alpha
        update = error * pre_filtered

        signals.scatter(self.output_data, update)

    @staticmethod
    def mergeable(x, y):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product.
        # the error signals also have to have the same shape.
        return (x.pre_filtered.shape[0] == y.pre_filtered.shape[0] and
                x.error.shape[0] == y.error.shape[0])
