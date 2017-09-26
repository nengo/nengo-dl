from nengo import Ensemble, Lowpass
from nengo.builder import Signal
from nengo.builder import Builder as NengoBuilder
from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja, get_post_ens
from nengo.builder.operator import Reset, DotInc, ElementwiseInc, Copy
from nengo.connection import Neurons
from nengo.exceptions import BuildError
from nengo.learning_rules import PES
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder


@Builder.register(SimBCM)
class SimBCMBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimBCM`
    operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.theta_data = signals.combine([op.theta for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))
        self.pre_data.load_indices()

        self.learning_rate = tf.constant(
            [[op.learning_rate] for op in ops
             for _ in range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

        self.output_data = signals.combine([op.delta for op in ops])

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        theta = signals.gather(self.theta_data)

        post = self.learning_rate * signals.dt * post * (post - theta)

        post = tf.expand_dims(post, 1)

        signals.scatter(self.output_data, post * pre)


@Builder.register(SimOja)
class SimOjaBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimOja`
        operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_filtered.shape[0]))
        self.pre_data.load_indices()

        self.weights_data = signals.combine([op.weights for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.learning_rate = tf.constant(
            [[[op.learning_rate]] for op in ops
             for _ in range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

        self.beta = tf.constant(
            [[[op.beta]] for op in ops for _ in
             range(op.post_filtered.shape[0])],
            dtype=signals.dtype)

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


@Builder.register(SimVoja)
class SimVojaBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.learning_rules.SimVoja`
        operators."""

    def __init__(self, ops, signals):
        self.post_data = signals.combine([op.post_filtered for op in ops])

        self.pre_data = signals.combine(
            [op.pre_decoded for op in ops
             for _ in range(op.post_filtered.shape[0])], load_indices=False)
        self.pre_data = self.pre_data.reshape((self.post_data.shape[0],
                                               ops[0].pre_decoded.shape[0]))
        self.pre_data.load_indices()

        self.learning_data = signals.combine(
            [op.learning_signal for op in ops
             for _ in range(op.post_filtered.shape[0])])
        self.encoder_data = signals.combine([op.scaled_encoders for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.scale = tf.constant(
            np.concatenate([op.scale[:, None, None] for op in ops], axis=0),
            dtype=signals.dtype)

        self.learning_rate = tf.constant(
            [[op.learning_rate] for op in ops
             for _ in range(op.post_filtered.shape[0])], dtype=signals.dtype)

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


@NengoBuilder.register(PES)
def build_pes(model, pes, rule):
    """Builds a :class:`~nengo:nengo.PES` object into a model.

    A re-implementation of the Nengo PES rule builder, so that we can avoid
    slicing the encoders.

    See :func:`~nengo:nengo.builder.learning_rules.build_pes:`.

    Parameters
    ----------
    model : :class:`~nengo:nengo.builder.Model`
        The model to build into.
    pes : :class:`~nengo:nengo.PES`
        Learning rule type to build.
    rule : :class:`~nengo:nengo.connection.LearningRule`
        The learning rule object corresponding to the neuron type.
    """

    conn = rule.connection

    # Create input error signal
    error = Signal(np.zeros(rule.size_in), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error  # error connection will attach here

    acts = model.build(Lowpass(pes.pre_tau), model.sig[conn.pre_obj]['out'])

    # Compute the correction, i.e. the scaled negative error
    correction = Signal(np.zeros(error.shape), name="PES:correction")
    model.add_op(Reset(correction))

    # correction = -learning_rate * (dt / n_neurons) * error
    n_neurons = (conn.pre_obj.n_neurons if isinstance(conn.pre_obj, Ensemble)
                 else conn.pre_obj.size_in)
    lr_sig = Signal(-pes.learning_rate * model.dt / n_neurons,
                    name="PES:learning_rate")
    model.add_op(ElementwiseInc(lr_sig, error, correction, tag="PES:correct"))

    if not conn.is_decoded:
        # NOTE: only this `if` block is changed from the regular nengo PES
        # builder

        post = get_post_ens(conn)
        weights = model.sig[conn]['weights']
        encoders = model.sig[post]['encoders']

        if conn.post_obj is not conn.post:
            # in order to avoid slicing encoders, we pad `correction` out to
            # the full base dimensionality and then do the dotinc with the full
            # encoder matrix
            padded_correction = Signal(np.zeros(encoders.shape[1]))
            model.add_op(Copy(correction, padded_correction,
                              dst_slice=conn.post_slice))
        else:
            padded_correction = correction

        # error = dot(encoders, correction)
        local_error = Signal(np.zeros(weights.shape[0]), name="PES:encoded")
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, padded_correction, local_error,
                            tag="PES:encode"))
    elif isinstance(conn.pre_obj, (Ensemble, Neurons)):
        local_error = correction
    else:  # pragma: no cover
        raise BuildError("'pre' object '%s' not suitable for PES learning"
                         % conn.pre_obj)

    # delta = local_error * activities
    model.add_op(Reset(model.sig[rule]['delta']))
    model.add_op(ElementwiseInc(
        local_error.column(), acts.row(), model.sig[rule]['delta'],
        tag="PES:Inc Delta"))

    # expose these for probes
    model.sig[rule]['error'] = error
    model.sig[rule]['correction'] = correction
    model.sig[rule]['activities'] = acts
