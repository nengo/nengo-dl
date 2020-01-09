"""
Build classes for Nengo learning rule operators.
"""

from nengo import rc as nengo_rc
from nengo.builder import Signal
from nengo.builder.learning_rules import (
    SimBCM,
    SimOja,
    SimPES,
    SimVoja,
    get_post_ens,
    build_or_passthrough,
)
from nengo.builder.operator import Reset, DotInc, Copy
from nengo.learning_rules import PES
import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


@Builder.register(SimBCM)
class SimBCMBuilder(OpBuilder):
    """Build a group of `~nengo.builder.learning_rules.SimBCM`
    operators."""

    def __init__(self, ops, signals, config):
        super().__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.post_data = self.post_data.reshape(self.post_data.shape + (1,))
        self.theta_data = signals.combine([op.theta for op in ops])
        self.theta_data = self.theta_data.reshape(self.theta_data.shape + (1,))

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops for _ in range(op.post_filtered.shape[0])]
        )
        self.pre_data = self.pre_data.reshape(
            (self.post_data.shape[0], ops[0].pre_filtered.shape[0])
        )

        self.learning_rate = signals.op_constant(
            ops,
            [op.post_filtered.shape[0] for op in ops],
            "learning_rate",
            signals.dtype,
            shape=(1, -1, 1),
        )

        self.output_data = signals.combine([op.delta for op in ops])

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        theta = signals.gather(self.theta_data)

        post = self.learning_rate * signals.dt * post * (post - theta)

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
        super().__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.post_data = self.post_data.reshape(self.post_data.shape + (1,))

        self.pre_data = signals.combine(
            [op.pre_filtered for op in ops for _ in range(op.post_filtered.shape[0])]
        )
        self.pre_data = self.pre_data.reshape(
            (self.post_data.shape[0], ops[0].pre_filtered.shape[0])
        )

        self.weights_data = signals.combine([op.weights for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.learning_rate = signals.op_constant(
            ops,
            [op.post_filtered.shape[0] for op in ops],
            "learning_rate",
            signals.dtype,
            shape=(1, -1, 1),
        )

        self.beta = signals.op_constant(
            ops,
            [op.post_filtered.shape[0] for op in ops],
            "beta",
            signals.dtype,
            shape=(1, -1, 1),
        )

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        weights = signals.gather(self.weights_data)

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
        super().__init__(ops, signals, config)

        self.post_data = signals.combine([op.post_filtered for op in ops])
        self.post_data = self.post_data.reshape(self.post_data.shape + (1,))

        self.pre_data = signals.combine(
            [op.pre_decoded for op in ops for _ in range(op.post_filtered.shape[0])]
        )
        self.pre_data = self.pre_data.reshape(
            (self.post_data.shape[0], ops[0].pre_decoded.shape[0])
        )

        self.learning_data = signals.combine(
            [op.learning_signal for op in ops for _ in range(op.post_filtered.shape[0])]
        )
        self.learning_data = self.learning_data.reshape(self.learning_data.shape + (1,))
        self.encoder_data = signals.combine([op.scaled_encoders for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.scale = tf.constant(
            np.concatenate([op.scale[None, :, None] for op in ops], axis=1),
            dtype=signals.dtype,
        )

        self.learning_rate = signals.op_constant(
            ops,
            [op.post_filtered.shape[0] for op in ops],
            "learning_rate",
            signals.dtype,
            shape=(1, -1, 1),
        )

    def build_step(self, signals):
        pre = signals.gather(self.pre_data)
        post = signals.gather(self.post_data)
        learning_signal = signals.gather(self.learning_data)
        scaled_encoders = signals.gather(self.encoder_data)

        alpha = self.learning_rate * signals.dt * learning_signal

        update = alpha * (self.scale * post * pre - post * scaled_encoders)

        signals.scatter(self.output_data, update)

    @staticmethod
    def mergeable(x, y):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product
        return x.pre_decoded.shape[0] == y.pre_decoded.shape[0]


@NengoBuilder.register(PES)
def build_pes(model, pes, rule):
    """
    Builds a `nengo.PES` object into a Nengo model.

    Overrides the standard Nengo PES builder in order to avoid slicing on axes > 0
    (not currently supported in NengoDL).

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
    error = Signal(np.zeros(rule.size_in, dtype=nengo_rc.float_dtype), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    acts = build_or_passthrough(model, pes.pre_synapse, model.sig[conn.pre_obj]["out"])

    if not conn.is_decoded:
        # multiply error by post encoders to get a per-neuron error

        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"]

        if conn.post_obj is not conn.post:
            # in order to avoid slicing encoders along an axis > 0, we pad
            # `error` out to the full base dimensionality and then do the
            # dotinc with the full encoder matrix
            padded_error = Signal(
                np.zeros(encoders.shape[1], dtype=nengo_rc.float_dtype)
            )
            model.add_op(Copy(error, padded_error, dst_slice=conn.post_slice))
        else:
            padded_error = error

        # error = dot(encoders, error)
        local_error = Signal(
            np.zeros(post.n_neurons, dtype=nengo_rc.float_dtype), name="PES:encoded"
        )
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, padded_error, local_error, tag="PES:encode"))
    else:
        local_error = error

    model.operators.append(
        SimPES(acts, local_error, model.sig[rule]["delta"], pes.learning_rate)
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts


@Builder.register(SimPES)
class SimPESBuilder(OpBuilder):
    """Build a group of `~nengo.builder.learning_rules.SimPES` operators."""

    def __init__(self, ops, signals, config):
        super().__init__(ops, signals, config)

        self.error_data = signals.combine([op.error for op in ops])
        self.error_data = self.error_data.reshape((len(ops), ops[0].error.shape[0], 1))

        self.pre_data = signals.combine([op.pre_filtered for op in ops])
        self.pre_data = self.pre_data.reshape(
            (len(ops), 1, ops[0].pre_filtered.shape[0])
        )

        self.alpha = signals.op_constant(
            ops, [1 for _ in ops], "learning_rate", signals.dtype, shape=(1, -1, 1, 1)
        ) * (-signals.dt_val / ops[0].pre_filtered.shape[0])

        assert all(op.encoders is None for op in ops)

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
        return (
            x.pre_filtered.shape[0] == y.pre_filtered.shape[0]
            and x.error.shape[0] == y.error.shape[0]
        )
