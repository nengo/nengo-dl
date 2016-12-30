from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja
import numpy as np
import tensorflow as tf

from nengo_deeplearning import Builder, utils, DEBUG


@Builder.register(SimBCM)
def sim_bcm(ops, signals, dt):
    pre = signals[[op.pre_filtered for op in ops]]
    pre = tf.expand_dims(pre, 0)

    post = signals[[op.post_filtered for op in ops]]

    learning_rate = tf.constant(
        [op.learning_rate for op in ops
         for _ in range(op.post_filtered.shape[0])],
        dtype=signals.dtype)

    theta = signals[[op.theta for op in ops]]

    post = learning_rate * dt * post * (post - theta)
    post = tf.expand_dims(post, 1)

    signals[[op.delta for op in ops]] = post * pre
    # return signals[op.delta]


@Builder.register(SimOja)
def sim_oja(ops, signals, dt):
    pre = signals[[op.pre_filtered for op in ops]]

    post = signals[[op.post_filtered for op in ops]]

    learning_rate = tf.constant(
        [op.learning_rate for op in ops
         for _ in range(op.post_filtered.shape[0])],
        dtype=signals.dtype)

    beta = tf.constant(
        [[op.beta] for op in ops for _ in range(op.post_filtered.shape[0])],
        dtype=signals.dtype)

    weights = signals[[op.weights for op in ops]]

    update = learning_rate * dt * post ** 2
    update = -beta * weights * tf.expand_dims(update, 1)
    update += (tf.expand_dims(learning_rate, 1) * dt * update *
               tf.expand_dims(pre, 0))

    signals[[op.delta for op in ops]] = update
    # return signals[op.delta]


@Builder.register(SimVoja)
def sim_voja(ops, signals, dt):
    # if DEBUG:
    #     print("sim_voja")
    #     print(op)
    #     print("pre_decoded", signals[op.pre_decoded])
    #     print("post_filtered", signals[op.post_filtered])
    #     print("scaled_encoders", signals[op.scaled_encoders])

    pre = signals[[op.pre_decoded for op in ops]]
    post = signals[[op.post_filtered for op in ops]]

    scale = tf.constant(
        np.concatenate([op.scale[:, None] for op in ops], axis=0),
        dtype=signals.dtype)

    learning_rate = tf.constant(
        [op.learning_rate for op in ops
         for _ in range(op.post_filtered.shape[0])], dtype=signals.dtype)

    learning_signal = signals[[op.learning_signal for op in ops
                               for _ in range(op.post_filtered.shape[0])]]

    scaled_encoders = signals[[op.scaled_encoders for op in ops]]

    alpha = tf.expand_dims(learning_rate * dt * learning_signal, 1)
    post = tf.expand_dims(post, 1)
    pre = tf.expand_dims(pre, 0)

    # alpha = utils.print_op(alpha, "alpha")
    # pre = utils.print_op(pre, "pre")
    # post = utils.print_op(post, "post")

    update = alpha * (scale * post * pre - post * scaled_encoders)

    # update = utils.print_op(update, "update")

    signals[[op.delta for op in ops]] = update
    # return signals[op.delta]
