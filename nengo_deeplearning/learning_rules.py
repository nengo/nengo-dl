from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja
import tensorflow as tf

from nengo_deeplearning import operators, Builder, DEBUG


@Builder.register(SimBCM)
def sim_bcm(op, signals, dt):
    pre = tf.expand_dims(signals[op.pre_filtered], 0)

    post = signals[op.post_filtered]
    post = op.learning_rate * dt * post * (post - signals[op.theta])
    post = tf.expand_dims(post, 1)

    return signals.assign_view(op.delta, post * pre)


@Builder.register(SimOja)
def sim_oja(op, signals, dt):
    update = op.learning_rate * dt * signals[op.post_filtered] ** 2
    update = -op.beta * signals[op.weights] * tf.expand_dims(update, 1)
    update += op.learning_rate * dt * update * tf.expand_dims(
        signals[op.pre_filtered], 0)

    return signals.assign_view(op.delta, update)


@Builder.register(SimVoja)
def sim_voja(op, signals, dt):
    if DEBUG:
        print("sim_voja")
        print(op)
        print("pre_decoded", signals[op.pre_decoded])
        print("post_filtered", signals[op.post_filtered])
        print("scaled_encoders", signals[op.scaled_encoders])

    scale = op.scale[:, None]
    alpha = op.learning_rate * dt * signals[op.learning_signal]
    post = tf.expand_dims(signals[op.post_filtered], 1)
    pre = tf.expand_dims(signals[op.pre_decoded], 0)

    update = alpha * (scale * post * pre - post * signals[op.scaled_encoders])

    return signals.assign_view(op.delta, update)