from nengo.neurons import RectifiedLinear, Sigmoid, LIF, LIFRate
from nengo.builder.neurons import SimNeurons
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


@Builder.register(SimNeurons)
def sim_neurons(op, signals, dt):
    if DEBUG:
        print("sim_neurons")
        print(op)
        print("J", signals[op.J])
        print("output", signals[op.output])
        print("states", [str(signals[s]) for s in op.states])

    output = signals[op.output]
    states = [signals[s] for s in op.states]

    if type(op.neurons) == RectifiedLinear:
        result = (tf.nn.relu(signals[op.J]),)
    elif type(op.neurons) == Sigmoid:
        result = (tf.nn.sigmoid(signals[op.J]) / op.neurons.tau_ref,)
    elif type(op.neurons) == LIF:
        result = lif_spiking(op.neurons, signals[op.J], dt, *states)
    elif type(op.neurons) == LIFRate:
        result = lif_rate(op.neurons, signals[op.J])
    else:
        # generic python function
        def return_step_math(dt, J, output, *states):
            op.neurons.step_math(dt, J, output, *states)

            return (output,) + states

        result = tf.py_func(return_step_math,
                            [tf.constant(dt), signals[op.J], output] + states,
                            [output.dtype] + [s.dtype for s in states],
                            name=utils.sanitize_name(repr(op.neurons)))

    for i in range(len(states)):
        signals.assign_view(op.states[i], result[i + 1])

    # we need the control_dependencies to force the state update operators
    # to run (otherwise they look like unused nodes and get optimized out)
    with tf.control_dependencies([signals[s] for s in op.states]):
        output = signals.assign_view(op.output, result[0])

    return output


def lif_rate(neurons, J):
    j = J - 1
    indices = tf.cast(tf.where(j > 0), tf.int32)

    return (tf.scatter_nd(
        indices,
        1 / (neurons.tau_ref +
             neurons.tau_rc * tf.log1p(1 / tf.gather_nd(j, indices))),
        tf.shape(j)),)


def lif_spiking(neurons, J, dt, voltage, refractory):
    refractory = refractory - dt
    delta_t = tf.clip_by_value(dt - refractory, 0, dt)

    voltage = voltage - (J - voltage) * (tf.exp(-delta_t / neurons.tau_rc) - 1)

    spiked = voltage > 1
    # spikes = tf.scatter_nd(indices, 1/dt, tf.shape(voltage))
    spikes = tf.cast(spiked, tf.float64) / dt

    indices = tf.cast(tf.where(spiked), tf.int32)
    t_spike = neurons.tau_ref + dt + neurons.tau_rc * tf.log1p(
        -(tf.gather_nd(voltage, indices) - 1) / (tf.gather_nd(J, indices) - 1))
    refractory = tf.where(
        spiked, tf.scatter_nd(indices, t_spike, tf.shape(refractory)),
        refractory)

    # TODO: is it faster to do the scatter/gather, or to just do the
    # computation on the full array?
    # t_spike = dt + neurons.tau_rc * tf.log1p(-(voltage - 1) / (J - 1))
    # refractory = tf.where(spiked, neurons.tau_ref + t_spike, refractory)

    voltage = tf.where(spiked, tf.zeros_like(voltage),
                       tf.maximum(voltage, neurons.min_voltage))

    return spikes, voltage, refractory
