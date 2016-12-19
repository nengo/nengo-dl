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

    # signal creation needs to be handled specifically for this op, because
    # the output/states are marked as sets, but they're actually reads as well.
    # (at least the states are reads; it is unclear whether there are neuron
    # types that need to read output, or if we just need a blank dummy
    # variable as I've assumed here)
    assert op.output not in signals
    signals[op.output] = tf.zeros(op.output.shape,
                                  utils.cast_dtype(op.output.dtype,
                                                   signals.dtype))
    for s in op.states:
        assert s not in signals
        signals.create_variable(s)

    output = signals[op.output]
    states = [signals[s] for s in op.states]

    if DEBUG:
        print("output", output)
        print("states", [str(s) for s in states])

    if type(op.neurons) == RectifiedLinear:
        result = (tf.nn.relu(signals[op.J]),)
    elif type(op.neurons) == Sigmoid:
        result = (tf.nn.sigmoid(signals[op.J]) / op.neurons.tau_ref,)
    elif type(op.neurons) == LIF:
        result = lif_spiking(op.neurons, signals[op.J], dt, *states)
    elif type(op.neurons) == LIFRate:
        result = lif_rate(op.neurons, signals[op.J])
    else:
        output_dtype = utils.cast_dtype(output.dtype, signals.dtype)
        state_dtypes = [utils.cast_dtype(s.dtype, signals.dtype)
                        for s in states]

        # generic python function
        def return_step_math(dt, J, output, *states):
            op.neurons.step_math(dt, J, output, *states)

            return ([output.astype(output_dtype.as_numpy_dtype)] +
                    [x.astype(d.as_numpy_dtype) for x, d in zip(states,
                                                                state_dtypes)])

        result = tf.py_func(return_step_math,
                            [tf.constant(dt), signals[op.J], output] + states,
                            [output_dtype] + state_dtypes,
                            name=utils.sanitize_name(repr(op.neurons)))

    for i in range(len(states)):
        signals[op.states[i]] = result[i + 1]

    # we need the control_dependencies to force the state update operators
    # to run (otherwise they look like unused nodes and get optimized out)
    with tf.control_dependencies([signals[s] for s in op.states]):
        # note: need the identity so that an operator is created within
        # the dependency scope
        signals[op.output] = tf.identity(result[0])

    return signals[op.output]


def lif_rate(neurons, J):
    j = J - 1
    indices = tf.cast(tf.where(j > 0), tf.int32)

    return (tf.scatter_nd(
        indices,
        1 / (neurons.tau_ref +
             neurons.tau_rc * tf.log1p(1 / tf.gather_nd(j, indices))),
        tf.shape(j)),)


def lif_spiking(neurons, J, dt, voltage, refractory):
    # TODO: use sparse operators when dealing with spikes

    refractory = refractory - dt
    delta_t = tf.clip_by_value(dt - refractory, 0, dt)

    voltage = voltage - (J - voltage) * (tf.exp(-delta_t / neurons.tau_rc) - 1)

    spiked = voltage > 1
    spikes = tf.cast(spiked, J.dtype) / dt

    indices = tf.cast(tf.where(spiked), tf.int32)
    t_spike = neurons.tau_ref + dt + neurons.tau_rc * tf.log1p(
        -(tf.gather_nd(voltage, indices) - 1) / (tf.gather_nd(J, indices) - 1))
    refractory = tf.where(
        spiked, tf.scatter_nd(indices, t_spike, tf.shape(refractory)),
        refractory)

    # TODO: is it faster to do the scatter/gather as above, or to just do the
    # computation on the full array?
    # t_spike = dt + neurons.tau_rc * tf.log1p(-(voltage - 1) / (J - 1))
    # refractory = tf.where(spiked, neurons.tau_ref + t_spike, refractory)

    voltage = tf.where(spiked, tf.zeros_like(voltage),
                       tf.maximum(voltage, neurons.min_voltage))

    return spikes, voltage, refractory
