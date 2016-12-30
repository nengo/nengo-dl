from nengo.neurons import RectifiedLinear, Sigmoid, LIF, LIFRate
from nengo.builder.neurons import SimNeurons
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG

# the neuron types for which we have a custom tensorflow implementation
TF_NEURON_IMPL = (RectifiedLinear, Sigmoid, LIF, LIFRate)


@Builder.register(SimNeurons)
def sim_neurons(ops, signals, dt):
    if DEBUG:
        print("sim_neurons")
        print([op for op in ops])
        print("J", [op.J for op in ops])

    neuron_type = type(ops[0].neurons)

    if neuron_type in TF_NEURON_IMPL:
        # note: we do this two-step check (even though it's redundant) to make
        # sure that TF_NEURON_IMPL is kept up to date

        J = signals[[op.J for op in ops]]
        n_states = len(ops[0].states)
        states = [signals.combine([op.states[i] for op in ops])
                  for i in range(n_states)]

        if neuron_type == RectifiedLinear:
            result = (tf.nn.relu(J),)
        elif neuron_type == Sigmoid:
            tau_ref = tf.constant(
                [op.neurons.tau_ref for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            result = (tf.nn.sigmoid(J) / tau_ref,)
        elif neuron_type == LIF:
            tau_ref = tf.constant(
                [op.neurons.tau_ref for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            tau_rc = tf.constant(
                [op.neurons.tau_rc for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            min_voltage = tf.constant(
                [op.neurons.min_voltage for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            result = lif_spiking(tau_ref, tau_rc, min_voltage, J, dt,
                                 *[signals[s] for s in states])
        elif neuron_type == LIFRate:
            tau_ref = tf.constant(
                [op.neurons.tau_ref for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            tau_rc = tf.constant(
                [op.neurons.tau_rc for op in ops
                 for _ in range(op.J.shape[0])], dtype=signals.dtype)
            result = lif_rate(tau_ref, tau_rc, J)

        signals[[op.output for op in ops]] = result[0]
        for i in range(n_states):
            signals[states[i]] = result[i + 1]

        if DEBUG:
            print("output states", [str(x) for x in result[1:]])

        # return result[1:]
        return

    # output_sig = signals[[op.output for op in ops]]
    output_dtype = utils.cast_dtype(ops[0].output.dtype, signals.dtype)

    # state_dtypes = [utils.cast_dtype(s.dtype, signals.dtype)
    #                 for s in states]

    # generic python function
    # def return_step_math(dt, J, output, *states):
    #     op.neurons.step_math(dt, J, output, *states)
    #
    #     return ([output.astype(output_dtype.as_numpy_dtype)] +
    #             [x.astype(d.as_numpy_dtype) for x, d in zip(states,
    #                                                         state_dtypes)])

    def neuron_step_math(dt, J, states):
        output = None
        J_offset = 0
        state_offset = 0
        for i, op in enumerate(ops):
            # slice out the individual state vectors from the overall array
            op_J = J[J_offset:J_offset + op.J.shape[0]]
            J_offset += op.J.shape[0]

            op_states = []
            for s in op.states:
                op_states += [
                    states[state_offset:state_offset + s.shape[0]]]
                state_offset += s.shape[0]

            # blank output variable
            neuron_output = np.zeros(op.output.shape,
                                     output_dtype.as_numpy_dtype)

            # call step_math function
            op.neurons.step_math(dt, op_J, neuron_output, *op_states)

            # concatenate outputs
            if output is None:
                output = neuron_output
            else:
                output = np.concatenate((output, neuron_output), axis=0)

        return [output] + [states]

    J = signals[[op.J for op in ops]]
    states = signals.combine([s for op in ops for s in op.states])
    states_dtype = [] if states is None else [states.dtype]

    neuron_out, state_out = tf.py_func(
        neuron_step_math,
        [tf.constant(dt), J] + ([] if states is None else
                                [signals[states]]),
        [output_dtype] + states_dtype,
        name=utils.sanitize_name("_".join([repr(op.neurons)
                                           for op in ops])))
    # for i, s in enumerate([op.output] + op.states):
    #     result[i].set_shape(s.shape)
    neuron_out.set_shape(np.sum([op.output.shape[0] for op in ops]))
    state_out.set_shape(states.shape)

    signals[[op.output for op in ops]] = neuron_out
    signals[states] = state_out
    # return [signals[s] for op in ops for s in op.states]


def lif_rate(tau_ref, tau_rc, J):
    j = J - 1
    indices = tf.cast(tf.where(j > 0), tf.int32)
    tau_ref = tf.gather_nd(tau_ref, indices)
    tau_rc = tf.gather_nd(tau_rc, indices)
    j = tf.gather_nd(j, indices)

    return (tf.scatter_nd(
        indices, 1 / (tau_ref + tau_rc * tf.log1p(1 / j)), tf.shape(J)),)


def lif_spiking(tau_ref, tau_rc, min_voltage, J, dt, voltage, refractory):
    # TODO: use sparse operators when dealing with spikes

    refractory = refractory - dt
    delta_t = tf.clip_by_value(dt - refractory, 0, dt)

    voltage = voltage - (J - voltage) * (tf.exp(-delta_t / tau_rc) - 1)

    spiked = voltage > 1
    spikes = tf.cast(spiked, J.dtype) / dt

    indices = tf.cast(tf.where(spiked), tf.int32)
    tau_rc = tf.gather_nd(tau_rc, indices)
    tau_ref = tf.gather_nd(tau_ref, indices)
    J = tf.gather_nd(J, indices)
    t_spike = tau_ref + dt + tau_rc * tf.log1p(
        -(tf.gather_nd(voltage, indices) - 1) / (J - 1))
    refractory = tf.where(
        spiked, tf.scatter_nd(indices, t_spike, tf.shape(refractory)),
        refractory)

    # TODO: is it faster to do the scatter/gather as above, or to just do the
    # computation on the full array?
    # t_spike = dt + neurons.tau_rc * tf.log1p(-(voltage - 1) / (J - 1))
    # refractory = tf.where(spiked, neurons.tau_ref + t_spike, refractory)

    voltage = tf.where(spiked, tf.zeros_like(voltage),
                       tf.maximum(voltage, min_voltage))

    return spikes, voltage, refractory
