from nengo.neurons import RectifiedLinear, Sigmoid, LIF, LIFRate
from nengo.builder.neurons import SimNeurons
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG
from nengo_deeplearning.builder import Builder, OpBuilder

# the neuron types for which we have a custom tensorflow implementation
TF_NEURON_IMPL = (RectifiedLinear, Sigmoid, LIF, LIFRate)


@Builder.register(SimNeurons)
class SimNeuronsBuilder(OpBuilder):
    def __init__(self, ops, signals):
        if DEBUG:
            print("sim_neurons")
            print([op for op in ops])
            print("J", [op.J for op in ops])

        self.neuron_type = type(ops[0].neurons)
        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        # if we have a custom tensorflow implementation for this neuron type,
        # then we build that. otherwise we'll just execute the neuron step
        # function externally (using `tf.py_func`), so we just need to set up
        # the inputs/outputs for that.
        if self.neuron_type in TF_NEURON_IMPL:
            # note: we do this two-step check (even though it's redundant) to
            # make sure that TF_NEURON_IMPL is kept up to date

            self.state_data = [signals.combine([op.states[i] for op in ops])
                               for i in range(len(ops[0].states))]

            if self.neuron_type == Sigmoid:
                self.tau_ref = tf.constant(
                    [op.neurons.tau_ref for op in ops
                     for _ in range(op.J.shape[0])], dtype=signals.dtype)
            elif self.neuron_type in (LIF, LIFRate):
                self.tau_ref = tf.constant(
                    [op.neurons.tau_ref for op in ops
                     for _ in range(op.J.shape[0])], dtype=signals.dtype)
                self.tau_rc = tf.constant(
                    [op.neurons.tau_rc for op in ops
                     for _ in range(op.J.shape[0])], dtype=signals.dtype)
                if self.neuron_type == LIF:
                    self.min_voltage = tf.constant(
                        [op.neurons.min_voltage for op in ops
                         for _ in range(op.J.shape[0])], dtype=signals.dtype)
        else:
            # we combine all the state signals into a single tensor
            self.state_data = signals.combine([s for op in ops
                                               for s in op.states])

            def neuron_step_math(dt, J, states):
                output = None
                J_offset = 0
                state_offset = 0
                for i, op in enumerate(ops):
                    # slice out the individual state vectors from the overall
                    # array
                    op_J = J[J_offset:J_offset + op.J.shape[0]]
                    J_offset += op.J.shape[0]

                    op_states = []
                    for s in op.states:
                        op_states += [
                            states[state_offset:state_offset + s.shape[0]]]
                        state_offset += s.shape[0]

                    # blank output variable
                    neuron_output = np.zeros(
                        op.output.shape, self.output_data.dtype)

                    # call step_math function
                    # note: `op_states` are views into `states`, which will
                    # be updated in-place
                    op.neurons.step_math(dt, op_J, neuron_output, *op_states)

                    # concatenate outputs
                    if output is None:
                        output = neuron_output
                    else:
                        output = np.concatenate((output, neuron_output),
                                                axis=0)

                return output, states

            self.neuron_step_math = neuron_step_math
            self.neuron_step_math.__name__ = utils.sanitize_name(
                "_".join([repr(op.neurons) for op in ops]))

    def build_step(self, signals):
        J = signals.gather(self.J_data)

        if self.neuron_type in TF_NEURON_IMPL:
            if self.neuron_type == RectifiedLinear:
                result = (tf.nn.relu(J),)
            elif self.neuron_type == Sigmoid:
                result = (tf.nn.sigmoid(J) / self.tau_ref,)
            elif self.neuron_type == LIF:
                result = lif_spiking(
                    self.tau_ref, self.tau_rc, self.min_voltage, J, signals.dt,
                    *[signals.gather(s) for s in self.state_data])
            elif self.neuron_type == LIFRate:
                result = lif_rate(self.tau_ref, self.tau_rc, J)

            signals.scatter(self.output_data, result[0])
            for i, s in enumerate(self.state_data):
                signals.scatter(self.state_data[i], result[i + 1])
        else:
            states = ([] if self.state_data == [] else
                      [signals.gather(self.state_data)])
            states_dtype = ([] if self.state_data == [] else
                            [self.state_data.dtype])

            neuron_out, state_out = tf.py_func(
                self.neuron_step_math, [signals.dt, J] + states,
                [self.output_data.dtype] + states_dtype,
                name=self.neuron_step_math.__name__)
            neuron_out.set_shape(self.output_data.shape)
            state_out.set_shape(self.state_data.shape)

            signals.scatter(self.output_data, neuron_out)
            if self.state_data is not None:
                signals.scatter(self.state_data, state_out)


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

    refractory -= dt
    delta_t = tf.clip_by_value(dt - refractory, 0, dt)

    voltage -= (J - voltage) * (tf.exp(-delta_t / tau_rc) - 1)

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
