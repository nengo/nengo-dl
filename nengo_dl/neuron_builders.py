"""
Build classes for Nengo neuron operators.
"""

import logging

from nengo.builder.neurons import SimNeurons
from nengo.neurons import (RectifiedLinear, SpikingRectifiedLinear, Sigmoid,
                           LIF, LIFRate)
import numpy as np
import tensorflow as tf

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.neurons import SoftLIFRate

logger = logging.getLogger(__name__)


class GenericNeuronBuilder(OpBuilder):
    """
    Builds all neuron types for which there is no custom Tensorflow
    implementation.

    Notes
    -----
    These will be executed as native Python functions, requiring execution to
    move in and out of TensorFlow.  This can significantly slow down the
    simulation, so any performance-critical neuron models should consider
    adding a custom TensorFlow implementation for their neuron type instead.
    """

    def __init__(self, ops, signals, config):
        super(GenericNeuronBuilder, self).__init__(ops, signals, config)

        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])
        self.state_data = [signals.combine([op.states[i] for op in ops])
                           for i in range(len(ops[0].states))]

        self.prev_result = []

        def neuron_step_math(dt, J, *states):  # pragma: no cover
            output = None
            J_offset = 0
            state_offset = [0 for _ in states]
            for op in ops:
                # slice out the individual state vectors from the overall
                # array
                op_J = J[J_offset:J_offset + op.J.shape[0]]
                J_offset += op.J.shape[0]

                op_states = []
                for j, s in enumerate(op.states):
                    op_states += [states[j][state_offset[j]:
                                            state_offset[j] + s.shape[0]]]
                    state_offset[j] += s.shape[0]

                # call step_math function
                # note: `op_states` are views into `states`, which will
                # be updated in-place
                mini_out = []
                for j in range(signals.minibatch_size):
                    # blank output variable
                    neuron_output = np.zeros(
                        op.output.shape, self.output_data.dtype)
                    op.neurons.step_math(dt, op_J[..., j], neuron_output,
                                         *[s[..., j] for s in op_states])
                    mini_out += [neuron_output]
                neuron_output = np.stack(mini_out, axis=-1)

                # concatenate outputs
                if output is None:
                    output = neuron_output
                else:
                    output = np.concatenate((output, neuron_output),
                                            axis=0)

            return (output,) + states

        self.neuron_step_math = neuron_step_math
        self.neuron_step_math.__name__ = utils.sanitize_name(
            "_".join([repr(op.neurons) for op in ops]))

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        states = [signals.gather(x) for x in self.state_data]
        states_dtype = [x.dtype for x in self.state_data]

        # note: we need to make sure that the previous call to this function
        # has completed before the next starts, since we don't know that the
        # functions are thread safe
        with tf.control_dependencies(self.prev_result), tf.device("/cpu:0"):
            ret = tf.py_func(
                self.neuron_step_math, [signals.dt, J] + states,
                [self.output_data.dtype] + states_dtype,
                name=self.neuron_step_math.__name__)
            neuron_out, state_out = ret[0], ret[1:]
        self.prev_result = [neuron_out]

        neuron_out.set_shape(
            self.output_data.shape + (signals.minibatch_size,))
        signals.scatter(self.output_data, neuron_out)

        for i, s in enumerate(self.state_data):
            state_out[i].set_shape(s.shape + (signals.minibatch_size,))
            signals.scatter(s, state_out[i])


class RectifiedLinearBuilder(OpBuilder):
    """Build a group of `~nengo.RectifiedLinear`
    neuron operators."""

    def __init__(self, ops, signals, config):
        super(RectifiedLinearBuilder, self).__init__(ops, signals, config)

        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        if all(op.neurons.amplitude == 1 for op in ops):
            self.amplitude = None
        else:
            self.amplitude = signals.op_constant(
                [op.neurons for op in ops], [op.J.shape[0] for op in ops],
                "amplitude", signals.dtype)

    def _step(self, J):
        out = tf.nn.relu(J)
        if self.amplitude is not None:
            out *= self.amplitude
        return out

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        out = self._step(J)
        signals.scatter(self.output_data, out)


class SpikingRectifiedLinearBuilder(RectifiedLinearBuilder):
    """Build a group of `~nengo.SpikingRectifiedLinear` neuron
       operators."""

    def __init__(self, ops, signals, config):
        super(SpikingRectifiedLinearBuilder, self).__init__(
            ops, signals, config)

        self.voltage_data = signals.combine([op.states[0] for op in ops])

        self.alpha = 1 if self.amplitude is None else self.amplitude
        self.alpha /= signals.dt

    def _step(self, J, voltage, dt):
        voltage += tf.nn.relu(J) * dt
        n_spikes = tf.floor(voltage)
        voltage -= n_spikes
        out = n_spikes * self.alpha

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return tf.stop_gradient(out), tf.stop_gradient(voltage)

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        voltage = signals.gather(self.voltage_data)

        spike_out, spike_voltage = self._step(J, voltage, signals.dt)

        if self.config.inference_only:
            out, voltage = spike_out, spike_voltage
        else:
            rate_out = super(SpikingRectifiedLinearBuilder, self)._step(J)

            out, voltage = tf.cond(
                signals.training,
                lambda: (rate_out, voltage),
                lambda: (spike_out, spike_voltage))

        signals.scatter(self.output_data, out)
        signals.scatter(self.voltage_data, voltage)


class SigmoidBuilder(OpBuilder):
    """Build a group of `~nengo.Sigmoid` neuron operators."""

    def __init__(self, ops, signals, config):
        super(SigmoidBuilder, self).__init__(ops, signals, config)

        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        self.tau_ref = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "tau_ref", signals.dtype)

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        signals.scatter(self.output_data, tf.nn.sigmoid(J) / self.tau_ref)


class LIFRateBuilder(OpBuilder):
    """Build a group of `~nengo.LIFRate` neuron operators."""

    def __init__(self, ops, signals, config):
        super(LIFRateBuilder, self).__init__(ops, signals, config)

        self.tau_ref = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "tau_ref", signals.dtype)
        self.tau_rc = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "tau_rc", signals.dtype)
        self.amplitude = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "amplitude", signals.dtype)

        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])
        self.zeros = tf.zeros(self.J_data.shape + (signals.minibatch_size,),
                              signals.dtype)

        self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

        # copy these so that they're easily accessible in the _step functions
        self.zero = signals.zero
        self.one = signals.one

    def _step(self, j):
        j -= self.one

        # note: we convert all the j to be positive before this calculation
        # (even though we'll only use the values that are already positive),
        # otherwise we can end up with nans in the gradient
        rates = self.amplitude / (
            self.tau_ref + self.tau_rc * tf.log1p(tf.reciprocal(
                tf.maximum(j, self.epsilon))))

        return tf.where(j > self.zero, rates, self.zeros)

    def build_step(self, signals):
        j = signals.gather(self.J_data)
        rates = self._step(j)
        signals.scatter(self.output_data, rates)


class SoftLIFRateBuilder(LIFRateBuilder):
    """Build a group of `.SoftLIFRate` neuron operators."""

    def __init__(self, ops, signals, config):
        super(SoftLIFRateBuilder, self).__init__(ops, signals, config)

        self.sigma = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "sigma", signals.dtype)

    def _step(self, J):
        J -= self.one

        js = J / self.sigma
        j_valid = js > -20
        js_safe = tf.where(j_valid, js, self.zeros)

        # softplus(js) = log(1 + e^js)
        z = tf.nn.softplus(js_safe) * self.sigma

        # as z->0
        #   z = s*log(1 + e^js) = s*e^js
        #   log(1 + 1/z) = log(1/z) = -log(s*e^js) = -js - log(s)
        q = tf.where(j_valid,
                     tf.log1p(tf.reciprocal(z)),
                     -js - tf.log(self.sigma))

        rates = self.amplitude / (self.tau_ref + self.tau_rc * q)

        return rates

    def build_step(self, signals):
        j = signals.gather(self.J_data)

        rates = self._step(j)

        signals.scatter(self.output_data, rates)


class LIFBuilder(SoftLIFRateBuilder):
    """Build a group of `~nengo.LIF` neuron operators."""

    def __init__(self, ops, signals, config):
        # note: we skip the SoftLIFRateBuilder init
        # pylint: disable=bad-super-call
        super(SoftLIFRateBuilder, self).__init__(ops, signals, config)

        self.min_voltage = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "min_voltage", signals.dtype)
        self.alpha = self.amplitude / signals.dt

        self.voltage_data = signals.combine([op.states[0] for op in ops])
        self.refractory_data = signals.combine([op.states[1] for op in ops])

        if self.config.lif_smoothing:
            self.sigma = tf.constant(self.config.lif_smoothing,
                                     dtype=signals.dtype)

    def _step(self, J, voltage, refractory, dt):
        delta_t = tf.clip_by_value(dt - refractory, self.zero, dt)

        dV = (voltage - J) * tf.expm1(-delta_t / self.tau_rc)
        voltage += dV

        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha

        partial_ref = -self.tau_rc * tf.log1p((self.one - voltage) /
                                              (J - self.one))
        # FastLIF version (linearly approximate spike time when calculating
        # remaining refractory period)
        # partial_ref = signals.dt * (voltage - self.one) / dV

        refractory = tf.where(spiked, self.tau_ref - partial_ref,
                              refractory - dt)

        voltage = tf.where(spiked, self.zeros,
                           tf.maximum(voltage, self.min_voltage))

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return (tf.stop_gradient(spikes), tf.stop_gradient(voltage),
                tf.stop_gradient(refractory))

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        voltage = signals.gather(self.voltage_data)
        refractory = signals.gather(self.refractory_data)

        spike_out, spike_voltage, spike_ref = self._step(
            J, voltage, refractory, signals.dt)

        if self.config.inference_only:
            spikes, voltage, refractory = spike_out, spike_voltage, spike_ref
        else:
            rate_out = (LIFRateBuilder._step(self, J)
                        if self.config.lif_smoothing is None else
                        SoftLIFRateBuilder._step(self, J))

            spikes, voltage, refractory = tf.cond(
                signals.training,
                lambda: (rate_out, voltage, refractory),
                lambda: (spike_out, spike_voltage, spike_ref)
            )

        signals.scatter(self.output_data, spikes)
        signals.mark_gather(self.J_data)
        signals.scatter(self.refractory_data, refractory)
        signals.scatter(self.voltage_data, voltage)


@Builder.register(SimNeurons)
class SimNeuronsBuilder(OpBuilder):
    """
    Builds a group of `~nengo.builder.neurons.SimNeurons` operators.

    Calls the appropriate sub-build class for the different neuron types.

    Attributes
    ----------
    TF_NEURON_IMPL : dict of {`~nengo.neurons.NeuronType`, \
                              `.builder.OpBuilder`}
        Mapping from neuron types to custom build classes (neurons without
        a custom builder will use the generic builder).
    """

    TF_NEURON_IMPL = {
        RectifiedLinear: RectifiedLinearBuilder,
        SpikingRectifiedLinear: SpikingRectifiedLinearBuilder,
        Sigmoid: SigmoidBuilder,
        LIF: LIFBuilder,
        LIFRate: LIFRateBuilder,
        SoftLIFRate: SoftLIFRateBuilder,
    }

    def __init__(self, ops, signals, config):
        super(SimNeuronsBuilder, self).__init__(ops, signals, config)

        logger.debug("J %s", [op.J for op in ops])

        neuron_type = type(ops[0].neurons)

        # if we have a custom tensorflow implementation for this neuron type,
        # then we build that. otherwise we'll just execute the neuron step
        # function externally (using `tf.py_func`).
        if neuron_type in self.TF_NEURON_IMPL:
            self.built_neurons = self.TF_NEURON_IMPL[neuron_type](
                ops, signals, config)
        else:
            self.built_neurons = GenericNeuronBuilder(ops, signals, config)

    def build_step(self, signals):
        self.built_neurons.build_step(signals)

    @staticmethod
    def mergeable(x, y):
        # neuron ops must all have the same type
        return type(x.neurons) == type(y.neurons)
