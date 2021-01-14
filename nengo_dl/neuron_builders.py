"""
Build classes for Nengo neuron operators.
"""

import contextlib
import logging
import warnings

import numpy as np
import tensorflow as tf
from nengo.builder.neurons import SimNeurons
from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid, SpikingRectifiedLinear
from tensorflow.python.framework import smart_cond

from nengo_dl import compat, utils
from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.neurons import LeakyReLU, SoftLIFRate, SpikingLeakyReLU

logger = logging.getLogger(__name__)


class GenericNeuronBuilder(OpBuilder):
    """
    Builds all neuron types for which there is no custom TensorFlow
    implementation.

    Notes
    -----
    These will be executed as native Python functions, requiring execution to
    move in and out of TensorFlow.  This can significantly slow down the
    simulation, so any performance-critical neuron models should consider
    adding a custom TensorFlow implementation for their neuron type instead.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.J_data = signals.combine([op.J for op in self.ops])
        self.output_data = signals.combine([op.output for op in self.ops])

        state_keys = compat.neuron_state(self.ops[0]).keys()
        self.state_data = [
            signals.combine([compat.neuron_state(op)[key] for op in self.ops])
            for key in state_keys
        ]

        self.prev_result = []

        def neuron_step(dt, J, *states):  # pragma: no cover (runs in TF)
            output = None
            J_offset = 0
            state_offset = [0 for _ in states]
            for op in self.ops:
                # slice out the individual state vectors from the overall
                # array
                op_J = J[:, J_offset : J_offset + op.J.shape[0]]
                J_offset += op.J.shape[0]

                op_states = []
                for j, key in enumerate(state_keys):
                    s = compat.neuron_state(op)[key]
                    op_states += [
                        states[j][:, state_offset[j] : state_offset[j] + s.shape[0]]
                    ]
                    state_offset[j] += s.shape[0]

                # call neuron step function
                # note: `op_states` are views into `states`, which will
                # be updated in-place
                mini_out = []
                for j in range(signals.minibatch_size):
                    # blank output variable
                    neuron_output = np.zeros(op.output.shape, self.output_data.dtype)
                    compat.neuron_step(
                        op,
                        dt,
                        op_J[j],
                        neuron_output,
                        dict(zip(state_keys, [s[j] for s in op_states])),
                    )
                    mini_out.append(neuron_output)
                neuron_output = np.stack(mini_out, axis=0)

                # concatenate outputs
                if output is None:
                    output = neuron_output
                else:
                    output = np.concatenate((output, neuron_output), axis=1)

            return (output,) + states

        self.neuron_step = neuron_step
        self.neuron_step.__name__ = utils.sanitize_name(
            "_".join([repr(op.neurons) for op in self.ops])
        )

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        states = [signals.gather(x) for x in self.state_data]
        states_dtype = [x.dtype for x in self.state_data]

        if compat.eager_enabled():
            # noop
            control_deps = contextlib.suppress()
        else:
            # we need to make sure that the previous call to this function
            # has completed before the next starts, since we don't know that the
            # functions are thread safe
            control_deps = tf.control_dependencies(self.prev_result)

        with control_deps:
            ret = tf.numpy_function(
                self.neuron_step,
                [signals.dt, J] + states,
                [self.output_data.dtype] + states_dtype,
                name=self.neuron_step.__name__,
            )

        neuron_out, state_out = ret[0], ret[1:]

        self.prev_result = [neuron_out]

        neuron_out.set_shape((signals.minibatch_size,) + self.output_data.shape)
        signals.scatter(self.output_data, neuron_out)

        for i, s in enumerate(self.state_data):
            state_out[i].set_shape((signals.minibatch_size,) + s.shape)
            signals.scatter(s, state_out[i])


class TFNeuronBuilder(OpBuilder):
    """Base class for `~nengo.neurons.NeuronType` builders with a TF implementation."""

    # TODO: this can be delegated to op.neurons.spiking if we increase the minimum
    #  Nengo version to one where that attribute is guaranteed to exist
    spiking = False

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        if hasattr(self.ops[0].neurons, "amplitude"):
            if all(op.neurons.amplitude == 1 for op in self.ops):
                self.amplitude = None
            else:
                self.amplitude = signals.op_constant(
                    [op.neurons for op in self.ops],
                    [op.J.shape[0] for op in self.ops],
                    "amplitude",
                    signals.dtype,
                )
        else:
            self.amplitude = None
        self.alpha = 1 if self.amplitude is None else self.amplitude
        self.alpha /= signals.dt

        self.J_data = signals.combine([op.J for op in self.ops])
        self.output_data = signals.combine([op.output for op in self.ops])

        self.state_data = {
            state: signals.combine([compat.neuron_state(op)[state] for op in self.ops])
            for state in compat.neuron_state(self.ops[0])
        }

    def step(self, J, dt, **state):
        """
        Implements the logic for a single inference step.

        If the neuron has no states, returns only the neuron output. Otherwise, returns
        a tuple where the first element is the neuron output, and subsequent elements
        correspond to each of the neuron's states. The order of the states must be the
        same as the order they appear in the neuron type's ``state`` dictionary.
        """
        raise NotImplementedError("Subclasses must implement")

    def training_step(self, J, dt, **state):
        """
        Implements the logic for a single training step.

        Returns only the neuron output. Therefore, the ``training_step`` cannot affect
        any neuron states; they will all be held constant during training.

        Note: subclasses only need to implement this if ``spiking=True``. It is used
        to specify an alternate (differentiable) implementation of the neuron model
        to be used during training.
        """

    def build_step(self, signals, **step_kwargs):
        J = signals.gather(self.J_data)
        state = {s: signals.gather(d) for s, d in self.state_data.items()}

        step_output = self.step(J, signals.dt, **state)
        if isinstance(step_output, tuple) and len(step_output) != 1 + len(state):
            raise ValueError(
                f"`{type(self).__name__}.step` must return a tuple with the neuron "
                f"output followed by tensors with each updated state (one tensor per "
                f"state). `{type(self).__name__}` has {len(state)} states, but only "
                f"received output for {len(step_output) - 1} states."
            )

        step_output = tf.nest.flatten(step_output)

        if not self.spiking or self.config.inference_only:
            out = step_output
        else:
            out = tf.nest.flatten(
                smart_cond.smart_cond(
                    self.config.training,
                    true_fn=lambda: (self.training_step(J, signals.dt, **state),)
                    + tuple(state.values()),
                    # we use stop_gradient to avoid propagating any nans (those get
                    # propagated through the cond even if the spiking version isn't
                    # being used at all)
                    false_fn=lambda: tuple(
                        tf.stop_gradient(x) for x in tf.nest.flatten(step_output)
                    ),
                )
            )

        signals.scatter(self.output_data, out[0])
        for state_data, v in zip(self.state_data.values(), out[1:]):
            signals.scatter(state_data, v)


class RectifiedLinearBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.RectifiedLinear` neuron operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        if all(getattr(op.neurons, "negative_slope", 0) == 0 for op in self.ops):
            self.negative_slope = None
        else:
            self.negative_slope = signals.op_constant(
                [op.neurons for op in self.ops],
                [op.J.shape[0] for op in self.ops],
                "negative_slope",
                signals.dtype,
            )

    def step(self, J, dt):
        out = tf.nn.relu(J)
        if self.negative_slope is not None:
            out -= self.negative_slope * tf.nn.relu(-J)
        if self.amplitude is not None:
            out *= self.amplitude
        return out


class SpikingRectifiedLinearBuilder(RectifiedLinearBuilder):
    """Build a group of `~nengo.SpikingRectifiedLinear` neuron operators."""

    spiking = True

    def step(self, J, dt, voltage):
        if self.negative_slope is None:
            voltage += tf.nn.relu(J) * dt
            n_spikes = tf.floor(voltage)
        else:
            voltage += (tf.nn.relu(J) - self.negative_slope * tf.nn.relu(-J)) * dt
            n_spikes = tf.floor(voltage) + tf.cast(voltage < 0, voltage.dtype)

        voltage -= n_spikes
        out = n_spikes * self.alpha

        return out, voltage

    def training_step(self, J, dt, **state):
        return super().step(J, dt)


class SigmoidBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.Sigmoid` neuron operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.tau_ref = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "tau_ref",
            signals.dtype,
        )

    def step(self, J, dt):
        return tf.nn.sigmoid(J) / self.tau_ref


class TanhBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.Tanh` neuron operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.tau_ref = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "tau_ref",
            signals.dtype,
        )

    def step(self, J, dt):
        return tf.nn.tanh(J) / self.tau_ref


class LIFRateBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.LIFRate` neuron operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.tau_ref = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "tau_ref",
            signals.dtype,
        )
        self.tau_rc = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "tau_rc",
            signals.dtype,
        )

        self.zeros = tf.zeros(
            (signals.minibatch_size,) + self.J_data.shape, signals.dtype
        )

        self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

        # copy these so that they're easily accessible in the step functions
        self.zero = signals.zero
        self.one = signals.one

    def step(self, J, dt):
        J -= self.one

        # note: we convert all the j to be positive before this calculation
        # (even though we'll only use the values that are already positive),
        # otherwise we can end up with nans in the gradient
        rates = (self.one if self.amplitude is None else self.amplitude) / (
            self.tau_ref
            + self.tau_rc
            * tf.math.log1p(tf.math.reciprocal(tf.maximum(J, self.epsilon)))
        )

        return tf.where(J > self.zero, rates, self.zeros)


class SoftLIFRateBuilder(LIFRateBuilder):
    """Build a group of `.SoftLIFRate` neuron operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.sigma = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "sigma",
            signals.dtype,
        )

    def step(self, J, dt):
        J -= self.one

        js = J / self.sigma
        j_valid = js > -20
        js_safe = tf.where(j_valid, js, self.zeros)

        # softplus(js) = log(1 + e^js)
        z = tf.nn.softplus(js_safe) * self.sigma

        # as z->0
        #   z = s*log(1 + e^js) = s*e^js
        #   log(1 + 1/z) = log(1/z) = -log(s*e^js) = -js - log(s)
        q = tf.where(
            j_valid, tf.math.log1p(tf.math.reciprocal(z)), -js - tf.math.log(self.sigma)
        )

        rates = (self.one if self.amplitude is None else self.amplitude) / (
            self.tau_ref + self.tau_rc * q
        )

        return rates


class LIFBuilder(SoftLIFRateBuilder):
    """Build a group of `~nengo.LIF` neuron operators."""

    spiking = True

    def build_pre(self, signals, config):
        # note: we skip the SoftLIFRateBuilder init
        # pylint: disable=bad-super-call
        super(SoftLIFRateBuilder, self).build_pre(signals, config)

        self.min_voltage = signals.op_constant(
            [op.neurons for op in self.ops],
            [op.J.shape[0] for op in self.ops],
            "min_voltage",
            signals.dtype,
        )

        if self.config.lif_smoothing:
            self.sigma = tf.constant(self.config.lif_smoothing, dtype=signals.dtype)

    def step(self, J, dt, voltage, refractory_time):
        delta_t = tf.clip_by_value(dt - refractory_time, self.zero, dt)

        dV = (voltage - J) * tf.math.expm1(
            -delta_t / self.tau_rc  # pylint: disable=invalid-unary-operand-type
        )
        voltage += dV

        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha

        partial_ref = -self.tau_rc * tf.math.log1p(
            (self.one - voltage) / (J - self.one)
        )
        # FastLIF version (linearly approximate spike time when calculating
        # remaining refractory period)
        # partial_ref = signals.dt * (voltage - self.one) / dV

        refractory_time = tf.where(
            spiked, self.tau_ref - partial_ref, refractory_time - dt
        )

        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, self.min_voltage))

        return spikes, voltage, refractory_time

    def training_step(self, J, dt, **state):
        return (
            LIFRateBuilder.step(self, J, dt)
            if self.config.lif_smoothing is None
            else SoftLIFRateBuilder.step(self, J, dt)
        )


class RegularSpikingBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.RegularSpiking` neuron operators."""

    spiking = True

    def step(self, J, dt, voltage):
        voltage += J * dt
        n_spikes = tf.floor(voltage)

        voltage -= n_spikes
        out = n_spikes * self.alpha

        return out, voltage

    def training_step(self, J, dt, **state):
        return J if self.amplitude is None else J * self.amplitude


class StochasticSpikingBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.StochasticSpiking` neuron operators."""

    spiking = True

    def step(self, J, dt):
        x = dt * tf.math.abs(J)
        n_spikes = tf.floor(x)
        frac = x - n_spikes

        n_spikes += tf.cast(
            tf.random.uniform(frac.shape, dtype=frac.dtype) < frac, n_spikes.dtype
        )

        n_spikes *= self.alpha * tf.math.sign(J)

        return n_spikes

    def training_step(self, J, dt):
        return J if self.amplitude is None else J * self.amplitude


class PoissonSpikingBuilder(TFNeuronBuilder):
    """Build a group of `~nengo.PoissonSpiking` neuron operators."""

    spiking = True

    def step(self, J, dt):
        n_spikes = (
            self.alpha
            * tf.random.poisson((), tf.math.abs(J) * dt, dtype=J.dtype)
            * tf.math.sign(J)
        )
        n_spikes.set_shape(J.shape)

        return n_spikes

    def training_step(self, J, dt):
        return J if self.amplitude is None else J * self.amplitude


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
        LeakyReLU: RectifiedLinearBuilder,
        SpikingLeakyReLU: SpikingRectifiedLinearBuilder,
        Sigmoid: SigmoidBuilder,
        compat.Tanh: TanhBuilder,
        LIF: LIFBuilder,
        LIFRate: LIFRateBuilder,
        SoftLIFRate: SoftLIFRateBuilder,
        compat.RegularSpiking: RegularSpikingBuilder,
        compat.StochasticSpiking: StochasticSpikingBuilder,
        compat.PoissonSpiking: PoissonSpikingBuilder,
    }

    def __init__(self, ops):
        super().__init__(ops)

        neuron_type = type(ops[0].neurons)

        # if we have a custom tensorflow implementation for this neuron type,
        # then we build that. otherwise we'll just execute the neuron step
        # function externally (using `tf.py_func`).
        if neuron_type in self.TF_NEURON_IMPL:
            self.built_neurons = self.TF_NEURON_IMPL[neuron_type](ops)
        else:
            warnings.warn(
                f"{neuron_type} does not have a native TensorFlow implementation; "
                "falling back to Python implementation"
            )
            self.built_neurons = GenericNeuronBuilder(ops)

    def build_pre(self, signals, config):
        self.built_neurons.build_pre(signals, config)

    def build_step(self, signals):
        self.built_neurons.build_step(signals)

    @staticmethod
    def mergeable(x, y):
        # neuron ops must all have the same type
        return type(x.neurons) == type(y.neurons)  # noqa: E721
