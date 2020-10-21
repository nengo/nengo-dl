"""
Additions to the `neuron types included with Nengo <nengo.neurons.NeuronType>`.
"""

import numpy as np
from nengo.neurons import LIFRate, RectifiedLinear, SpikingRectifiedLinear
from nengo.params import NumberParam


class SoftLIFRate(LIFRate):
    """LIF neuron with smoothing around the firing threshold.

    This is a rate version of the LIF neuron whose tuning curve has a
    continuous first derivative, due to the smoothing around the firing
    threshold. It can be used as a substitute for LIF neurons in deep networks
    during training, and then replaced with LIF neurons when running
    the network [1]_.

    Parameters
    ----------
    sigma : float
        Amount of smoothing around the firing threshold. Larger values mean
        more smoothing.
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.

    References
    ----------
    .. [1] Eric Hunsberger and Chris Eliasmith (2015): Spiking deep networks
       with LIF neurons. https://arxiv.org/abs/1510.08829.

    Notes
    -----
    Adapted from
    https://github.com/nengo/nengo-extras/blob/master/nengo_extras/neurons.py
    """

    sigma = NumberParam("sigma", low=0, low_open=True)

    def __init__(self, sigma=1.0, **lif_args):
        super().__init__(**lif_args)
        self.sigma = sigma
        self._epsilon = 1e-15

    @property
    def _argreprs(self):
        args = super()._argreprs
        if self.sigma != 1.0:
            args.append(f"sigma={self.sigma}")
        return args

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""

        J = gain * x
        J += bias
        out = np.zeros_like(J)
        self.step(dt=1, J=J, output=out)
        return out

    def step(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""

        j = J - 1
        js = j / self.sigma
        j_valid = js > -20

        z = np.where(js > 30, js, np.log1p(np.exp(js))) * self.sigma

        q = np.where(j_valid, np.log1p(1 / z), -js - np.log(self.sigma))
        output[:] = self.amplitude / (self.tau_ref + self.tau_rc * q)

    # note: need to specify these (even though they're defined in the base class)
    # so that this works with Nengo<3.1.0 (where these attributes won't be defined)
    # TODO: remove if we increase the minimum nengo version
    negative = False
    spiking = False

    def step_math(self, dt, J, output):
        """Backwards compatibility alias for ``self.step``."""
        return self.step(dt, J, output)


class LeakyReLU(RectifiedLinear):
    """
    Rectified linear neuron with nonzero slope for values < 0.

    Parameters
    ----------
    negative_slope : float
        Scaling factor applied to values less than zero.
    amplitude : float
        Scaling factor on the neuron output. Note that this will combine
        multiplicatively with ``negative_slope`` for values < 0.
    """

    def __init__(self, negative_slope=0.3, amplitude=1, **kwargs):
        super().__init__(amplitude=amplitude, **kwargs)

        self.negative_slope = negative_slope

    def step(self, dt, J, output):
        """Implement the leaky relu nonlinearity."""

        output[...] = self.amplitude * np.where(J < 0, self.negative_slope * J, J)

    # note: need to specify these (even though they're defined in the base class)
    # so that this works with Nengo<3.1.0 (where these attributes won't be defined)
    # TODO: remove if we increase the minimum nengo version
    negative = False
    spiking = False

    def step_math(self, dt, J, output):
        """Backwards compatibility alias for ``self.step``."""
        return self.step(dt, J, output)


class SpikingLeakyReLU(SpikingRectifiedLinear):
    """
    Spiking version of `.LeakyReLU`.

    Note that this may output "negative spikes" (i.e. a spike with a sign of -1).

    Parameters
    ----------
    negative_slope : float
        Scaling factor applied to values less than zero.
    amplitude : float
        Scaling factor on the neuron output. Note that this will combine
        multiplicatively with ``negative_slope`` for values < 0.
    """

    def __init__(self, negative_slope=0.3, amplitude=1, **kwargs):
        super().__init__(amplitude=amplitude, **kwargs)

        self.negative_slope = negative_slope

    def rates(self, x, gain, bias):
        """Use `.LeakyReLU` to determine rates."""

        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        LeakyReLU.step(self, dt=1, J=J, output=out)
        return out

    def step(self, dt, J, output, voltage):
        """
        Implement the spiking leaky relu nonlinearity.
        """

        voltage += np.where(J < 0, self.negative_slope * J, J) * dt
        n_spikes = np.trunc(voltage)
        output[:] = (self.amplitude / dt) * n_spikes
        voltage -= n_spikes

    # note: need to specify these (even though they're defined in the base class)
    # so that this works with Nengo<3.1.0 (where these attributes won't be defined)
    # TODO: remove if we increase the minimum nengo version
    negative = False
    spiking = True

    def step_math(self, dt, J, output, voltage):
        """Backwards compatibility alias for ``self.step``."""
        return self.step(dt, J, output, voltage)
