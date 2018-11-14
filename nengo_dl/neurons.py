"""
Additions to the `neuron types included with Nengo <nengo.neurons.NeuronType>`.
"""

from nengo.neurons import LIFRate
from nengo.params import NumberParam
import numpy as np


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

    sigma = NumberParam('sigma', low=0, low_open=True)

    def __init__(self, sigma=1.0, **lif_args):
        super(SoftLIFRate, self).__init__(**lif_args)
        self.sigma = sigma
        self._epsilon = 1e-15

    @property
    def _argreprs(self):
        args = super(SoftLIFRate, self)._argreprs
        if self.sigma != 1.0:
            args.append("sigma=%s" % self.sigma)
        return args

    def rates(self, x, gain, bias):
        J = gain * x
        J += bias
        out = np.zeros_like(J)
        self.step_math(dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""

        j = J - 1
        js = j / self.sigma
        j_valid = js > -20

        z = np.where(js > 30, js, np.log1p(np.exp(js))) * self.sigma

        q = np.where(j_valid, np.log1p(1 / z), -js - np.log(self.sigma))
        output[:] = self.amplitude / (self.tau_ref + self.tau_rc * q)
