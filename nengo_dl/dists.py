from __future__ import division

from nengo.dists import Distribution
from nengo.params import NumberParam, EnumParam
import numpy as np


class VarianceScaling(Distribution):
    """Variance scaling distribution for weight initialization (analogous to
    TensorFlow ``init_ops.VarianceScaling`).

    Parameters
    ----------
    scale : float, optional
        overall scale on values
    mode : "fan_in" or "fan_out" or "fan_avg", optional
        whether to scale based on input or output dimensionality, or average of
        the two
    distribution: "uniform" or "normal", optional
        whether to use a uniform or normal distribution for weights
    """

    scale = NumberParam("scale", low=0)
    mode = EnumParam("mode", values=["fan_in", "fan_out", "fan_avg"])
    distribution = EnumParam("distribution", values=["uniform", "normal"])

    def __init__(self, scale=1, mode="fan_avg", distribution="uniform"):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def sample(self, n, d=None, rng=np.random):
        """Samples the distribution.

        Parameters
        ----------
        n : int
            Number samples to take.
        d : int or None, optional
            The number of dimensions to return. If this is an int, the return
            value will be of shape ``(n, d)``. If None, the return
            value will be of shape ``(n,)``.
        rng : `numpy.random.RandomState`, optional
            Random number generator state.

        Returns
        -------
        samples : (n,) or (n, d) array_like
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """

        fan_out = n
        fan_in = 1 if d is None else d
        scale = self.scale
        if self.mode == "fan_in":
            scale /= fan_in
        elif self.mode == "fan_out":
            scale /= fan_out
        elif self.mode == "fan_avg":
            scale /= (fan_in + fan_out) / 2

        shape = (n,) if d is None else (n, d)
        if self.distribution == "uniform":
            limit = np.sqrt(3.0 * scale)
            return rng.uniform(-limit, limit, size=shape)
        elif self.distribution == "normal":
            stddev = np.sqrt(scale)
            # TODO: use truncated normal distribution
            return rng.normal(scale=stddev, size=shape)


class Glorot(VarianceScaling):
    """Weight initialization method from Glorot and Bengio (2010).

    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Parameters
    ----------
    scale : float, optional
        scale on weight distribution. for rectified linear units this should
        be sqrt(2), otherwise usually 1
    distribution: "uniform" or "normal", optional
        whether to use a uniform or normal distribution for weights
    """
    def __init__(self, scale=1, distribution="uniform"):
        super(Glorot, self).__init__(scale=scale, mode="fan_avg",
                                     distribution=distribution)


class He(VarianceScaling):
    """Weight initialization method from He et al. (2015).

    https://arxiv.org/abs/1502.01852

    Parameters
    ----------
    scale : float, optional
        scale on weight distribution. for rectified linear units this should
        be sqrt(2), otherwise usually 1
    distribution: "uniform" or "normal", optional
        whether to use a uniform or normal distribution for weights
    """

    def __init__(self, scale=1, distribution="normal"):
        super(He, self).__init__(scale=scale ** 2, mode="fan_in",
                                 distribution=distribution)
