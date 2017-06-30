from __future__ import division

from nengo.dists import Distribution
from nengo.params import NumberParam, EnumParam
import numpy as np


class TruncatedNormal(Distribution):
    """Normal distribution where any values more than some distance from the
    mean are resampled.

    Parameters
    ----------
    mean : float, optional
        mean of the normal distribution
    stddev : float, optional
        standard deviation of the normal distribution
    limit : float, optional
        resample any values more than this distance from the mean. if None,
        then limit will be set to 2 standard deviations
    """

    mean = NumberParam("mean")
    stddev = NumberParam("stddev", low=0)
    limit = NumberParam("limit", low=0, low_open=True)

    def __init__(self, mean=0, stddev=1, limit=None):
        self.mean = mean
        self.stddev = stddev
        self.limit = 2 * stddev if limit is None else limit

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

        sample_shape = (n,) if d is None else (n, d)
        samples = rng.normal(loc=self.mean, scale=self.stddev,
                             size=sample_shape)
        outliers = np.abs(samples - self.mean) > self.limit
        n_out = np.sum(outliers)
        while n_out > 0:
            samples[outliers] = rng.normal(self.mean, self.stddev,
                                           size=n_out)
            outliers = np.abs(samples - self.mean) > self.limit
            n_out = np.sum(outliers)

        return samples


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
            return TruncatedNormal(stddev=stddev).sample(n, d)


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
