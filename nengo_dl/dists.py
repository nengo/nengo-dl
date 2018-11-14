"""
Additions to the `distributions included with Nengo
<nengo.dists.Distribution>`.
These distributions are usually used to initialize weight matrices, e.g.
``nengo.Connection(a.neurons, b.neurons, transform=nengo_dl.dists.Glorot())``.
"""

from __future__ import division

from nengo.dists import Distribution
from nengo.params import NumberParam, EnumParam
import numpy as np


class TruncatedNormal(Distribution):
    """Normal distribution where any values more than some distance from the
    mean are resampled.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution
    stddev : float
        Standard deviation of the normal distribution
    limit : float
        Resample any values more than this distance from the mean. If None,
        then limit will be set to 2 standard deviations.
    """

    mean = NumberParam("mean")
    stddev = NumberParam("stddev", low=0)
    limit = NumberParam("limit", low=0, low_open=True)

    def __init__(self, mean=0, stddev=1, limit=None):
        super(TruncatedNormal, self).__init__()

        self.mean = mean
        self.stddev = stddev
        self.limit = 2 * stddev if limit is None else limit

    def sample(self, n, d=None, rng=None):
        """Samples the distribution.

        Parameters
        ----------
        n : int
            Number samples to take.
        d : int or None
            The number of dimensions to return. If this is an int, the return
            value will be of shape ``(n, d)``. If None, the return
            value will be of shape ``(n,)``.
        rng : `numpy.random.RandomState`
            Random number generator state (if None, will use the default
            numpy random number generator).

        Returns
        -------
        samples : (n,) or (n, d) array_like
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """

        if rng is None:
            rng = np.random
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
    TensorFlow ``init_ops.VarianceScaling``).

    Parameters
    ----------
    scale : float
        Overall scale on values
    mode : "fan_in" or "fan_out" or "fan_avg"
        Whether to scale based on input or output dimensionality, or average of
        the two
    distribution: "uniform" or "normal"
        Whether to use a uniform or normal distribution for weights
    """

    scale = NumberParam("scale", low=0)
    mode = EnumParam("mode", values=["fan_in", "fan_out", "fan_avg"])
    distribution = EnumParam("distribution", values=["uniform", "normal"])

    def __init__(self, scale=1, mode="fan_avg", distribution="uniform"):
        super(VarianceScaling, self).__init__()

        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def sample(self, n, d=None, rng=None):
        """Samples the distribution.

        Parameters
        ----------
        n : int
            Number samples to take.
        d : int or None
            The number of dimensions to return. If this is an int, the return
            value will be of shape ``(n, d)``. If None, the return
            value will be of shape ``(n,)``.
        rng : `numpy.random.RandomState`
            Random number generator state (if None, will use the default
            numpy random number generator).

        Returns
        -------
        samples : (n,) or (n, d) array_like
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """

        if rng is None:
            rng = np.random
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
            return TruncatedNormal(stddev=stddev).sample(n, d, rng=rng)
        else:
            # note: this should be caught by the enumparam check
            raise NotImplementedError


class Glorot(VarianceScaling):
    """Weight initialization method from [1]_ (also known as Xavier
    initialization).

    Parameters
    ----------
    scale : float
        Scale on weight distribution. For rectified linear units this should
        be sqrt(2), otherwise usually 1.
    distribution: "uniform" or "normal"
        Whether to use a uniform or normal distribution for weights

    References
    ----------
    .. [1] Xavier Glorot and Yoshua Bengio (2010): Understanding the difficulty
       of training deep feedforward neural networks. International conference
       on artificial intelligence and statistics.
       http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf.
    """

    def __init__(self, scale=1, distribution="uniform"):
        super(Glorot, self).__init__(scale=scale, mode="fan_avg",
                                     distribution=distribution)


class He(VarianceScaling):
    """Weight initialization method from [1]_.

    Parameters
    ----------
    scale : float
        Scale on weight distribution. For rectified linear units this should
        be sqrt(2), otherwise usually 1.
    distribution: "uniform" or "normal"
        Whether to use a uniform or normal distribution for weights

    References
    ----------
    .. [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. (2015):
       Delving deep into rectifiers: Surpassing human-level performance on
       ImageNet classification. https://arxiv.org/abs/1502.01852.
    """

    def __init__(self, scale=1, distribution="normal"):
        super(He, self).__init__(scale=scale ** 2, mode="fan_in",
                                 distribution=distribution)
