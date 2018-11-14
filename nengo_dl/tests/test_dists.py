# pylint: disable=missing-docstring

from __future__ import division

import math

import numpy as np
import pytest

from nengo_dl import dists


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def norm_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)


def tnorm_var(scale, limit):
    # note: this assumes a mean of 0
    a = -limit / scale
    b = limit / scale

    pdf_a = norm_pdf(a)
    pdf_b = norm_pdf(b)
    z = norm_cdf(b) - norm_cdf(a)

    return scale ** 2 * (1 + (a * pdf_a - b * pdf_b) / z -
                         ((pdf_a - pdf_b) / z) ** 2)


def _test_variance_scaling(dist, scale, mode, seed):
    shape = (1000, 2000)
    rng = np.random.RandomState(seed)

    if mode == "fan_in":
        scale /= shape[1]
    elif mode == "fan_out":
        scale /= shape[0]
    else:
        scale /= np.mean(shape)

    if dist.distribution == "uniform":
        scale *= 3

    std = np.sqrt(scale)

    samples = dist.sample(shape[0], shape[1], rng=rng)

    assert samples.shape == shape
    assert np.allclose(np.mean(samples), 0.0, atol=5e-4)
    if dist.distribution == "uniform":
        var = 4 * std ** 2 / 12
        assert np.allclose(np.var(samples), var, rtol=5e-3)
    else:
        assert np.allclose(np.var(samples), tnorm_var(std, 2 * std),
                           rtol=5e-3)

    # test with default rng
    samples = dist.sample(shape[0], shape[1])
    assert samples.shape == shape


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out", "fan_avg"])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_variance_scaling(scale, mode, distribution, seed):
    dist = dists.VarianceScaling(scale=scale, mode=mode,
                                 distribution=distribution)
    _test_variance_scaling(dist, scale, mode, seed)


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_glorot(scale, distribution, seed):
    dist = dists.Glorot(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale, "fan_avg", seed)


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_he(scale, distribution, seed):
    dist = dists.He(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale ** 2, "fan_in", seed)


@pytest.mark.parametrize("limit", [None, 0.5])
@pytest.mark.parametrize("stddev", [1, 0.2])
def test_truncated_normal(limit, stddev, seed):
    rng = np.random.RandomState(seed)
    dist = dists.TruncatedNormal(mean=0, stddev=stddev, limit=limit)
    if limit is None:
        limit = 2 * stddev
    samples = dist.sample(1000, 2000, rng=rng)
    assert samples.shape == (1000, 2000)
    assert np.allclose(np.mean(samples), 0.0, atol=5e-3)
    assert np.allclose(np.var(samples), tnorm_var(stddev, limit), rtol=5e-3)
    assert np.all(samples < limit)
    assert np.all(samples > -limit)

    # test with default rng
    samples = dist.sample(1000, 2000)
    assert samples.shape == (1000, 2000)


@pytest.mark.parametrize("dist", [
    dists.TruncatedNormal(), dists.VarianceScaling(), dists.Glorot(),
    dists.He()])
def test_seeding(dist, seed):
    assert np.allclose(dist.sample(100, rng=np.random.RandomState(seed)),
                       dist.sample(100, rng=np.random.RandomState(seed)))
