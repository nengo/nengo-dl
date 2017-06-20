from __future__ import division

import itertools

import numpy as np
import pytest

from nengo_dl import dists


def _test_variance_scaling(dist, scale, mode):
    shape = (1000, 500)

    if mode == "fan_in":
        scale /= shape[0]
    elif mode == "fan_out":
        scale /= shape[1]
    else:
        scale /= np.mean(shape)

    if mode == "uniform":
        scale *= 3

    scale = np.sqrt(scale)

    samples = dist.sample(shape[0], shape[1])

    assert np.allclose(np.mean(samples), 0.0, atol=5e-4)
    assert np.allclose(np.std(samples), scale, atol=5e-4)


@pytest.mark.parametrize(
    "scale, mode, distribution",
    itertools.product([1, 2], ["fan_in", "fan_out", "fan_avg"],
                      ["uniform", "normal"]))
def test_variance_scaling(scale, mode, distribution):
    dist = dists.VarianceScaling(scale=scale, mode=mode,
                                 distribution=distribution)
    _test_variance_scaling(dist, scale, mode)


@pytest.mark.parametrize(
    "scale, distribution",
    itertools.product([1, 2], ["uniform", "normal"]))
def test_glorot(scale, distribution):
    dist = dists.Glorot(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale, "fan_avg")


@pytest.mark.parametrize(
    "scale, distribution",
    itertools.product([1, 2], ["uniform", "normal"]))
def test_he(scale, distribution):
    dist = dists.He(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale ** 2, "fan_in")
