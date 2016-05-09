import nengo
import numpy as np


def default_config():
    config = nengo.Config(nengo.Ensemble, nengo.Connection)
    config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

    config[nengo.Connection].synapse = None

    return config


def to_array(x, shape):
    if isinstance(x, nengo.dists.Distribution):
        x = x.sample(*shape)

    assert x.shape == shape
    return np.asarray(x, dtype=np.float32)
