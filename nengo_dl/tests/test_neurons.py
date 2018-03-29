import nengo
from nengo.builder.neurons import SimNeurons
from nengo.builder.signal import Signal
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import SoftLIFRate, neuron_builders


def test_lif_deterministic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(
            100, 1, noise=nengo.processes.WhiteNoise(seed=seed))
        p = nengo.Probe(ens.neurons)

    with nengo.Simulator(net) as sim:
        sim.run_steps(50)

    canonical = sim.data[p]

    for _ in range(5):
        with Simulator(net) as sim:
            sim.run_steps(50)

        assert np.allclose(sim.data[p], canonical)


@pytest.mark.parametrize("sigma", (1, 0.5))
def test_soft_lif(Simulator, sigma, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0.5])
        ens = nengo.Ensemble(10, 1, neuron_type=SoftLIFRate(sigma=sigma),
                             intercepts=nengo.dists.Uniform(-1, 0),
                             encoders=nengo.dists.Choice([[1]]))
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens.neurons)

    x = str(ens.neuron_type)
    if sigma == 1:
        assert "sigma" not in x
    else:
        assert "sigma=%s" % sigma in x

    with nengo.Simulator(net) as sim:
        _, nengo_curves = nengo.utils.ensemble.tuning_curves(ens, sim)
        sim.run_steps(30)

    with Simulator(net, dtype=tf.float64) as sim2:
        _, nengo_dl_curves = nengo.utils.ensemble.tuning_curves(ens, sim2)
        sim2.run_steps(30)

    assert np.allclose(nengo_curves, nengo_dl_curves)
    assert np.allclose(sim.data[p], sim2.data[p])


@pytest.mark.parametrize(
    "neuron_type", (nengo.LIFRate, SoftLIFRate))
def test_neuron_gradients(Simulator, neuron_type, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node(output=[0])
        b = nengo.Ensemble(50, 1, neuron_type=neuron_type())
        nengo.Connection(a, b, synapse=None)
        nengo.Probe(b)

    with Simulator(net, seed=seed) as sim:
        sim.check_gradients()


@pytest.mark.parametrize("dtype", (tf.float32, tf.float64))
@pytest.mark.parametrize("diff", (True, False))
def test_get_constant(dtype, diff):
    ops = (SimNeurons(nengo.LIF(tau_rc=1), Signal(np.zeros(10)), None),
           SimNeurons(nengo.LIF(tau_rc=2 if diff else 1), Signal(np.zeros(10)),
                      None))

    const = neuron_builders.get_constant(ops, "tau_rc", dtype)

    assert const.dtype == dtype

    with tf.Session() as sess:
        x = sess.run(const)

    if diff:
        assert np.array_equal(x, [[1]] * 10 + [[2]] * 10)
    else:
        assert np.array_equal(x, 1.0)
