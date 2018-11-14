# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import config, dists, SoftLIFRate


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

    with net:
        config.configure_settings(dtype=tf.float64)

    with Simulator(net) as sim2:
        _, nengo_dl_curves = nengo.utils.ensemble.tuning_curves(ens, sim2)
        sim2.run_steps(30)

    assert np.allclose(nengo_curves, nengo_dl_curves)
    assert np.allclose(sim.data[p], sim2.data[p])


@pytest.mark.parametrize(
    "neuron_type", (nengo.LIFRate, nengo.RectifiedLinear, SoftLIFRate))
@pytest.mark.training
def test_neuron_gradients(Simulator, neuron_type, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node(output=[0])
        b = nengo.Ensemble(50, 1, neuron_type=neuron_type())
        c = nengo.Ensemble(50, 1, neuron_type=neuron_type(amplitude=0.1))
        nengo.Connection(a, b, synapse=None)
        nengo.Connection(b, c, synapse=None)
        nengo.Probe(c)

    with Simulator(net, seed=seed) as sim:
        sim.check_gradients()


@pytest.mark.parametrize("rate, spiking", [
    (nengo.RectifiedLinear, nengo.SpikingRectifiedLinear),
    (nengo.LIFRate, nengo.LIF),
    (SoftLIFRate, nengo.LIF)])
def test_spiking_swap(Simulator, rate, spiking, seed):
    grads = []
    for neuron_type in [rate, spiking]:
        with nengo.Network(seed=seed) as net:
            config.configure_settings(dtype=tf.float64)

            if rate == SoftLIFRate and neuron_type == spiking:
                config.configure_settings(lif_smoothing=1.0)

            a = nengo.Node(output=[1])
            b = nengo.Ensemble(50, 1, neuron_type=neuron_type())
            c = nengo.Ensemble(50, 1, neuron_type=neuron_type(amplitude=0.1))
            nengo.Connection(a, b, synapse=None)

            # note: we avoid decoders, as the rate/spiking models may have
            # different rate implementations in nengo, resulting in different
            # decoders
            nengo.Connection(b.neurons, c.neurons, synapse=None,
                             transform=dists.He())
            p = nengo.Probe(c.neurons)

        with Simulator(net) as sim:
            grads.append(sim.sess.run(
                tf.gradients(sim.tensor_graph.probe_arrays[p],
                             tf.trainable_variables()),
                feed_dict=sim._fill_feed(10, training=True)))

            sim.soft_reset()
            sim.run(0.5)

        # check that the normal output is unaffected by the swap logic
        with nengo.Simulator(net) as sim2:
            sim2.run(0.5)

            assert np.allclose(sim.data[p], sim2.data[p])

    # check that the gradients match
    assert all(np.allclose(g0, g1) for g0, g1 in zip(*grads))
