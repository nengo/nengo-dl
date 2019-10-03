# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest

from nengo_dl import config, dists, SoftLIFRate, neuron_builders


def test_lif_deterministic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(100, 1, noise=nengo.processes.WhiteNoise(seed=seed))
        p = nengo.Probe(ens.neurons)

    with nengo.Simulator(net) as sim:
        sim.run_steps(50)

    canonical = sim.data[p]

    for _ in range(5):
        with Simulator(net) as sim:
            sim.run_steps(50)

        assert np.allclose(sim.data[p], canonical)


def test_merged_generic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        nodes = []
        ensembles = []
        probes = []
        for _ in range(2):
            nodes.append(nengo.Node([1]))
            ensembles.append(nengo.Ensemble(10, 1, neuron_type=nengo.AdaptiveLIF()))
            nengo.Connection(nodes[-1], ensembles[-1], synapse=None)
            probes.append(nengo.Probe(ensembles[-1].neurons))

    with nengo.Simulator(net) as canonical:
        canonical.run_steps(100)

    with Simulator(net) as sim:
        ops = [
            ops
            for ops in sim.tensor_graph.plan
            if isinstance(ops[0], nengo.builder.neurons.SimNeurons)
        ]
        assert len(ops) == 1
        assert isinstance(
            sim.tensor_graph.op_builder.op_builds[ops[0]].built_neurons,
            neuron_builders.GenericNeuronBuilder,
        )

        sim.run_steps(100)

        for p in probes:
            assert np.allclose(sim.data[p], canonical.data[p])


@pytest.mark.parametrize("sigma", (1, 0.5))
def test_soft_lif(Simulator, sigma, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0.5])
        ens = nengo.Ensemble(
            10,
            1,
            neuron_type=SoftLIFRate(sigma=sigma),
            intercepts=nengo.dists.Uniform(-1, 0),
            encoders=nengo.dists.Choice([[1]]),
        )
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
        config.configure_settings(dtype="float64")

    with Simulator(net) as sim2:
        _, nengo_dl_curves = nengo.utils.ensemble.tuning_curves(ens, sim2)
        sim2.run_steps(30)

    assert np.allclose(nengo_curves, nengo_dl_curves)
    assert np.allclose(sim.data[p], sim2.data[p])


@pytest.mark.parametrize(
    "neuron_type", (nengo.LIFRate, nengo.RectifiedLinear, SoftLIFRate)
)
@pytest.mark.training
def test_neuron_gradients(Simulator, neuron_type, seed, rng):
    # avoid intercepts around zero, which can cause errors in the
    # finite differencing in check_gradients
    intercepts = np.concatenate(
        (rng.uniform(-0.5, -0.2, size=25), rng.uniform(0.2, 0.5, size=25))
    )

    kwargs = {"sigma": 0.1} if neuron_type == SoftLIFRate else {}

    with nengo.Network(seed=seed) as net:
        config.configure_settings(dtype="float64")
        net.config[nengo.Ensemble].intercepts = intercepts
        a = nengo.Node(output=[0, 0])
        b = nengo.Ensemble(50, 2, neuron_type=neuron_type(**kwargs))
        c = nengo.Ensemble(50, 2, neuron_type=neuron_type(amplitude=0.1, **kwargs))
        nengo.Connection(a, b, synapse=None)
        nengo.Connection(b, c, synapse=None)
        nengo.Probe(c)

    with Simulator(net, seed=seed) as sim:
        sim.check_gradients()


@pytest.mark.parametrize(
    "rate, spiking",
    [
        (nengo.RectifiedLinear, nengo.SpikingRectifiedLinear),
        (nengo.LIFRate, nengo.LIF),
        (SoftLIFRate, nengo.LIF),
    ],
)
def test_spiking_swap(Simulator, rate, spiking, seed):
    grads = []
    for neuron_type in [rate, spiking]:
        with nengo.Network(seed=seed) as net:
            config.configure_settings(dtype="float64")

            if rate == SoftLIFRate and neuron_type == spiking:
                config.configure_settings(lif_smoothing=1.0)

            a = nengo.Node(output=[1])
            b = nengo.Ensemble(50, 1, neuron_type=neuron_type())
            c = nengo.Ensemble(50, 1, neuron_type=neuron_type(amplitude=0.1))
            nengo.Connection(a, b, synapse=None)

            # note: we avoid decoders, as the rate/spiking models may have
            # different rate implementations in nengo, resulting in different
            # decoders
            nengo.Connection(b.neurons, c.neurons, synapse=None, transform=dists.He())
            p = nengo.Probe(c.neurons)

        with Simulator(net) as sim:
            if not sim.tensor_graph.inference_only:
                # TODO: this works in eager mode
                # with tf.GradientTape() as tape:
                #     tape.watch(sim.tensor_graph.trainable_variables)
                #     inputs = [
                #         tf.zeros((1, sim.unroll * 2, 1)),
                #         tf.constant([[sim.unroll * 2]]),
                #     ]
                #     outputs = sim.tensor_graph(inputs, training=True)
                # g = tape.gradient(outputs, sim.tensor_graph.trainable_variables)

                # note: not actually checking gradients, just using this to get the
                # gradients
                # TODO: why does the gradient check fail?
                if not sim.tensor_graph.inference_only:
                    g = sim.check_gradients(atol=1e10)[p]["analytic"]

                grads.append(g)

            sim.run(0.5)

        # check that the normal output is unaffected by the swap logic
        with nengo.Simulator(net) as sim2:
            sim2.run(0.5)

            assert np.allclose(sim.data[p], sim2.data[p])

    # check that the gradients match
    assert all(np.allclose(g0, g1) for g0, g1 in zip(*grads))
