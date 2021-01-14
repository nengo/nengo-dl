# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf
from packaging import version

from nengo_dl import (
    LeakyReLU,
    SoftLIFRate,
    SpikingLeakyReLU,
    compat,
    config,
    dists,
    neuron_builders,
)


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
        assert f"sigma={sigma}" in x

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


@pytest.mark.eager_only
@pytest.mark.parametrize(
    "rate, spiking",
    [
        (nengo.RectifiedLinear, nengo.SpikingRectifiedLinear),
        (nengo.LIFRate, nengo.LIF),
        (SoftLIFRate, nengo.LIF),
        pytest.param(
            nengo.LIFRate,
            lambda **kwargs: nengo.PoissonSpiking(nengo.LIFRate(**kwargs)),
            marks=pytest.mark.skipif(
                version.parse(nengo.__version__) < version.parse("3.1.0"),
                reason="PoissonSpiking does not exist",
            ),
        ),
        pytest.param(
            nengo.LIFRate,
            lambda **kwargs: nengo.RegularSpiking(nengo.LIFRate(), **kwargs),
            marks=pytest.mark.skipif(
                version.parse(nengo.__version__) < version.parse("3.1.0"),
                reason="RegularSpiking does not exist",
            ),
        ),
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
                with tf.GradientTape() as tape:
                    tape.watch(sim.tensor_graph.trainable_variables)
                    inputs = [
                        tf.zeros((1, sim.unroll * 2, 1)),
                        tf.constant([[sim.unroll * 2]]),
                    ]
                    outputs = sim.tensor_graph(inputs, training=True)[:-1]
                g = tape.gradient(outputs, sim.tensor_graph.trainable_variables)

                grads.append(g)

            sim.run(0.5)

        # check that the normal output is unaffected by the swap logic
        with nengo.Simulator(net) as sim2:
            sim2.run(0.5)

            if not isinstance(neuron_type(), compat.PoissonSpiking):
                # we don't expect these to match for poissonspiking, since we have
                # different rng implementations in numpy vs tensorflow
                assert np.allclose(sim.data[p], sim2.data[p])

    # check that the gradients match
    assert all(np.allclose(g0, g1) for g0, g1 in zip(*grads))


@pytest.mark.parametrize("Neurons", (LeakyReLU, SpikingLeakyReLU))
def test_leaky_relu(Simulator, Neurons):
    assert np.allclose(Neurons(negative_slope=0.1).rates([-2, 2], 1, 0), [[-0.2], [2]])

    assert np.allclose(
        Neurons(negative_slope=0.1, amplitude=0.1).rates([-2, 2], 1, 0),
        [[-0.02], [0.2]],
    )

    if Neurons.spiking and version.parse(nengo.__version__) > version.parse("3.0.0"):
        kwargs = dict(initial_state={"voltage": nengo.dists.Choice([0])})
    else:
        kwargs = dict()

    with nengo.Network() as net:
        vals = np.linspace(-400, 400, 10)
        ens0 = nengo.Ensemble(
            10,
            1,
            neuron_type=Neurons(negative_slope=0.1, amplitude=2, **kwargs),
            gain=nengo.dists.Choice([1]),
            bias=vals,
        )
        ens1 = nengo.Ensemble(
            10,
            1,
            neuron_type=Neurons(negative_slope=0.5, **kwargs),
            gain=nengo.dists.Choice([1]),
            bias=vals,
        )
        p0 = nengo.Probe(ens0.neurons)
        p1 = nengo.Probe(ens1.neurons)

    with Simulator(net) as sim:
        # make sure that ops have been merged
        assert (
            len(
                [
                    ops
                    for ops in sim.tensor_graph.plan
                    if isinstance(ops[0], nengo.builder.neurons.SimNeurons)
                ]
            )
            == 1
        )

        sim.run(1.0)

        assert np.allclose(
            np.sum(sim.data[p0], axis=0) * sim.dt,
            np.where(vals < 0, vals * 0.1 * 2, vals * 2),
            atol=1,
        )

        assert np.allclose(
            np.sum(sim.data[p1], axis=0) * sim.dt,
            np.where(vals < 0, vals * 0.5, vals),
            atol=1,
        )

    # check that it works in the regular nengo simulator as well
    with nengo.Simulator(net) as sim:
        sim.run(1.0)

        assert np.allclose(
            np.sum(sim.data[p0], axis=0) * sim.dt,
            np.where(vals < 0, vals * 0.1 * 2, vals * 2),
            atol=1,
        )

        assert np.allclose(
            np.sum(sim.data[p1], axis=0) * sim.dt,
            np.where(vals < 0, vals * 0.5, vals),
            atol=1,
        )


@pytest.mark.xfail(
    version.parse(nengo.__version__) < version.parse("3.1.0"),
    reason="RegularSpiking does not exist",
)
@pytest.mark.parametrize("inference_only", (True, False))
def test_regular_spiking(Simulator, inference_only, seed):
    with nengo.Network() as net:
        config.configure_settings(inference_only=inference_only)

        inp = nengo.Node([1])
        ens0 = nengo.Ensemble(
            100, 1, neuron_type=nengo.SpikingRectifiedLinear(amplitude=2), seed=seed
        )
        ens1 = nengo.Ensemble(
            100,
            1,
            neuron_type=nengo.RegularSpiking(nengo.RectifiedLinear(), amplitude=2),
            seed=seed,
        )

        nengo.Connection(inp, ens0)
        nengo.Connection(inp, ens1)

        p0 = nengo.Probe(ens0.neurons)
        p1 = nengo.Probe(ens1.neurons)

    with pytest.warns(None) as recwarns:
        with Simulator(net) as sim:
            sim.run_steps(50)

    assert np.allclose(sim.data[p0], sim.data[p1])
    # check that it is actually using the tensorflow implementation
    assert not any(
        "native TensorFlow implementation" in str(w.message) for w in recwarns
    )


@pytest.mark.skipif(
    version.parse(nengo.__version__) < version.parse("3.1.0"),
    reason="Stochastic/PoissonSpiking do not exist",
)
@pytest.mark.parametrize("inference_only", (True, False))
def test_random_spiking(Simulator, inference_only, seed):
    with nengo.Network() as net:
        config.configure_settings(inference_only=inference_only)

        inp = nengo.Node([1])
        ens0 = nengo.Ensemble(100, 1, neuron_type=nengo.Tanh(), seed=seed)
        ens1 = nengo.Ensemble(
            100, 1, neuron_type=nengo.StochasticSpiking(nengo.Tanh()), seed=seed
        )
        ens2 = nengo.Ensemble(
            100, 1, neuron_type=nengo.PoissonSpiking(nengo.Tanh()), seed=seed
        )

        nengo.Connection(inp, ens0, synapse=None)
        nengo.Connection(inp, ens1, synapse=None)
        nengo.Connection(inp, ens2, synapse=None)

        p0 = nengo.Probe(ens0.neurons)
        p1 = nengo.Probe(ens1.neurons)
        p2 = nengo.Probe(ens2.neurons)

    with pytest.warns(None) as recwarns:
        with Simulator(net, seed=seed) as sim:
            sim.run_steps(10000)

    assert not any(
        "native TensorFlow implementation" in str(w.message) for w in recwarns
    )

    assert np.allclose(
        sim.data[p0][0], np.mean(sim.data[p1], axis=0), atol=1, rtol=2e-1
    )
    assert np.allclose(sim.data[p0], np.mean(sim.data[p2], axis=0), atol=1, rtol=1e-1)


def test_bad_step_return_error(Simulator, monkeypatch):
    class CustomNeuron(nengo.LIF):
        pass

    class CustomNeuronBuilder(neuron_builders.TFNeuronBuilder):
        def step(self, J, dt, voltage, refractory_time):
            return J, voltage

    monkeypatch.setitem(
        neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL,
        CustomNeuron,
        CustomNeuronBuilder,
    )

    with nengo.Network(seed=0) as net:
        nengo.Ensemble(10, 1, neuron_type=CustomNeuron())

    with pytest.raises(ValueError, match="must return a tuple with the neuron output"):
        with Simulator(net):
            pass
