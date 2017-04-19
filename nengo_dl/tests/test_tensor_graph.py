import nengo
import pytest

from nengo_dl import tensor_graph


@pytest.mark.parametrize("unroll", (True, False))
def test_gradients(Simulator, unroll, seed):
    step_blocks = 10
    minibatch_size = 4

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        inp = nengo.Node([0], label="inp")

        # sigmoid neurons
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())

        # normal decoded connection
        nengo.Connection(inp, ens)

        # recurrent connection
        nengo.Connection(ens, ens, transform=0.1)

        # rectified neurons
        ens2 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear())

        # neuron--neuron connection
        nengo.Connection(ens, ens2, transform=[[1], [1]],
                         solver=nengo.solvers.LstsqL2(weights=True))

        # sliced output, no synapse
        nengo.Connection(inp, ens2[0], synapse=None, transform=0.5)

        # sliced input, sliced output
        inp2 = nengo.Node([0, 0], label="inp2")
        nengo.Connection(inp2[0], ens2[1])

        nengo.Probe(ens)
        nengo.Probe(ens2)

    with Simulator(net, step_blocks=step_blocks, unroll_simulation=unroll,
                   minibatch_size=minibatch_size) as sim:
        sim.check_gradients(atol=1e-4)


def test_build_loss(Simulator):
    # check that the loss caching works

    with nengo.Network() as net:
        inp = nengo.Node([0])
        p = nengo.Probe(inp)

    with Simulator(net) as sim:
        assert (sim.tensor_graph.build_loss("mse", (p,)) is
                sim.tensor_graph.build_loss("mse", (p,)))

        def loss(*args):
            return args[0]

        assert (sim.tensor_graph.build_loss(loss, (p,)) is
                sim.tensor_graph.build_loss(loss, (p,)))

# TODO: add test for optimizer caching


def test_mark_signals():
    with nengo.Network() as net:
        ens0 = nengo.Ensemble(10, 1, neuron_type=nengo.LIF())
        ens1 = nengo.Ensemble(20, 1, neuron_type=nengo.Direct())
        ens2 = nengo.Ensemble(30, 1)
        conn0 = nengo.Connection(ens0, ens1)
        conn1 = nengo.Connection(ens0, ens1, learning_rule_type=nengo.PES())
        conn2 = nengo.Connection(ens0, ens2, learning_rule_type=nengo.Voja())
        nengo.Probe(ens2)

    model = nengo.builder.Model()
    model.build(net)

    tensor_graph.mark_signals(model)

    assert model.sig[ens0]["encoders"].trainable
    assert model.sig[ens1]["encoders"].trainable
    assert not model.sig[ens2]["encoders"].trainable
    assert model.sig[ens0.neurons]["bias"].trainable
    assert model.sig[ens2.neurons]["bias"].trainable
    assert model.sig[conn0]["weights"].trainable
    assert not model.sig[conn1]["weights"].trainable
    assert model.sig[conn2]["weights"].trainable

    trainables = (
        model.sig[ens0]["encoders"], model.sig[ens1]["encoders"],
        model.sig[ens0.neurons]["bias"], model.sig[ens2.neurons]["bias"],
        model.sig[conn0]["weights"], model.sig[conn2]["weights"])

    for op in model.operators:
        for sig in op.all_signals:
            if sig in trainables:
                assert sig.trainable
            else:
                assert not sig.trainable
