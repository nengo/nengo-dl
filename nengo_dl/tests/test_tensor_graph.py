import nengo
import pytest


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
