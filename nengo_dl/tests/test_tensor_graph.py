import nengo
from nengo.exceptions import SimulationError
import pytest
import tensorflow as tf

from nengo_dl import tensor_graph, utils


@pytest.mark.parametrize("unroll", (1, 2))
def test_gradients(Simulator, unroll, seed):
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

    with Simulator(net, unroll_simulation=unroll,
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


def test_build_optimizer(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    # check optimizer caching
    with Simulator(net) as sim:
        opt = tf.train.GradientDescentOptimizer(0)
        assert (sim.tensor_graph.build_optimizer(opt, (p,), "mse") is
                sim.tensor_graph.build_optimizer(opt, (p,), "mse"))

    # error when no trainable elements
    with nengo.Network() as net:
        inp = nengo.Node([0])
        p = nengo.Probe(inp)

    with Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.tensor_graph.build_optimizer(opt, (p,), "mse")


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


def test_mark_signals_config():
    with nengo.Network() as net:
        utils.configure_trainable(net)
        net.config[nengo.Ensemble].trainable = False

        with nengo.Network() as subnet:
            utils.configure_trainable(subnet)

            # check that object in subnetwork inherits config from parent
            ens0 = nengo.Ensemble(10, 1, label="ens0")

            # check that ens.neurons can be set independent of ens
            subnet.config[ens0.neurons].trainable = True

            with nengo.Network():
                with nengo.Network() as subsubnet:
                    utils.configure_trainable(subsubnet)

                    # check that subnetworks can override parent configs
                    subsubnet.config[nengo.Ensemble].trainable = True
                    ens1 = nengo.Ensemble(10, 1, label="ens1")

            # check that instances can be set independent of class
            ens2 = nengo.Ensemble(10, 1, label="ens2")
            subnet.config[ens2].trainable = True

    model = nengo.builder.Model()
    model.build(net)
    tensor_graph.mark_signals(model)

    assert not model.sig[ens0]["encoders"].trainable
    assert model.sig[ens0.neurons]["bias"].trainable

    assert model.sig[ens1]["encoders"].trainable

    assert model.sig[ens2]["encoders"].trainable

    # check that learning rule connections can be manually set to True
    with nengo.Network() as net:
        utils.configure_trainable(net)

        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        conn0 = nengo.Connection(a, b, learning_rule_type=nengo.PES())
        net.config[conn0].trainable = True

    model = nengo.builder.Model()
    model.build(net)

    with pytest.warns(UserWarning):
        tensor_graph.mark_signals(model)

    assert model.sig[conn0]["weights"].trainable

    with nengo.Network() as net:
        utils.configure_trainable(net)

        a = nengo.Node([0])
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(a, ens, learning_rule_type=nengo.Voja())
        net.config[nengo.Ensemble].trainable = True

    model = nengo.builder.Model()
    model.build(net)

    with pytest.warns(UserWarning):
        tensor_graph.mark_signals(model)

    assert model.sig[ens]["encoders"].trainable

    # check that models with no toplevel work
    sig = nengo.builder.signal.Signal([0])
    op = nengo.builder.operator.Reset(sig, 1)
    model = nengo.builder.Model()
    model.add_op(op)

    with pytest.warns(UserWarning):
        tensor_graph.mark_signals(model)

    assert not sig.trainable
