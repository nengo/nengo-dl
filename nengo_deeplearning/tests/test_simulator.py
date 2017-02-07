import nengo
import numpy as np

from nengo_deeplearning.tests import TestSimulator


def test_persistent_state():
    """Make sure that state is preserved between runs."""

    with nengo.Network(seed=0) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(1000, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with TestSimulator(net, step_blocks=5) as sim:
        sim.run_steps(100)
        data = sim.data[p]
        sim.reset()

        sim.run_steps(100)
        data2 = sim.data[p]
        sim.reset()

        for _ in range(20):
            sim.run_steps(5)
        data3 = sim.data[p]

    assert np.allclose(data, data2)
    assert np.allclose(data2, data3)


def test_step_blocks():
    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.sin)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with TestSimulator(net, step_blocks=25) as sim1:
        sim1.run_steps(50)
    with TestSimulator(net, step_blocks=10) as sim2:
        sim2.run_steps(50)
    with TestSimulator(net, unroll_simulation=False, step_blocks=None) as sim3:
        sim3.run_steps(50)

    assert np.allclose(sim1.data[p], sim2.data[p])
    assert np.allclose(sim2.data[p], sim3.data[p])


def test_unroll_simulation():
    # note: we run this multiple times because the effects of unrolling can
    # be somewhat stochastic depending on the op order
    for _ in range(10):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.sin)
            ens = nengo.Ensemble(10, 1)
            nengo.Connection(inp, ens)
            p = nengo.Probe(ens)

        with TestSimulator(net, step_blocks=10,
                           unroll_simulation=False) as sim1:
            sim1.run_steps(50)

        with TestSimulator(net, step_blocks=10,
                           unroll_simulation=True) as sim2:
            sim2.run_steps(50)

        assert np.allclose(sim1.data[p], sim2.data[p])


def test_minibatch():
    with nengo.Network(seed=0) as net:
        inp = [nengo.Node(output=[0.5]), nengo.Node(output=np.sin),
               nengo.Node(output=nengo.processes.WhiteSignal(5, 0.5, seed=0))]

        ens = [
            nengo.Ensemble(10, 1, neuron_type=nengo.AdaptiveLIF()),
            nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate()),
            nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise(seed=0))]

        nengo.Connection(inp[0], ens[0])
        nengo.Connection(inp[1], ens[1], synapse=None)
        nengo.Connection(inp[2], ens[2], synapse=nengo.Alpha(0.1),
                         transform=[[1], [1]])
        conn = nengo.Connection(ens[0], ens[1], learning_rule_type=nengo.PES())
        nengo.Connection(inp[0], conn.learning_rule)

        ps = [nengo.Probe(e) for e in ens]

    with TestSimulator(net, minibatch_size=None) as sim:
        probe_data = [[] for _ in ps]
        for i in range(5):
            sim.run_steps(1000)

            for j, p in enumerate(ps):
                probe_data[j] += [sim.data[p]]

            sim.reset()

        probe_data = [np.stack(x, axis=0) for x in probe_data]

    with TestSimulator(net, minibatch_size=5) as sim:
        sim.run_steps(1000)

    assert np.allclose(sim.data[ps[0]], probe_data[0])
    assert np.allclose(sim.data[ps[1]], probe_data[1])
    assert np.allclose(sim.data[ps[2]], probe_data[2])

# TODO: test input_feeds
