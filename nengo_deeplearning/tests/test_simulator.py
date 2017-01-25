import nengo
import numpy as np

import nengo_deeplearning as nengo_dl


def test_persistent_state():
    """Make sure that state is preserved between runs."""

    with nengo.Network(seed=0) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(1000, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with nengo_dl.Simulator(net, step_blocks=5) as sim:
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

    sim1 = nengo_dl.Simulator(net, step_blocks=25)
    sim2 = nengo_dl.Simulator(net, step_blocks=10)
    sim3 = nengo_dl.Simulator(net, unroll_simulation=False, step_blocks=None)

    sim1.run_steps(50)
    sim2.run_steps(50)
    sim3.run_steps(50)
    sim1.close()
    sim2.close()
    sim3.close()

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

        sim1 = nengo_dl.Simulator(net, step_blocks=10, unroll_simulation=False)
        sim2 = nengo_dl.Simulator(net, step_blocks=10, unroll_simulation=True)

        sim1.run_steps(50)
        sim2.run_steps(50)
        sim1.close()
        sim2.close()

        assert np.allclose(sim1.data[p], sim2.data[p])


# TODO: tests for minibatching (operators, processes, neurons, learning rules)
