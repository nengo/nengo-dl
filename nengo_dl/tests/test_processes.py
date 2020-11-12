# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
from nengo.builder.processes import SimProcess
from nengo.synapses import Alpha, LinearFilter, Lowpass, Triangle
from nengo.tests.test_synapses import run_synapse
from nengo.utils.filter_design import ss2tf
from nengo.utils.testing import signals_allclose


@pytest.mark.parametrize("Synapse", (Alpha, Triangle))
def test_merged(Simulator, Synapse, seed):
    with nengo.Network() as net:
        u = nengo.Node(output=nengo.processes.WhiteSignal(1, high=10, seed=seed))
        p0 = nengo.Probe(u, synapse=Synapse(0.03))
        p1 = nengo.Probe(u, synapse=Synapse(0.1))

    with nengo.Simulator(net) as sim:
        sim.run(1)
        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net, minibatch_size=3) as sim:
        assert (
            len([p for p in sim.tensor_graph.plan if isinstance(p[0], SimProcess)]) == 1
        )
        sim.run(1)
        assert np.allclose(sim.data[p0], canonical[0], atol=5e-5)
        assert np.allclose(sim.data[p1], canonical[1], atol=5e-5)


def test_general_minibatched(Simulator):
    with nengo.Network() as net:
        u = nengo.Node([0])
        p = nengo.Probe(u, synapse=Triangle(0.01))

    with Simulator(net, minibatch_size=3) as sim:
        data = {u: np.ones((3, 100, 1)) * np.arange(1, 4)[:, None, None]}
        sim.run_steps(100, data=data)

        for i in range(3):
            filt = p.synapse.filt(data[u][i], y0=0)
            assert np.allclose(sim.data[p][i, 1:], filt[:-1])


def test_alpha_multidim(Simulator, seed):
    with nengo.Network() as net:
        u0 = nengo.Node(
            output=nengo.processes.WhiteSignal(1, high=10, seed=seed), size_out=3
        )
        u1 = nengo.Node(
            output=nengo.processes.WhiteSignal(1, high=10, seed=seed), size_out=3
        )
        p0 = nengo.Probe(u0, synapse=Alpha(0.03))
        p1 = nengo.Probe(u1, synapse=Alpha(0.1))

    with nengo.Simulator(net) as sim:
        sim.run(1)
        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net) as sim:
        sim.run(1)
        assert np.allclose(sim.data[p0], canonical[0], atol=5e-5)
        assert np.allclose(sim.data[p1], canonical[1], atol=5e-5)


def test_linear_analog(Simulator, seed):
    dt = 1e-3

    # The following num, den are for a 4th order analog Butterworth filter,
    # generated with `scipy.signal.butter(4, 200, analog=True)`
    num = np.array([1.60000000e09])
    den = np.array(
        [1.00000000e00, 5.22625186e02, 1.36568542e05, 2.09050074e07, 1.60000000e09]
    )

    synapse = LinearFilter(num, den, analog=True)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-4)


@pytest.mark.parametrize("zero", ("C", "D", "X"))
def test_zero_matrices(Simulator, zero, seed):
    dt = 1e-3

    A = np.diag(np.ones(2) * dt)
    B = np.zeros((2, 1))
    B[0] = 1
    C = np.ones((1, 2))
    D = np.ones((1,))

    if zero == "C":
        C[...] = 0
    elif zero == "D":
        D[...] = 0

    num, den = ss2tf(A, B, C, D)
    num = num.flatten()

    synapse = LinearFilter(num, den, analog=False)

    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-5)


@pytest.mark.training
def test_linear_filter_gradient(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=Alpha(0.01))
        nengo.Probe(b, synapse=Alpha(0.1))

    with Simulator(net) as sim:
        sim.check_gradients()


def test_linearfilter_onex(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node(lambda t: np.sin(t * 10))

        tau = 0.1

        # check that linearfilter and lowpass are equivalent
        # (two versions of each to check that merging works as expected)
        p_lowpass0 = nengo.Probe(inp, synapse=tau)
        p_lowpass1 = nengo.Probe(inp, synapse=tau * 2)
        p_linear0 = nengo.Probe(inp, synapse=LinearFilter([1], [tau, 1]))
        p_linear1 = nengo.Probe(inp, synapse=LinearFilter([1], [tau * 2, 1]))

    with Simulator(net) as sim:
        sim.run(0.1)

        assert np.allclose(sim.data[p_lowpass0], sim.data[p_linear0])
        assert np.allclose(sim.data[p_lowpass1], sim.data[p_linear1])


@pytest.mark.parametrize(
    "synapse",
    (
        LinearFilter([0.1], [1], analog=False),  # NoX
        LinearFilter([1], [0.1, 1]),  # OneX
        Alpha(0.1),  # NoD
        LinearFilter(
            [0.0004166, 0.0016664, 0.0024996, 0.0016664, 0.0004166],
            [1.0, -3.18063855, 3.86119435, -2.11215536, 0.43826514],
        ),  # General
    ),
)
def test_linearfilter_minibatched(Simulator, synapse):
    n_steps = 10
    mini_size = 4
    signal_d = 3

    with nengo.Network() as net:
        inp = nengo.Node(np.zeros(signal_d))

        p0 = nengo.Probe(inp, synapse=synapse)
        p1 = nengo.Probe(inp, synapse=synapse)

    with Simulator(net, minibatch_size=mini_size) as sim:
        assert (
            len(
                [
                    ops
                    for ops in sim.tensor_graph.plan
                    if isinstance(ops[0], nengo.builder.processes.SimProcess)
                ]
            )
            == 1
        )

        data = (
            np.zeros((mini_size, n_steps, signal_d))
            + np.arange(mini_size)[:, None, None]
        )
        sim.run_steps(n_steps, data={inp: data})

    for i in range(mini_size):
        filt = synapse.filt(np.ones((n_steps, signal_d)) * i, y0=0)

        assert np.allclose(sim.data[p0][i, 1:], filt[:-1])
        assert np.allclose(sim.data[p1][i, 1:], filt[:-1])


@pytest.mark.parametrize("synapse", (Lowpass(0.1), LinearFilter([1], [0.1, 1])))
def test_weight_filter(Simulator, synapse):
    if not isinstance(synapse, Lowpass):
        pytest.xfail(reason="Multidimensional LinearFilter not implemented")

    with nengo.Network() as net:
        a = nengo.Node([0, 0])
        b = nengo.Node(size_in=2)
        c = nengo.Connection(a, b, transform=np.ones((2, 2)))
        p = nengo.Probe(c, "weights", synapse=synapse)

    with Simulator(net) as sim:
        sim.run_steps(10)

        assert np.allclose(sim.data[p][1:], synapse.filt(np.ones((9, 2, 2)), y0=0))
