# pylint: disable=missing-docstring

import nengo
from nengo.synapses import Alpha, LinearFilter
from nengo.tests.test_synapses import run_synapse, allclose
from nengo.utils.filter_design import ss2tf
import numpy as np
import pytest


def test_alpha(Simulator, seed):
    dt = 1e-3
    tau = 0.03
    num, den = [1], [tau ** 2, 2 * tau, 1]

    t, x, yhat = run_synapse(Simulator, seed, Alpha(tau), dt=dt)
    y = LinearFilter(num, den).filt(x, dt=dt, y0=0)

    assert allclose(t, y, yhat, delay=dt, atol=5e-5)


def test_alpha_merged(Simulator, seed):
    with nengo.Network() as net:
        u = nengo.Node(output=nengo.processes.WhiteSignal(
            1, high=10, seed=seed))
        p0 = nengo.Probe(u, synapse=Alpha(0.03))
        p1 = nengo.Probe(u, synapse=Alpha(0.1))

    with nengo.Simulator(net) as sim:
        sim.run(1)
        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net) as sim:
        sim.run(1)
        assert np.allclose(sim.data[p0], canonical[0], atol=5e-5)
        assert np.allclose(sim.data[p1], canonical[1], atol=5e-5)


def test_alpha_multidim(Simulator, seed):
    with nengo.Network() as net:
        u0 = nengo.Node(output=nengo.processes.WhiteSignal(
            1, high=10, seed=seed), size_out=3)
        u1 = nengo.Node(output=nengo.processes.WhiteSignal(
            1, high=10, seed=seed), size_out=3)
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
    num = np.array([1.60000000e+09])
    den = np.array([1.00000000e+00, 5.22625186e+02, 1.36568542e+05,
                    2.09050074e+07, 1.60000000e+09])

    synapse = LinearFilter(num, den, analog=True)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert allclose(t, y, yhat, delay=dt, atol=5e-4)


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

    assert allclose(t, y, yhat, delay=dt * 2 if zero == "D" else dt, atol=5e-5)


@pytest.mark.training
def test_linear_filter_gradient(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=Alpha(0.01))
        nengo.Probe(b, synapse=Alpha(0.1))

    with Simulator(net) as sim:
        sim.check_gradients()
