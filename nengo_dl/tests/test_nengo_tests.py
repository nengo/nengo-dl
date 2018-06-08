import pkg_resources

import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import ElementwiseInc, DotInc
import numpy as np
import pytest


import nengo_dl


def test_warn_on_opensim_del(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)

    sim = Simulator(net)
    with pytest.warns(RuntimeWarning):
        sim.__del__()
    sim.close()


def test_args(Simulator):
    class Fn(object):
        def __init__(self):
            self.last_x = None

        def __call__(self, t, x):
            assert t.dtype == x.dtype
            assert t.shape == ()
            assert isinstance(x, np.ndarray)
            assert self.last_x is not x  # x should be a new copy on each call
            self.last_x = x
            assert np.allclose(x[0], t)

    with nengo.Network() as model:
        u = nengo.Node(lambda t: t)
        v = nengo.Node(Fn(), size_in=1, size_out=0)
        nengo.Connection(u, v, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.01)


def test_signal_init_values(Simulator):
    """Tests that initial values are not overwritten."""

    zero = Signal([0.0])
    one = Signal([1.0])
    five = Signal([5.0])
    zeroarray = Signal([[0.0], [0.0], [0.0]])
    array = Signal([1.0, 2.0, 3.0])

    class DummyProbe(nengo.Probe):
        # pylint: disable=super-init-not-called
        def __init__(self, target):
            # bypass target validation
            nengo.Probe.target.data[self] = target

    m = nengo.builder.Model(dt=0)
    m.operators += [ElementwiseInc(zero, zero, five),
                    DotInc(zeroarray, one, array)]

    probes = [DummyProbe(zero, add_to_container=False),
              DummyProbe(one, add_to_container=False),
              DummyProbe(five, add_to_container=False),
              DummyProbe(array, add_to_container=False)]
    m.probes += probes
    for p in probes:
        m.sig[p]['in'] = p.target

    with Simulator(None, model=m) as sim:
        sim.run_steps(3)
        assert np.allclose(sim.data[probes[0]], 0)
        assert np.allclose(sim.data[probes[1]], 1)
        assert np.allclose(sim.data[probes[2]], 5)
        assert np.allclose(sim.data[probes[3]], [1, 2, 3])


def test_entry_point():
    sims = [ep.load() for ep in
            pkg_resources.iter_entry_points(group='nengo.backends')]
    assert nengo_dl.Simulator in sims


def test_unconnected_node(Simulator):
    hits = np.array(0)
    dt = 0.001

    def f(_):
        hits[...] += 1

    model = nengo.Network()
    with model:
        nengo.Node(f, size_in=0, size_out=0)
    with Simulator(model, unroll_simulation=1) as sim:
        assert hits == 0
        sim.run(dt)
        assert hits == 1
        sim.run(dt)
        assert hits == 2
