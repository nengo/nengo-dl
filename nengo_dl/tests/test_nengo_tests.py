# pylint: disable=missing-docstring

from distutils.version import LooseVersion
import pkg_resources

import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import ElementwiseInc, DotInc
import numpy as np
import pytest
import tensorflow as tf

import nengo_dl
from nengo_dl.tests import dummies


def test_warn_on_opensim_del(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)

    sim = Simulator(net)
    with pytest.warns(RuntimeWarning):
        sim.__del__()
    sim.close()


def test_args(Simulator):
    class Fn:
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

    m = nengo.builder.Model(dt=0)
    m.operators += [ElementwiseInc(zero, zero, five),
                    DotInc(zeroarray, one, array)]

    probes = [dummies.Probe(zero, add_to_container=False),
              dummies.Probe(one, add_to_container=False),
              dummies.Probe(five, add_to_container=False),
              dummies.Probe(array, add_to_container=False)]
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
    if LooseVersion(tf.__version__) == "1.11.0":
        pytest.xfail("TensorFlow 1.11.0 has conflicting dependencies")

    sims = [ep.load(require=False) for ep in
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
