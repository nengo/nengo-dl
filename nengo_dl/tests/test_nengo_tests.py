# pylint: disable=missing-docstring

import nengo
import numpy as np
import pkg_resources
import pytest
from nengo.builder.operator import DotInc, ElementwiseInc
from nengo.builder.signal import Signal

import nengo_dl
from nengo_dl.compat import SimProbe
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

    zero = Signal([0.0], name="zero")
    one = Signal([1.0], name="one")
    five = Signal([5.0], name="five")
    zeroarray = Signal([[0.0], [0.0], [0.0]], name="zeroarray")
    array = Signal([1.0, 2.0, 3.0], name="array")

    m = nengo.builder.Model(dt=0)
    m.add_op(ElementwiseInc(zero, zero, five))
    m.add_op(DotInc(zeroarray, one, array))
    m.add_op(SimProbe(zero))
    m.add_op(SimProbe(one))
    m.add_op(SimProbe(five))
    m.add_op(SimProbe(zeroarray))
    m.add_op(SimProbe(array))

    probes = [
        dummies.Probe(zero, add_to_container=False),
        dummies.Probe(one, add_to_container=False),
        dummies.Probe(five, add_to_container=False),
        dummies.Probe(array, add_to_container=False),
    ]
    m.probes += probes
    for p in probes:
        m.sig[p]["in"] = p.target

    with Simulator(None, model=m) as sim:
        sim.run_steps(3)
        assert np.allclose(sim.data[probes[0]], 0)
        assert np.allclose(sim.data[probes[1]], 1)
        assert np.allclose(sim.data[probes[2]], 5)
        assert np.allclose(sim.data[probes[3]], [1, 2, 3])


def test_entry_point():
    sims = [
        ep.load(require=False)
        for ep in pkg_resources.iter_entry_points(group="nengo.backends")
    ]
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


@pytest.mark.parametrize("bits", ["16", "32", "64"])
def test_dtype(Simulator, request, seed, bits):
    # Ensure dtype is set back to default after the test, even if it fails
    default = nengo.rc.get("precision", "bits")
    request.addfinalizer(lambda: nengo.rc.set("precision", "bits", default))

    float_dtype = np.dtype(getattr(np, f"float{bits}"))
    int_dtype = np.dtype(getattr(np, f"int{bits}"))

    with nengo.Network() as model:
        nengo_dl.configure_settings(dtype=f"float{bits}")

        u = nengo.Node([0.5, -0.4])
        a = nengo.Ensemble(10, 2)
        nengo.Connection(u, a)
        p = nengo.Probe(a)

    with Simulator(model) as sim:
        sim.step()

        # check that the builder has created signals of the correct dtype
        # (note that we may not necessarily use that dtype during simulation)
        for sig in sim.tensor_graph.signals:
            assert sig.dtype in (float_dtype, int_dtype), f"Signal '{sig}' wrong dtype"

        objs = (obj for obj in model.all_objects if sim.data[obj] is not None)
        for obj in objs:
            for x in (x for x in sim.data[obj] if isinstance(x, np.ndarray)):
                assert x.dtype == float_dtype, obj

        assert sim.data[p].dtype == float_dtype


@pytest.mark.parametrize("use_dist", (False, True))
def test_sparse(use_dist, Simulator, rng, seed, monkeypatch):
    # modified version of nengo test_sparse for scipy=False, where we
    # don't expect a warning

    scipy_sparse = pytest.importorskip("scipy.sparse")

    input_d = 4
    output_d = 2
    shape = (output_d, input_d)

    inds = np.asarray([[0, 0], [1, 1], [0, 2], [1, 3]])
    weights = rng.uniform(0.25, 0.75, size=4)
    if use_dist:
        init = nengo.dists.Uniform(0.25, 0.75)
        indices = inds
    else:
        init = scipy_sparse.csr_matrix((weights, inds.T), shape=shape)
        indices = None

    transform = nengo.transforms.Sparse(shape, indices=indices, init=init)

    sim_time = 1.0
    with nengo.Network(seed=seed) as net:
        x = nengo.processes.WhiteSignal(period=sim_time, high=10, seed=seed + 1)
        u = nengo.Node(x, size_out=4)
        a = nengo.Ensemble(100, 2)
        conn = nengo.Connection(u, a, synapse=None, transform=transform)
        ap = nengo.Probe(a, synapse=0.03)

    def run_sim():
        with Simulator(net) as sim:
            sim.run(sim_time)
        return sim

    sim = run_sim()

    actual_weights = sim.data[conn].weights

    full_transform = np.zeros(shape)
    full_transform[inds[:, 0], inds[:, 1]] = weights
    if use_dist:
        actual_weights = actual_weights.toarray()
        assert np.array_equal(actual_weights != 0, full_transform != 0)
        full_transform[:] = actual_weights

    conn.transform = full_transform
    with Simulator(net) as ref_sim:
        ref_sim.run(sim_time)

    assert np.allclose(sim.data[ap], ref_sim.data[ap])


def test_gain_bias(Simulator):
    N = 17
    D = 2

    gain = np.random.uniform(low=0.2, high=5, size=N)
    bias = np.random.uniform(low=0.2, high=1, size=N)

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(N, D)
        a.gain = gain
        a.bias = bias

    with Simulator(model) as sim:
        assert np.allclose(gain, sim.data[a].gain)
        assert np.allclose(bias, sim.data[a].bias)


def test_multirun(Simulator, rng):
    model = nengo.Network(label="Multi-run")

    with Simulator(model) as sim:
        t_stops = sim.dt * rng.randint(low=100, high=2000, size=10)

        # round times to be multiples of 10*dt, so that simulation times will still
        # fall on the right time with unroll_simulation != 1
        t_stops = np.around(t_stops, decimals=2)

        t_sum = 0
        for ti in t_stops:
            sim.run(ti)
            sim_t = sim.trange()
            t = sim.dt * np.arange(1, len(sim_t) + 1)
            assert np.allclose(sim_t, t)

            t_sum += ti
            assert np.allclose(sim_t[-1], t_sum)


def test_time_absolute(Simulator):
    m = nengo.Network()
    with Simulator(m) as sim:
        # modify runtime so that it is a multiple of unroll_simulations
        sim.run(0.01)
    assert np.allclose(sim.trange(), np.arange(sim.dt, 0.01 + sim.dt, sim.dt))


def test_invalid_run_time(Simulator):
    net = nengo.Network()
    with Simulator(net) as sim:
        with pytest.raises(nengo.exceptions.ValidationError):
            sim.run(-0.0001)
        with pytest.warns(UserWarning):
            sim.run(0)
        sim.run(0.0006)  # Rounds up to 0.001

        # may run for more than 1 steps if unroll_simulation != 1
        assert sim.n_steps == sim.unroll


def test_steps(Simulator):
    dt = 0.001
    m = nengo.Network(label="test_steps")
    with Simulator(m, dt=dt) as sim:
        assert sim.n_steps == 0 * sim.unroll
        assert np.allclose(sim.time, 0 * dt * sim.unroll)
        sim.step()
        assert sim.n_steps == 1 * sim.unroll
        assert np.allclose(sim.time, 1 * dt * sim.unroll)
        sim.step()
        assert sim.n_steps == 2 * sim.unroll
        assert np.allclose(sim.time, 2 * dt * sim.unroll)

        assert np.isscalar(sim.n_steps)
        assert np.isscalar(sim.time)
