from collections import OrderedDict
import os
import shutil

import nengo
from nengo.exceptions import SimulationError, SimulatorClosed, ReadonlyError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import DATA_DIR
from nengo_dl.simulator import ProbeDict


def test_persistent_state(Simulator, seed):
    """Make sure that state is preserved between runs."""

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(1000, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with Simulator(net) as sim:
        sim.run_steps(100)
        data = sim.data[p]
        sim.reset()

        sim.run_steps(100)
        data2 = sim.data[p]
        sim.reset()

        for _ in range(100 // sim.unroll):
            sim.run_steps(sim.unroll)
        data3 = sim.data[p]

    assert np.allclose(data, data2)
    assert np.allclose(data2, data3)


def test_step_blocks(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.sin)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with Simulator(net, unroll_simulation=25) as sim1:
        sim1.run_steps(50)
    with Simulator(net, unroll_simulation=10) as sim2:
        sim2.run_steps(50)

    assert np.allclose(sim1.data[p], sim2.data[p])

    with pytest.warns(RuntimeWarning):
        with Simulator(net, unroll_simulation=5) as sim:
            sim.run_steps(2)


def test_minibatch(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        inp = [nengo.Node(output=[0.5]), nengo.Node(output=np.sin),
               nengo.Node(output=nengo.processes.WhiteSignal(5, 0.5,
                                                             seed=seed))]

        ens = [
            nengo.Ensemble(10, 1, neuron_type=nengo.AdaptiveLIF()),
            nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate()),
            nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise(seed=seed))]

        nengo.Connection(inp[0], ens[0])
        nengo.Connection(inp[1], ens[1], synapse=None)
        nengo.Connection(inp[2], ens[2], synapse=nengo.Alpha(0.1),
                         transform=[[1], [1]])
        conn = nengo.Connection(ens[0], ens[1], learning_rule_type=nengo.PES())
        nengo.Connection(inp[0], conn.learning_rule)

        ps = [nengo.Probe(e) for e in ens]

    with Simulator(net, minibatch_size=None) as sim:
        probe_data = [[] for _ in ps]
        for i in range(5):
            sim.run_steps(100)

            for j, p in enumerate(ps):
                probe_data[j] += [sim.data[p]]

            sim.reset()

        probe_data = [np.stack(x, axis=0) for x in probe_data]

    with Simulator(net, minibatch_size=5) as sim:
        sim.run_steps(100)

    assert np.allclose(sim.data[ps[0]], probe_data[0], atol=1e-6)
    assert np.allclose(sim.data[ps[1]], probe_data[1], atol=1e-6)
    assert np.allclose(sim.data[ps[2]], probe_data[2], atol=1e-6)


def test_input_feeds(Simulator):
    minibatch_size = 10

    with nengo.Network() as net:
        inp = nengo.Node([0, 0, 0])
        out = nengo.Node(size_in=3)
        nengo.Connection(inp, out, synapse=None)
        p = nengo.Probe(out)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        val = np.random.randn(minibatch_size, 50, 3)
        sim.run_steps(50, input_feeds={inp: val})
        assert np.allclose(sim.data[p], val)

        with pytest.raises(nengo.exceptions.SimulationError):
            sim.run_steps(5, input_feeds={
                inp: np.random.randn(
                    minibatch_size + 1, 5, 3)})

        with pytest.raises(nengo.exceptions.SimulationError):
            sim.run_steps(5, input_feeds={
                inp: np.random.randn(minibatch_size, 4, 3)})

        with pytest.raises(nengo.exceptions.SimulationError):
            sim.run_steps(5, input_feeds={
                inp: np.random.randn(minibatch_size, 5, 4)})


@pytest.mark.parametrize("neurons", (True, False))
def test_train_ff(Simulator, neurons, seed):
    minibatch_size = 4
    n_hidden = 20

    np.random.seed(seed)

    with nengo.Network() as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.1, 0.1)
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0])
        ens = nengo.Ensemble(n_hidden + 1, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(
            inp, ens.neurons if neurons else ens,
            transform=np.random.uniform(-1, 1, size=(n_hidden + neurons, 2)))
        nengo.Connection(
            ens.neurons if neurons else ens, out.neurons if neurons else out,
            transform=np.random.uniform(-1, 1, size=(1, n_hidden + neurons)))

        # TODO: why does training fail if we probe out instead of out.neurons?
        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed) as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.train({inp: x}, {p: y}, tf.train.GradientDescentOptimizer(1),
                  n_epochs=5000)

        sim.check_gradients(atol=5e-5)

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-4)


def test_train_recurrent(Simulator, seed):
    batch_size = 100
    minibatch_size = 100
    n_hidden = 30
    n_steps = 10

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(
            n_hidden, 1, neuron_type=nengo.RectifiedLinear(),
            gain=np.ones(n_hidden), bias=np.linspace(-1, 1, n_hidden))
        out = nengo.Node(size_in=1)

        nengo.Connection(inp, ens, synapse=None)
        nengo.Connection(ens, ens, synapse=0)
        nengo.Connection(ens, out, synapse=None)

        p = nengo.Probe(out)

    with Simulator(net, minibatch_size=minibatch_size, seed=seed) as sim:
        x = np.outer(np.linspace(0, 1, batch_size),
                     np.ones(n_steps))[:, :, None]
        y = np.outer(np.linspace(0, 1, batch_size),
                     np.linspace(0, 1, n_steps))[:, :, None]

        sim.train({inp: x}, {p: y}, tf.train.GradientDescentOptimizer(2e-3),
                  n_epochs=2000)

        sim.check_gradients(sim.tensor_graph.losses[("mse", (p,))])

        sim.run_steps(n_steps, input_feeds={inp: x[:minibatch_size]})

    assert np.sqrt(np.mean((sim.data[p] - y[:minibatch_size]) ** 2)) < 0.05


@pytest.mark.parametrize("unroll", (1, 2))
def test_train_objective(Simulator, unroll, seed):
    minibatch_size = 1
    n_hidden = 20
    n_steps = 10

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(n_hidden, 1, neuron_type=nengo.RectifiedLinear())
        nengo.Connection(inp, ens, synapse=0.01)
        p = nengo.Probe(ens)

    with Simulator(net, minibatch_size=minibatch_size,
                   unroll_simulation=unroll, seed=seed) as sim:
        x = np.ones((minibatch_size, n_steps, 1))
        y = np.zeros((minibatch_size, n_steps, 1))

        def obj(output, target):
            return tf.reduce_mean((output[:, -1] - 0.5 - target[:, -1]) ** 2)

        sim.train({inp: x}, {p: y}, tf.train.GradientDescentOptimizer(1e-2),
                  n_epochs=1000, objective=obj)

        sim.check_gradients(sim.tensor_graph.losses[(obj, (p,))])

        sim.run_steps(n_steps, input_feeds={inp: x})

        assert np.allclose(sim.data[p][:, -1], y[:, -1] + 0.5,
                           atol=1e-3)


def test_train_rmsprop(Simulator, seed):
    minibatch_size = 4
    n_hidden = 5

    np.random.seed(seed)

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.1, 0.1)
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0])
        ens = nengo.Ensemble(n_hidden, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(
            inp, ens, transform=np.random.uniform(-1, 1, size=(n_hidden, 2)))
        nengo.Connection(
            ens, out, transform=np.random.uniform(-1, 1, size=(1, n_hidden)))

        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed, device="/cpu:0") as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.train({inp: x}, {p: y}, tf.train.RMSPropOptimizer(2e-4),
                  n_epochs=10000)

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-4)


@pytest.mark.xfail
@pytest.mark.gpu
def test_train_rmsprop_gpu(Simulator, seed):
    minibatch_size = 4
    n_hidden = 5

    np.random.seed(seed)

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.1, 0.1)
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0])
        ens = nengo.Ensemble(n_hidden, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(
            inp, ens, transform=np.random.uniform(-1, 1, size=(n_hidden, 2)))
        nengo.Connection(
            ens, out, transform=np.random.uniform(-1, 1, size=(1, n_hidden)))

        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed, device="/gpu:0") as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.train({inp: x}, {p: y}, tf.train.RMSPropOptimizer(2e-4),
                  n_epochs=10000)

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-4)


def test_train_errors(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        p = nengo.Probe(a)

    n_steps = 20
    with Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.train({a: np.ones((1, n_steps + 1, 1))},
                      {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(SimulationError):
            sim.train({a: np.ones((1, n_steps, 2))},
                      {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(SimulationError):
            sim.train({a: np.ones((1, n_steps, 1))},
                      {p: np.ones((1, n_steps + 1, 1))}, None)

        with pytest.raises(SimulationError):
            sim.train({a: np.ones((1, n_steps, 1))},
                      {p: np.ones((1, n_steps, 2))}, None)

    with pytest.raises(SimulatorClosed):
        sim.train({None: np.zeros((1, 1))}, None, None)


def test_loss(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(30, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    n_steps = 20
    with Simulator(net) as sim:
        sim.run_steps(n_steps)
        data = sim.data[p]

        assert np.allclose(sim.loss({inp: np.ones((4, n_steps, 1))},
                                    {p: np.zeros((4, n_steps, 1))}, "mse"),
                           np.mean(data ** 2))

        assert np.allclose(sim.loss({inp: np.ones((4, n_steps, 1))},
                                    {p: np.zeros((4, n_steps, 1))},
                                    objective=lambda x, y: tf.constant(2)),
                           2)

        with pytest.raises(SimulationError):
            sim.loss({inp: np.ones((1, n_steps + 1, 1))},
                     {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(SimulationError):
            sim.loss({inp: np.ones((1, n_steps, 2))},
                     {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(SimulationError):
            sim.loss({inp: np.ones((1, n_steps, 1))},
                     {p: np.ones((1, n_steps + 1, 1))}, None)

        with pytest.raises(SimulationError):
            sim.loss({inp: np.ones((1, n_steps, 1))},
                     {p: np.ones((1, n_steps, 2))}, None)

    with pytest.raises(SimulatorClosed):
        sim.loss({None: np.zeros((1, 1))}, None, None)


def test_generate_inputs(Simulator, seed):
    with nengo.Network() as net:
        proc = nengo.processes.WhiteNoise(seed=seed)
        inp = [nengo.Node([1]), nengo.Node(np.sin), nengo.Node(proc),
               nengo.Node([2])]

        p = [nengo.Probe(x) for x in inp]

    with Simulator(net, minibatch_size=2, unroll_simulation=3) as sim:
        feed = sim._generate_inputs({inp[0]: np.zeros((2, 3, 1))}, 3)

        ph = [sim.tensor_graph.invariant_ph[x] for x in inp]

        assert len(sim.tensor_graph.invariant_inputs) == len(inp)
        assert len(feed) == len(inp)

        sim.reset()
        sim.run_steps(3, input_feeds={inp[0]: np.zeros((2, 3, 1))})

        vals = [np.zeros((3, 1, 2)),
                np.tile(np.sin(sim.trange())[:, None, None], (1, 1, 2)),
                np.tile(proc.run_steps(3)[:, :, None], (1, 1, 2)),
                np.ones((3, 1, 2)) * 2]
        for i, x in enumerate(vals):
            assert np.allclose(feed[ph[i]], x)
            assert np.allclose(sim.data[p[i]], x.transpose(2, 0, 1))


def test_save_load_params(Simulator):
    with nengo.Network(seed=0) as net:
        out = nengo.Node(size_in=1)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(ens, out)

    with nengo.Network(seed=1) as net2:
        out = nengo.Node(size_in=1)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(ens, out)

    if not os.path.exists("./tmp"):
        os.makedirs("tmp")

    with Simulator(net) as sim:
        weights_var = [x for x in sim.tensor_graph.base_vars
                       if x.get_shape() == (1, 10)][0]
        weights0 = sim.sess.run(weights_var)
        sim.save_params("./tmp/tmp")

        # just check that this doesn't produce an error
        sim.print_params()

    with pytest.raises(SimulationError):
        sim.save_params(None)
    with pytest.raises(SimulationError):
        sim.load_params(None)
    with pytest.raises(SimulationError):
        sim.print_params(None)

    with Simulator(net2) as sim:
        weights_var = [x for x in sim.tensor_graph.base_vars
                       if x.get_shape() == (1, 10)][0]
        weights1 = sim.sess.run(weights_var)
        assert not np.allclose(weights0, weights1)

        sim.load_params("./tmp/tmp")

        weights2 = sim.sess.run(weights_var)
        assert np.allclose(weights0, weights2)

    shutil.rmtree("./tmp")


def test_model_passing(Simulator):
    # make sure that passing a built model to the Simulator works properly

    with nengo.Network() as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(20, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    model = nengo.builder.Model()
    model.build(net)

    with nengo.Simulator(None, model=model) as sim:
        sim.run_steps(10)

    canonical = sim.data[p]

    with Simulator(None, model=model) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p], canonical)

    # make sure that passing the same model to Simulator twice works
    with Simulator(None, model=model) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p], canonical)

    # make sure that passing that model back to the reference simulator works
    with nengo.Simulator(None, model=model) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p], canonical)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["/cpu:0", "/gpu:0", None])
def test_devices(Simulator, device, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    with nengo.Simulator(net) as sim:
        sim.run_steps(50)
        canonical = sim.data[p]

    with Simulator(net, device=device) as sim:
        sim.run_steps(50)
        assert np.allclose(sim.data[p], canonical)


def test_side_effects(Simulator):
    class MyFunc:
        x = 0

        def __call__(self, t):
            self.x += 1

    func = MyFunc()

    with nengo.Network() as net:
        nengo.Node(output=func)

    with Simulator(net, unroll_simulation=1) as sim:
        sim.run_steps(10)

    assert func.x == 11


def test_tensorboard(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        nengo.Probe(a)

    with Simulator(net, tensorboard=True):
        assert os.path.exists("%s/None/run_0" % DATA_DIR)

    with Simulator(net, tensorboard=True):
        assert os.path.exists("%s/None/run_1" % DATA_DIR)

    net.label = "test"

    with Simulator(net, tensorboard=True):
        assert os.path.exists("%s/test/run_0" % DATA_DIR)


def test_profile(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        nengo.Probe(a)

    with Simulator(net) as sim:
        sim.run_steps(5, profile=True)

        assert os.path.exists("%s/nengo_dl_profile.json" % DATA_DIR)


def test_dt_readonly(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        nengo.Probe(a)

    with Simulator(net) as sim:
        with pytest.raises(ReadonlyError):
            sim.dt = 1


def test_probe_dict():
    a = ProbeDict(OrderedDict({0: [np.zeros((1, 3, 5)), np.ones((1, 3, 5))],
                               1: [np.ones((1, 3, 1)), np.zeros((1, 3, 1))]}),
                  {0: 5, 1: None})
    assert a[0].shape == (5, 2, 3)
    assert np.all(a[0][:, 0] == 0)
    assert np.all(a[0][:, 1] == 1)

    assert a[1].shape == (2, 3)
    assert np.all(a[1][1] == 0)
    assert np.all(a[1][0] == 1)

    assert len(a) == 2
    for x, y in zip(a, (0, 1)):
        assert x == y


def test_deprecation(Simulator):
    with pytest.warns(DeprecationWarning):
        Simulator(None, step_blocks=1)

    with pytest.warns(DeprecationWarning):
        Simulator(None, unroll_simulation=True)
