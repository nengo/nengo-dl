from collections import OrderedDict
import itertools
import os

import nengo
from nengo.exceptions import (SimulationError, SimulatorClosed, ReadonlyError,
                              ValidationError)
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import configure_settings, tensor_layer, dists, DATA_DIR
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


@pytest.mark.parametrize("neurons", (True, False))
def test_train_ff(Simulator, neurons, seed):
    minibatch_size = 4
    n_hidden = 20

    np.random.seed(seed)

    with nengo.Network() as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        # TODO: separate this into two different inputs to test that
        # functionality
        inp = nengo.Node([0, 0])
        ens = nengo.Ensemble(n_hidden + 1, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(
            inp, ens.neurons if neurons else ens, transform=dists.Glorot())
        nengo.Connection(
            ens.neurons if neurons else ens, out.neurons,
            transform=dists.Glorot())

        # TODO: why does training fail if we probe out instead of out.neurons?
        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed) as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.train({inp: x}, {p: y}, tf.train.MomentumOptimizer(1, 0.9),
                  n_epochs=500)

        sim.check_gradients(atol=5e-5)

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-3)


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

        sim.train(
            {inp: x}, {p: y}, tf.train.RMSPropOptimizer(1e-3), n_epochs=200)

        sim.check_gradients(sim.tensor_graph.losses[("mse", (p,))])

        sim.run_steps(n_steps, input_feeds={inp: x[:minibatch_size]})

    assert np.sqrt(np.mean((sim.data[p] - y[:minibatch_size]) ** 2)) < 0.05


@pytest.mark.parametrize("unroll", (1, 2))
def test_train_objective(Simulator, unroll, seed):
    minibatch_size = 1
    n_hidden = 20
    n_steps = 10

    # TODO: make this have multiple outputs to test that functionality

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

        sim.train({inp: x}, {p: y}, tf.train.MomentumOptimizer(1e-2, 0.9),
                  n_epochs=100, objective=obj)

        sim.check_gradients([p])

        sim.run_steps(n_steps, input_feeds={inp: x})

        assert np.allclose(sim.data[p][:, -1], y[:, -1] + 0.5,
                           atol=1e-3)


def test_train_sparse(Simulator, seed):
    minibatch_size = 4
    n_hidden = 5

    np.random.seed(seed)

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0, 0, 0, 0])
        ens = nengo.Ensemble(n_hidden, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(2, 2, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(inp[[0, 2, 3]], ens, transform=dists.Glorot())
        nengo.Connection(ens, out, transform=dists.Glorot())

        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed) as sim:
        x = np.asarray([[[0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0]],
                        [[1, 0, 0, 0, 0]], [[1, 0, 1, 0, 0]]])
        y = np.asarray([[[0.1, 0]], [[0.9, 0]], [[0.9, 0]], [[0.1, 0]]])

        sim.train({inp: x}, {p: y}, tf.train.MomentumOptimizer(1, 0.9),
                  n_epochs=500)

        sim.step(input_feeds={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-3)


def test_train_errors(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        p = nengo.Probe(a)

    n_steps = 20
    with Simulator(net) as sim:
        with pytest.raises(ValidationError):
            sim.train({a: np.ones((1, n_steps + 1, 1))},
                      {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(ValidationError):
            sim.train({a: np.ones((2, n_steps, 1))},
                      {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(ValidationError):
            sim.train({a: np.ones((1, n_steps, 1))},
                      {p: np.ones((1, n_steps + 1, 1))}, None)

        with pytest.raises(ValidationError):
            sim.train({a: np.ones((1, n_steps, 1))},
                      {p: np.ones((2, n_steps, 1))}, None)

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

        with pytest.raises(ValidationError):
            sim.loss({inp: np.ones((1, n_steps + 1, 1))},
                     {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(ValidationError):
            sim.loss({inp: np.ones((2, n_steps, 1))},
                     {p: np.ones((1, n_steps, 1))}, None)

        with pytest.raises(ValidationError):
            sim.loss({inp: np.ones((1, n_steps, 1))},
                     {p: np.ones((1, n_steps + 1, 1))}, None)

        with pytest.raises(ValidationError):
            sim.loss({inp: np.ones((1, n_steps, 1))},
                     {p: np.ones((2, n_steps, 1))}, None)

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


def test_save_load_params(Simulator, tmpdir):
    with nengo.Network(seed=0) as net:
        inp = nengo.Node([0])
        out = nengo.Node(size_in=1)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        nengo.Connection(ens, out)

        configure_settings(trainable=None)
        net.config[ens].trainable = False

    with Simulator(net) as sim:
        weights_var = [x for x in sim.tensor_graph.base_vars
                       if x.get_shape() == (1, 10)][0]
        enc_var = sim.tensor_graph.base_vars[
            list(sim.tensor_graph.base_arrays_init.keys()).index(
                sim.tensor_graph.sig_map[sim.model.sig[ens]["encoders"]].key)]
        weights0, enc0 = sim.sess.run([weights_var, enc_var])
        sim.save_params(os.path.join(str(tmpdir), "train"))
        sim.save_params(os.path.join(str(tmpdir), "local"),
                        include_local=True)

        # just check that this doesn't produce an error
        sim.print_params()

    with pytest.raises(SimulationError):
        sim.save_params(None)
    with pytest.raises(SimulationError):
        sim.load_params(None)
    with pytest.raises(SimulationError):
        sim.print_params(None)

    with nengo.Network(seed=1) as net2:
        inp = nengo.Node([0])
        out = nengo.Node(size_in=1)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        nengo.Connection(ens, out)

        configure_settings(trainable=None)
        net2.config[ens].trainable = False

    with Simulator(net2) as sim:
        weights_var = [x for x in sim.tensor_graph.base_vars
                       if x.get_shape() == (1, 10)][0]
        enc_var = sim.tensor_graph.base_vars[
            list(sim.tensor_graph.base_arrays_init.keys()).index(
                sim.tensor_graph.sig_map[sim.model.sig[ens]["encoders"]].key)]
        weights1, enc1 = sim.sess.run([weights_var, enc_var])
        assert not np.allclose(weights0, weights1)
        assert not np.allclose(enc0, enc1)

        sim.load_params(os.path.join(str(tmpdir), "train"))

        weights2, enc2 = sim.sess.run([weights_var, enc_var])
        assert np.allclose(weights0, weights2)
        assert not np.allclose(enc0, enc2)

        sim.load_params(os.path.join(str(tmpdir), "local"), include_local=True)

        weights3, enc3 = sim.sess.run([weights_var, enc_var])
        assert np.allclose(weights0, weights3)
        assert np.allclose(enc0, enc3)


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


@pytest.mark.parametrize("mode", ("run", "train"))
def test_profile(Simulator, mode):
    with nengo.Network() as net:
        a = nengo.Node([0])
        x = nengo.Node(size_in=1)
        nengo.Connection(a, x)
        p = nengo.Probe(x)

    filename = os.path.join(DATA_DIR, "nengo_dl_profile.json")
    if os.path.exists(filename):
        os.remove(filename)
    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with Simulator(net) as sim:
        if mode == "run":
            sim.run_steps(5, profile=True)
        else:
            sim.train({a: np.zeros((1, 5, 1))}, {p: np.zeros((1, 5, 1))},
                      tf.train.GradientDescentOptimizer(1), profile=True)

        assert os.path.exists(filename)
        os.remove(filename)

        if mode == "run":
            sim.run_steps(5, profile="tmp.txt")
        else:
            sim.train({a: np.zeros((1, 5, 1))}, {p: np.zeros((1, 5, 1))},
                      tf.train.GradientDescentOptimizer(1), profile="tmp.txt")

        assert os.path.exists("tmp.txt")
        os.remove("tmp.txt")


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


@pytest.mark.parametrize(
    "pre_val, post_val", itertools.product(
        [0, lambda t: 0, nengo.processes.WhiteNoise(seed=0)],
        [1, lambda t: 1, nengo.processes.WhiteNoise(seed=1)]))
def test_node_output_change(Simulator, pre_val, post_val, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(pre_val)
        p = nengo.Probe(inp)

    with Simulator(net, unroll_simulation=1) as sim:
        sim.step()
        inp.output = post_val
        sim.step()
        inp.output = pre_val
        sim.step()

    if isinstance(pre_val, nengo.Process):
        step0, step2 = pre_val.run_steps(2)[:, 0]
    else:
        step0 = step2 = 0

    if isinstance(post_val, nengo.Process):
        step1 = post_val.run_steps(1)[0, 0]
    else:
        step1 = 1

    assert np.allclose(sim.data[p][:, 0], (step0, step1, step2))


def test_check_gradients_error(Simulator):
    # check_gradients detects nans in gradient
    with nengo.Network() as net:
        x = nengo.Node([0])
        y = tensor_layer(x, lambda x: 1 / x)
        nengo.Probe(y)

    with Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.check_gradients()

    # check_gradients detects errors in gradient (in this case caused by the
    # fact that nengo.Triangle doesn't have a TensorFlow implementation)
    with nengo.Network() as net:
        x = nengo.Node([0])
        nengo.Probe(x, synapse=nengo.Triangle(0.1))

    with Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.check_gradients()


def test_check_data(Simulator):
    with nengo.Network() as net:
        inpa = nengo.Node([0, 0])
        inpb = nengo.Node([0])
        n = nengo.Node(size_in=2)
        pa = nengo.Probe(inpa)
        pb = nengo.Probe(inpb)

    with nengo.Network():
        inp2 = nengo.Node([0, 0])
        p2 = nengo.Probe(inp2)

    with Simulator(net) as sim:
        zeros1 = np.zeros((1, 1, 1))
        zeros2 = np.zeros((1, 1, 2))

        # make sure that valid inputs pass
        sim._check_data({inpa: zeros2, inpb: zeros1}, mode="input")
        sim._check_data({pa: zeros2, pb: zeros1}, mode="target")

        # invalid input objects
        with pytest.raises(ValidationError):
            sim._check_data({inp2: zeros2})
        with pytest.raises(ValidationError):
            sim._check_data({n: zeros2})

        # invalid target object
        with pytest.raises(ValidationError):
            sim._check_data({p2: zeros2}, mode="target")

        # mismatched input data
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2, inpb: np.zeros((2, 1, 1))})
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2, inpb: np.zeros((1, 2, 1))})

        # mismatched target data
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2, pb: np.zeros((2, 1, 1))},
                            mode="target")
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2, pb: np.zeros((1, 2, 1))},
                            mode="target")

        # data that doesn't match explicit target
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2}, n_batch=2)
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2}, n_steps=2)
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2}, n_batch=2, mode="target")
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2}, n_steps=2, mode="target")

        # data with wrong object dimensionality
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros1})
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros1}, mode="target")
