from collections import OrderedDict, defaultdict
import os

import nengo
from nengo.dists import Uniform
from nengo.exceptions import (SimulationError, SimulatorClosed, ReadonlyError,
                              ValidationError)
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import (configure_settings, tensor_layer, dists, TensorNode,
                      DATA_DIR)
from nengo_dl.simulator import SimulationData


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

        # check that sliced input nodes are fed properly as well
        inp2 = nengo.Node([0, 0, 0])
        out2 = nengo.Node(size_in=2)
        nengo.Connection(inp2[:2], out2, synapse=None)
        p2 = nengo.Probe(out2)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        val = np.random.randn(minibatch_size, 50, 3)
        sim.run_steps(50, input_feeds={inp: val, inp2: val})
        assert np.allclose(sim.data[p], val)
        assert np.allclose(sim.data[p2], val[..., :2])


@pytest.mark.parametrize("neurons", (True, False))
def test_train_ff(Simulator, neurons, seed):
    minibatch_size = 4
    n_hidden = 20

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        # note: we have these weird input setup just so that we can test
        # training with two distinct inputs
        inp_a = nengo.Node([0])
        inp_b = nengo.Node([0])
        inp = nengo.Node(size_in=2)
        nengo.Connection(inp_a, inp[0])
        nengo.Connection(inp_b, inp[1])

        ens = nengo.Ensemble(n_hidden + 1, n_hidden,
                             neuron_type=nengo.Sigmoid(tau_ref=1))
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(
            inp, ens.neurons if neurons else ens, transform=dists.Glorot())
        nengo.Connection(
            ens.neurons if neurons else ens, out.neurons,
            transform=dists.Glorot())

        p = nengo.Probe(out.neurons)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed) as sim:
        x = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.train({inp_a: x[..., [0]], inp_b: x[..., [1]]}, {p: y},
                  tf.train.AdamOptimizer(0.01), n_epochs=500)

        sim.check_gradients(atol=5e-5)

        sim.step(input_feeds={inp_a: x[..., [0]], inp_b: x[..., [1]]})

        assert np.allclose(sim.data[p], y, atol=1e-3)


@pytest.mark.parametrize("truncation", (None, 5))
def test_train_recurrent(Simulator, truncation, seed):
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

        sim.train({inp: x}, {p: y}, tf.train.RMSPropOptimizer(1e-3),
                  n_epochs=200, truncation=truncation)

        sim.check_gradients(sim.tensor_graph.build_loss({p: "mse"}))

        sim.run_steps(n_steps, input_feeds={inp: x[:minibatch_size]})

    assert np.sqrt(np.mean((sim.data[p] - y[:minibatch_size]) ** 2)) < (
        0.1 if truncation else 0.05)


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

        ens2 = nengo.Ensemble(n_hidden, 1, neuron_type=nengo.RectifiedLinear())
        nengo.Connection(inp, ens2, synapse=0.01)
        p2 = nengo.Probe(ens2)

    with Simulator(net, minibatch_size=minibatch_size,
                   unroll_simulation=unroll, seed=seed) as sim:
        x = np.ones((minibatch_size, n_steps, 1))
        y = np.zeros((minibatch_size, n_steps, 1))
        z = np.zeros((minibatch_size, n_steps, 1)) + 0.1

        def obj(output, target):
            return tf.reduce_mean((output[:, -1] - 0.5 - target[:, -1]) ** 2)

        sim.train({inp: x}, {p: y, p2: z},
                  tf.train.MomentumOptimizer(1e-2, 0.9),
                  n_epochs=200, objective=obj)

        sim.check_gradients([p, p2])

        sim.run_steps(n_steps, input_feeds={inp: x})

        assert np.allclose(sim.data[p][:, -1], y[:, -1] + 0.5, atol=1e-3)
        assert np.allclose(sim.data[p2][:, -1], z[:, -1] + 0.5, atol=1e-3)


def test_train_sparse(Simulator, seed):
    minibatch_size = 4
    n_hidden = 20

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node([0, 0, 0, 0, 0])
        ens = nengo.Ensemble(n_hidden, 1)
        out = nengo.Node(size_in=2)
        nengo.Connection(inp[[0, 2, 3]], ens.neurons, transform=dists.Glorot())
        nengo.Connection(ens.neurons, out, transform=dists.Glorot())

        p = nengo.Probe(out)

    with Simulator(net, minibatch_size=minibatch_size, unroll_simulation=1,
                   seed=seed) as sim:
        x = np.asarray([[[0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0]],
                        [[1, 0, 0, 0, 0]], [[1, 0, 1, 0, 0]]])
        y = np.asarray([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]])

        sim.train({inp: x}, {p: y},
                  tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov=True),
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

    with Simulator(net, unroll_simulation=2) as sim:
        with pytest.raises(ValidationError):
            sim.train({a: np.ones((1, 1, 1))},
                      {p: np.ones((1, 1, 1))}, None)

        with pytest.raises(ValidationError):
            sim.train({a: np.ones((1, 4, 1))}, {p: np.ones((1, 4, 1))}, None,
                      truncation=3)


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
                                    objective=lambda x, y: tf.constant(2.0)),
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

    with Simulator(net, unroll_simulation=2) as sim:
        with pytest.raises(ValidationError):
            sim.loss({inp: np.ones((1, 1, 1))},
                     {p: np.ones((1, 1, 1))}, None)


def test_generate_inputs(Simulator, seed):
    with nengo.Network() as net:
        proc = nengo.processes.WhiteNoise(seed=seed)
        inp = [nengo.Node([1]), nengo.Node(np.sin), nengo.Node(proc),
               nengo.Node([2]), nengo.Node(nengo.processes.WhiteNoise())]

        p = [nengo.Probe(x) for x in inp]

    with Simulator(net, minibatch_size=2, unroll_simulation=3) as sim:
        feed = sim._generate_inputs({inp[0]: np.zeros((2, 3, 1))}, 3)

        ph = [sim.tensor_graph.input_ph[x] for x in inp]

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

        # check that unseeded process was different in each minibatch item
        assert not np.allclose(feed[ph[-1]][..., 0], feed[ph[-1]][..., 1])


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
        weights_var = [x[0] for x in sim.tensor_graph.base_vars.values()
                       if x[0].get_shape() == (1, 10)][0]
        enc_var = sim.tensor_graph.base_vars[
            sim.tensor_graph.sig_map[sim.model.sig[ens]["encoders"]].key][0]
        weights0, enc0 = sim.sess.run([weights_var, enc_var])
        sim.save_params(os.path.join(str(tmpdir), "train"))
        sim.save_params(os.path.join(str(tmpdir), "local"),
                        include_local=True)

    with pytest.raises(SimulationError):
        sim.save_params(None)
    with pytest.raises(SimulationError):
        sim.load_params(None)

    with nengo.Network(seed=1) as net2:
        inp = nengo.Node([0])
        out = nengo.Node(size_in=1)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        nengo.Connection(ens, out)

        configure_settings(trainable=None)
        net2.config[ens].trainable = False

    with Simulator(net2) as sim:
        weights_var = [x[0] for x in sim.tensor_graph.base_vars.values()
                       if x[0].get_shape() == (1, 10)][0]
        enc_var = sim.tensor_graph.base_vars[
            sim.tensor_graph.sig_map[sim.model.sig[ens]["encoders"]].key][0]
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


def test_model_passing(Simulator, seed):
    # make sure that passing a built model to the Simulator works properly

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(20, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    model = nengo.builder.Model()
    model.build(net)

    ops = [op for op in model.operators]

    with nengo.Simulator(None, model=model, optimize=False) as sim:
        sim.run_steps(10)

    assert ops == model.operators
    canonical = sim.data[p]

    with Simulator(None, model=model) as sim:
        sim.run_steps(10)

    assert ops == model.operators
    assert np.allclose(sim.data[p], canonical)

    # make sure that passing the same model to Simulator twice works
    with Simulator(None, model=model) as sim:
        sim.run_steps(10)
        assert ops == model.operators

    assert ops == model.operators
    assert np.allclose(sim.data[p], canonical)

    # make sure that passing that model back to the reference simulator works
    with nengo.Simulator(None, model=model, optimize=False) as sim:
        sim.run_steps(10)

    assert ops == model.operators
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


def test_tensorboard(Simulator, tmpdir):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Connection(a, b)
        p = nengo.Probe(b)
        p2 = nengo.Probe(c)

    with Simulator(net, tensorboard=str(tmpdir)):
        assert os.path.exists("%s/run_0" % tmpdir)

    # check that the run incrementing works properly
    with Simulator(net, tensorboard=str(tmpdir)):
        assert os.path.exists("%s/run_1" % tmpdir)

    # check that training summaries are output properly
    with Simulator(net, tensorboard=str(tmpdir)) as sim:
        with tf.device("/cpu:0"):
            summ = tf.summary.scalar("step_var", sim.training_step)

        sim.train({a: np.zeros((1, 10, 1))}, {p: np.zeros((1, 10, 1)),
                                              p2: np.zeros((1, 10, 1))},
                  tf.train.GradientDescentOptimizer(0.0),
                  summaries=["loss", b, b.neurons, c, summ],
                  n_epochs=3)

    event_file = os.path.join(
        str(tmpdir), "run_2", os.listdir("%s/run_2" % tmpdir)[0])

    assert os.path.exists(event_file)

    for i, event in enumerate(tf.train.summary_iterator(event_file)):
        if i < 3:
            # metadata stuff
            continue

        assert event.step == i - 2
        tags = [s.tag for s in event.summary.value]
        assert len(tags) == 7
        assert "loss/loss" in tags[0]
        assert "loss/Probe_None_loss" in tags[1]
        assert "loss/Probe_None_loss" in tags[2]
        assert "Ensemble_None_encoders" == tags[3]
        assert "Ensemble.neurons_None_bias" == tags[4]
        assert "Connection_None_weights" == tags[5]
        assert "step_var" == tags[6]

    assert i == 5

    # check for warning if user requests summaries with tensorboard=None
    with pytest.warns(UserWarning):
        with Simulator(net, tensorboard=None) as sim:
            sim.train({a: np.zeros((1, 10, 1))}, {p: np.zeros((1, 10, 1))},
                      tf.train.GradientDescentOptimizer(0.0),
                      summaries=["loss"])

    # check for error on invalid object
    with pytest.raises(SimulationError):
        with Simulator(net, tensorboard=str(tmpdir)) as sim:
            sim.train({a: np.zeros((1, 10, 1))}, {p: np.zeros((1, 10, 1))},
                      tf.train.GradientDescentOptimizer(0.0),
                      summaries=[a])

    # check that nonexistent dir gets created
    with Simulator(net, tensorboard=str(tmpdir.join("new"))):
        assert os.path.exists(str(tmpdir.join("new")))


@pytest.mark.parametrize("mode", ("run", "train"))
@pytest.mark.parametrize("outfile", (None, "tmp.txt"))
def test_profile(Simulator, mode, outfile):
    with nengo.Network() as net:
        a = nengo.Node([0])
        x = nengo.Node(size_in=1)
        nengo.Connection(a, x)
        p = nengo.Probe(x)

    suffix = "" if tf.__version__ < "1.4.0" else "_-1"

    if outfile is None:
        filename = "nengo_dl_profile.json%s" % suffix
    else:
        filename = outfile + suffix
    filename = os.path.join(DATA_DIR, filename)
    if os.path.exists(filename):
        os.remove(filename)
    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with Simulator(net) as sim:
        if outfile is None:
            prof = True
        else:
            prof = {"output": "timeline:outfile=%s" % os.path.join(DATA_DIR,
                                                                   outfile)}

        if mode == "run":
            sim.run_steps(5, profile=prof)
        else:
            sim.train({a: np.zeros((1, 5, 1))}, {p: np.zeros((1, 5, 1))},
                      tf.train.GradientDescentOptimizer(1), profile=prof)

        assert os.path.exists(filename)
        os.remove(filename)


def test_dt_readonly(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        nengo.Probe(a)

    with Simulator(net) as sim:
        with pytest.raises(ReadonlyError):
            sim.dt = 1


def test_probe_data():
    class DummySignal(object):
        minibatched = True

    class DummySimulator(object):
        model = nengo.builder.Model()
        model.sig = defaultdict(lambda: defaultdict(lambda: DummySignal()))

    class DummyProbe(nengo.Probe):
        def __init__(self):
            pass

    sim = DummySimulator()
    a = DummyProbe(add_to_container=False)
    b = DummyProbe(add_to_container=False)
    sim.model.params = OrderedDict(
        {a: [np.zeros((5, 1, 3)), np.ones((5, 1, 3))],
         b: [np.ones((1, 1, 3)), np.zeros((1, 1, 3))]})
    sim.model.probes = (a, b)
    data = SimulationData(sim, True)
    assert data[a].shape == (5, 2, 3)
    assert np.all(data[a][:, 0] == 0)
    assert np.all(data[a][:, 1] == 1)

    data.minibatched = False
    assert data[b].shape == (2, 3)
    assert np.all(data[b][1] == 0)
    assert np.all(data[b][0] == 1)


@pytest.mark.parametrize(
    "pre_val", [0, lambda t: 0, nengo.processes.WhiteNoise(seed=0)])
@pytest.mark.parametrize(
    "post_val", [1, lambda t: 1, nengo.processes.WhiteNoise(seed=1)])
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

    with Simulator(net, minibatch_size=3) as sim:
        zeros1 = np.zeros((3, 1, 1))
        zeros2 = np.zeros((3, 1, 2))

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

        # data with batch size < minibatch_size
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2[[0]]})
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2[[0]]}, mode="target")

        # data with incorrect rank
        with pytest.raises(ValidationError):
            sim._check_data({inpa: zeros2[0]})
        with pytest.raises(ValidationError):
            sim._check_data({pa: zeros2[0]}, mode="target")


def test_matching_node_out(Simulator):
    # make sure that nodes with identical outputs are handled correctly (the
    # outputs will have the same hash, so that can cause some errors)

    with nengo.Network() as net:
        a = nengo.Node(output=nengo.processes.WhiteSignal(1, 10), size_out=2)
        b = nengo.Node(output=nengo.processes.WhiteSignal(1, 10), size_out=2)
        c = nengo.Node(output=nengo.processes.WhiteSignal(1, 10), size_out=3)

        p_a = nengo.Probe(a)
        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    with Simulator(net) as sim:
        sim.run_steps(10)

        assert sim.data[p_a].shape == (10, 2)
        assert sim.data[p_b].shape == (10, 2)
        assert not np.allclose(sim.data[p_a], sim.data[p_b])
        assert sim.data[p_c].shape == (10, 3)


def test_probe_no_data(Simulator):
    with nengo.Network() as net:
        u = nengo.Node([0])
        p = nengo.Probe(u)

    with Simulator(net) as sim:
        pass

    assert sim.data[p] == []


def test_train_state_save(Simulator):
    with nengo.Network() as net:
        u = nengo.Node([1])
        o = nengo.Node(size_in=1)
        nengo.Connection(u, o)
        p = nengo.Probe(u, synapse=0.1)

    with Simulator(net) as sim:
        sim.run_steps(20)

    with Simulator(net) as sim2:
        sim2.run_steps(10)

        sim2.train({u: np.ones((4, 10, 1))}, {p: np.ones((4, 10, 1))},
                   optimizer=tf.train.GradientDescentOptimizer(0))

        sim2.loss({u: np.ones((4, 10, 1))}, {p: np.ones((4, 10, 1))}, "mse")

        sim2.run_steps(10)

    assert np.allclose(sim.data[p], sim2.data[p])


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


def test_simulation_data(Simulator, seed):
    rng = np.random.RandomState(seed)
    N = 17
    d = 2

    gain = rng.uniform(low=1, high=2, size=N)
    bias = rng.uniform(low=0.2, high=1, size=N)
    enc = rng.uniform(-1, 1, size=(N, d))
    enc /= np.linalg.norm(enc, axis=1)[:, None]

    with nengo.Network() as net:
        u = nengo.Node([0] * d)
        a = nengo.Ensemble(N, d, gain=gain, bias=bias, encoders=enc, radius=3)
        b = nengo.Ensemble(N, d, gain=gain, bias=bias, encoders=enc * 2,
                           radius=3, normalize_encoders=False)
        conn0 = nengo.Connection(u, a)
        conn = nengo.Connection(
            a.neurons, b, transform=rng.uniform(-1, 1, size=(d, N)),
            learning_rule_type=nengo.PES())

        p = nengo.Probe(b)

    with Simulator(net) as sim:
        # check dict functions
        assert u in sim.data
        assert a in sim.data
        assert b in sim.data
        assert conn in sim.data
        assert conn0 in sim.data
        assert p in sim.data
        assert net in sim.data
        assert conn.learning_rule in sim.data
        assert len(sim.data) == 9  # 8 objects + probe connection
        for k, k2 in zip(sim.data, sim.data.keys()):
            assert k is k2

        # check gain/bias
        assert np.allclose(gain, sim.data[a].gain)
        assert np.allclose(bias, sim.data[a].bias)

        # check max_rates/intercepts
        max_rates, intercepts = a.neuron_type.max_rates_intercepts(
            gain, bias)
        assert np.allclose(max_rates, sim.data[a].max_rates)
        assert np.allclose(intercepts, sim.data[a].intercepts)

        # check encoders/scaled_encoders
        assert np.allclose(enc, sim.data[a].encoders)
        assert np.allclose(enc * gain[:, None] / a.radius,
                           sim.data[a].scaled_encoders)

        # make sure that the inferences still work with non-normalized encoders
        assert np.allclose(enc * 2, sim.data[b].encoders)
        assert np.allclose(gain, sim.data[b].gain)
        assert np.allclose(bias, sim.data[b].bias)
        assert np.allclose(enc * 2 * gain[:, None] / b.radius,
                           sim.data[b].scaled_encoders)

        # check connection weights
        assert np.allclose(conn.transform, sim.data[conn].weights)

        # check that batch dimension eliminated
        assert sim.data[conn].weights.shape == (d, N)

        # check that values can be updated live
        sig = sim.model.sig[a]['encoders']
        tensor_sig = sim.tensor_graph.sig_map[sig]
        base = sim.tensor_graph.base_vars[tensor_sig.key][0]
        op = tf.assign(base, tf.ones_like(base))
        sim.sess.run(op)

        assert np.allclose(sim.data[a].scaled_encoders, 1)
        assert np.allclose(sim.data[a].gain, np.sqrt(2) * a.radius)
        assert np.allclose(sim.data[a].encoders, 1 / np.sqrt(2))

    # check minibatch dimension moved to front
    with Simulator(net, minibatch_size=3) as sim:
        assert sim.data[conn].weights.shape == (3, d, N)

    # reverts back to init after simulator close, and warns
    with pytest.warns(UserWarning):
        assert np.allclose(sim.data[a].encoders, enc)

    with pytest.raises(ValidationError):
        sim.data[nengo.Ensemble(10, 1, add_to_container=False)]


def test_learning_rate_schedule(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        p = nengo.Probe(b)

    with Simulator(net) as sim:
        vals = [1.0, 0.1, 0.001]
        with tf.device("/cpu:0"):
            l_rate = tf.train.piecewise_constant(
                sim.training_step,
                [tf.constant(4, dtype=tf.int64),
                 tf.constant(9, dtype=tf.int64)],
                vals)
        opt = tf.train.GradientDescentOptimizer(l_rate)

        for i in range(3):
            assert np.allclose(sim.sess.run(l_rate), vals[i])
            sim.train({a: np.zeros((1, 10, 1))}, {p: np.zeros((1, 10, 1))},
                      opt, n_epochs=5)


def test_multiple_objective(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].synapse = None

        a = nengo.Node([0])

        # note: b is configured this way so that the output, and therefore
        # loss, will be equal to the input, so we can control it easily
        b = nengo.Ensemble(
            100, 1, neuron_type=nengo.RectifiedLinear(),
            gain=nengo.dists.Choice([1]), bias=nengo.dists.Choice([0]),
            encoders=nengo.dists.Choice([[1]]))

        c = nengo.Ensemble(10, 1)

        nengo.Connection(a, b)
        nengo.Connection(a, c)

        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    with Simulator(net, unroll_simulation=1) as sim:
        inputs = {a: np.ones((10, 1, 1))}
        targets = {p_b: np.zeros((10, 1, 1)), p_c: np.zeros((10, 1, 1))}
        objective = {
            p_b: lambda x, y: x,
            p_c: lambda x, y: x * 0}

        assert np.allclose(sim.loss(inputs, targets, objective), 1,
                           atol=1e-3)

        b_bias = np.copy(sim.data[b].bias)
        c_bias = sim.data[c].bias
        sim.train(inputs, targets, tf.train.GradientDescentOptimizer(1.0),
                  objective=objective, n_epochs=10)
        assert np.allclose(sim.data[c].bias, c_bias)
        assert not np.allclose(sim.data[b].bias, b_bias)


def test_get_nengo_params(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(12, 3, label="a")
        b = nengo.Ensemble(10, 4, label="b")
        n = nengo.Node([1])
        c = nengo.Connection(a.neurons[:5], b[:2], transform=Uniform(-1, 1),
                             label="c")
        d = nengo.Connection(a, b.neurons, function=lambda x: np.ones(5),
                             transform=Uniform(-1, 1), label="d")
        e = nengo.Connection(n, b, transform=Uniform(-1, 1), label="e")
        f = nengo.Ensemble(5, 1, label="a")
        p = nengo.Probe(b.neurons)

    with Simulator(net, seed=seed) as sim:
        # check that we get an error for non-ensemble/connection objects
        with pytest.raises(ValueError):
            sim.get_nengo_params(n)

        # check that we get an error for duplicate labels
        with pytest.raises(ValueError):
            sim.get_nengo_params([a, f], as_dict=True)

        # check that single objects are returned as single dicts
        params = sim.get_nengo_params(d)
        assert params["transform"] == 1

        params = sim.get_nengo_params([a.neurons, b, c, d, e], as_dict=True)
        sim.run_steps(100)

    with nengo.Network(seed=seed + 1) as net:
        a2 = nengo.Ensemble(12, 3, **params["a"])
        b2 = nengo.Ensemble(10, 4, **params["b"])
        n2 = nengo.Node([1])
        nengo.Connection(a2.neurons[:5], b2[:2], **params["c"])
        nengo.Connection(a2, b2.neurons, **params["d"])
        nengo.Connection(n2, b2, **params["e"])
        p2 = nengo.Probe(b2.neurons)

    with Simulator(net, seed=seed) as sim2:
        sim2.run_steps(100)

        assert np.allclose(sim.data[p], sim2.data[p2])


@pytest.mark.parametrize("progress", (True, False))
def test_progress_bar(Simulator, progress):
    # note: ideally we would capture the stdout and check that output is
    # actually being controlled. but the pytest capturing doesn't work,
    # because it's being printed in a different thread (I think). so we just
    # check that the parameter works without error
    with Simulator(nengo.Network(), progress_bar=progress) as sim:
        sim.run_steps(10, progress_bar=progress)


def test_extra_feeds(Simulator):
    # set up a tensornode that will fail unless a value is fed in for the
    # placeholder
    class NodeFunc(object):
        def pre_build(self, *_):
            self.ph = tf.placeholder_with_default(False, ())

        def __call__(self, t, x):
            with tf.device("/cpu:0"):
                check = tf.Assert(self.ph, [t])
            with tf.control_dependencies([check]):
                y = tf.identity(x)
            return y

    with nengo.Network() as net:
        a = nengo.Node([0])
        b = TensorNode(NodeFunc(), size_in=1, size_out=1)
        nengo.Connection(a, b)
        p = nengo.Probe(b)

    with Simulator(net) as sim:
        with pytest.raises(tf.errors.InvalidArgumentError):
            sim.run_steps(10)
        sim.run_steps(10, extra_feeds={b.tensor_func.ph: True})

        inputs = {a: np.zeros((1, 10, 1))}
        targets = {p: np.zeros((1, 10, 1))}
        with pytest.raises(tf.errors.InvalidArgumentError):
            sim.train(inputs, targets, tf.train.GradientDescentOptimizer(1))
        sim.train(inputs, targets, tf.train.GradientDescentOptimizer(1),
                  extra_feeds={b.tensor_func.ph: True})

        with pytest.raises(tf.errors.InvalidArgumentError):
            sim.loss(inputs, targets, "mse")
        sim.loss(inputs, targets, "mse", extra_feeds={b.tensor_func.ph: True})


@pytest.mark.parametrize("mixed", (False, True))
def test_direct_grads(Simulator, mixed):
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=None)
        p = nengo.Probe(b)

        if mixed:
            c = nengo.Node(size_in=1)
            nengo.Connection(a, c, synapse=None)
            p2 = nengo.Probe(c)

    with Simulator(net, minibatch_size=1) as sim:
        n_steps = 10
        for i in range(10):
            sim.run_steps(n_steps)

            targets = {p: (2. / n_steps) * (sim.data[p] - 2)}
            obj = {p: None}

            if mixed:
                targets[p2] = np.ones((1, n_steps, 1)) * 2
                obj[p2] = "mse"
                assert np.allclose(sim.data[p], sim.data[p2])

            sim.train(
                {a: np.ones((1, n_steps, 1))}, targets,
                tf.train.GradientDescentOptimizer(0.45), objective=obj)
            sim.soft_reset(include_probes=True)

        sim.run_steps(n_steps)
        assert np.allclose(sim.data[p], 2)
        if mixed:
            assert np.allclose(sim.data[p2], 2)
