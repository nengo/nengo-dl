# pylint: disable=missing-docstring

import contextlib
import logging
import pickle
import sys

import nengo
import numpy as np
import pytest
import tensorflow as tf
from nengo.dists import Uniform
from nengo.exceptions import (
    ReadonlyError,
    SimulationError,
    SimulatorClosed,
    ValidationError,
)
from packaging import version
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context

from nengo_dl import Layer, TensorNode, callbacks, configure_settings, dists, utils
from nengo_dl.compat import TFLogFilter, default_transform, eager_enabled
from nengo_dl.simulator import SimulationData
from nengo_dl.tests import dummies


def test_persistent_state(Simulator, seed):
    """Make sure that state is preserved between runs."""

    with nengo.Network(seed=seed) as net:
        configure_settings(dtype="float64")

        inp = nengo.Node(np.sin)
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

    with net:
        configure_settings(use_loop=False)
    with Simulator(net) as sim:
        for _ in range(100 // sim.unroll):
            sim.run_steps(sim.unroll)
        data4 = sim.data[p]

    assert np.allclose(data3, data4)


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
        inp = [
            nengo.Node(output=[0.5]),
            nengo.Node(output=np.sin),
            nengo.Node(output=nengo.processes.WhiteSignal(5, 0.5, seed=seed)),
        ]

        ens = [
            nengo.Ensemble(10, 1, neuron_type=nengo.AdaptiveLIF()),
            nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate()),
            nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise(seed=seed)),
        ]

        nengo.Connection(inp[0], ens[0])
        nengo.Connection(inp[1], ens[1], synapse=None)
        nengo.Connection(inp[2], ens[2], synapse=nengo.Alpha(0.1), transform=[[1], [1]])
        conn = nengo.Connection(ens[0], ens[1], learning_rule_type=nengo.PES())
        nengo.Connection(inp[0], conn.learning_rule)

        ps = [nengo.Probe(e) for e in ens]

    with Simulator(net, minibatch_size=None) as sim:
        probe_data = [[] for _ in ps]
        for _ in range(5):
            sim.run_steps(100)

            for j, p in enumerate(ps):
                probe_data[j] += [sim.data[p]]

            sim.reset()

        probe_data = [np.stack(x, axis=0) for x in probe_data]

    with Simulator(net, minibatch_size=5) as sim:
        sim.run_steps(100)

    for i, p in enumerate(ps):
        assert np.allclose(sim.data[p], probe_data[i], atol=1e-6)


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
        sim.run_steps(50, data={inp: val, inp2: val})
        assert np.allclose(sim.data[p], val)
        assert np.allclose(sim.data[p2], val[..., :2])

        # error for wrong minibatch size
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim.run_steps(10, data={inp: np.zeros((minibatch_size * 2, 10, 3))})
        # error for wrong number of steps
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim.run_steps(10, data={inp: np.zeros((minibatch_size, 15, 3))})


@pytest.mark.parametrize(
    "neurons, use_loop", [(True, False), (False, False), (True, True)]
)
@pytest.mark.training
def test_train_ff(Simulator, neurons, use_loop, seed):
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
        nengo.Connection(inp_a, inp[0], transform=1)
        nengo.Connection(inp_b, inp[1], transform=1)

        ens = nengo.Ensemble(
            n_hidden + 1, n_hidden, neuron_type=nengo.Sigmoid(tau_ref=1)
        )
        out = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(tau_ref=1))
        nengo.Connection(inp, ens.neurons if neurons else ens, transform=dists.Glorot())
        nengo.Connection(
            ens.neurons if neurons else ens, out.neurons, transform=dists.Glorot()
        )

        p = nengo.Probe(out.neurons)

    with Simulator(
        net, minibatch_size=minibatch_size, unroll_simulation=1, seed=seed
    ) as sim:
        x = np.asarray([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.compile(tf.optimizers.Adam(0.01), loss=tf.losses.mse)
        sim.fit({inp_a: x[..., [0]], inp_b: x[..., [1]]}, {p: y}, epochs=500, verbose=0)

        sim.step(data={inp_a: x[..., [0]], inp_b: x[..., [1]]})

        assert np.allclose(sim.data[p], y, atol=2e-3)


@pytest.mark.parametrize("truncation", (None, 5))
@pytest.mark.parametrize("use_loop", (True, False))
@pytest.mark.training
def test_train_recurrent(Simulator, truncation, use_loop, seed):
    batch_size = 100
    minibatch_size = 100
    n_hidden = 30
    n_steps = 10

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(
            n_hidden,
            1,
            neuron_type=nengo.RectifiedLinear(),
            gain=np.ones(n_hidden),
            bias=np.zeros(n_hidden),
        )
        out = nengo.Node(size_in=1)

        nengo.Connection(inp, ens, synapse=None)
        nengo.Connection(ens.neurons, ens.neurons, transform=dists.He(), synapse=0)
        nengo.Connection(ens, out, synapse=None)

        p = nengo.Probe(out)

    kwargs = dict(unroll_simulation=truncation or n_steps) if use_loop else dict()

    with Simulator(net, minibatch_size=minibatch_size, seed=seed, **kwargs) as sim:
        x = np.outer(np.linspace(0, 1, batch_size), np.ones(n_steps))[:, :, None]
        y = np.outer(np.linspace(0, 1, batch_size), np.linspace(0, 1, n_steps))[
            :, :, None
        ]

        sim.compile(tf.optimizers.RMSprop(1e-3), loss=tf.losses.mse)

        if truncation:
            truncation_steps = n_steps // truncation

            # TODO: why doesn't the callback approach work?
            # class ResetCallback(tf.keras.callbacks.Callback):
            #     def on_train_batch_end(self, batch, logs=None):
            #         if batch % truncation_steps == truncation_steps - 1:
            #             sim.reset(
            #                 include_probes=False,
            #                 include_trainable=False,
            #                 include_processes=False,
            #             )
            #
            #     # def on_epoch_end(self, epoch, logs=None):
            #     #     sim.soft_reset()
            #
            # sim.fit(
            #     {inp: np.reshape(x, (-1, truncation, x.shape[2]))},
            #     {p: np.reshape(y, (-1, truncation, y.shape[2]))},
            #     epochs=200,
            #     shuffle=False,
            #     stateful=True,
            #     callbacks=[ResetCallback()],
            #     verbose=2,
            # )

            # TODO: why does this produce non-deterministic results?
            for _ in range(200):
                for j in range(truncation_steps):
                    sim.fit(
                        {inp: x[:, j * truncation : (j + 1) * truncation]},
                        {p: y[:, j * truncation : (j + 1) * truncation]},
                        epochs=1,
                        stateful=True,
                        verbose=0,
                    )

                # note: this works because each epoch is a single minibatch. otherwise
                # we would also have to reset after each minibatch (which is what
                # the callback approach is trying to do)
                sim.reset(
                    include_probes=False,
                    include_trainable=False,
                    include_processes=False,
                )
        else:
            sim.fit({inp: x}, {p: y}, epochs=200, verbose=0)

        loss = sim.evaluate({inp: x}, {p: y})["loss"]
        assert loss < (0.007 if truncation else 0.0025)


@pytest.mark.training
def test_recurrent_gradients(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(
            10, 1, gain=nengo.dists.Choice([0.5]), bias=nengo.dists.Choice([0])
        )
        nengo.Connection(a, b.neurons, transform=nengo.dists.Gaussian(0, 1))
        p = nengo.Probe(b.neurons)

    with Simulator(net) as sim:
        sim.check_gradients(inputs=[np.ones((1, sim.unroll * 2, 1)) * 0.1], outputs=[p])


@pytest.mark.parametrize("unroll", (1, 2))
@pytest.mark.training
def test_train_objective(Simulator, unroll, seed):
    minibatch_size = 1
    n_hidden = 20
    n_steps = 10

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([1])

        ens = nengo.Ensemble(n_hidden, 1, neuron_type=nengo.RectifiedLinear())
        nengo.Connection(inp, ens, synapse=0.01, transform=1)
        p = nengo.Probe(ens)

        ens2 = nengo.Ensemble(n_hidden, 1, neuron_type=nengo.RectifiedLinear())
        nengo.Connection(inp, ens2, synapse=0.01, transform=1)
        p2 = nengo.Probe(ens2)

    with Simulator(
        net, minibatch_size=minibatch_size, unroll_simulation=unroll, seed=seed
    ) as sim:
        x = np.ones((minibatch_size, n_steps, 1))
        y = np.zeros((minibatch_size, n_steps, 1))
        z = np.zeros((minibatch_size, n_steps, 1)) + 0.1

        def obj(target, output):
            return tf.reduce_mean((output[:, -1] - 0.5 - target[:, -1]) ** 2)

        sim.compile(tf.optimizers.SGD(1e-2, momentum=0.9), loss={p: obj, p2: obj})
        sim.fit({inp: x}, {p: y, p2: z}, epochs=200, verbose=0)

        sim.check_gradients(outputs=[p, p2])

        sim.run_steps(n_steps, data={inp: x})

        assert np.allclose(sim.data[p][:, -1], y[:, -1] + 0.5, atol=1e-3)
        assert np.allclose(sim.data[p2][:, -1], z[:, -1] + 0.5, atol=1e-3)


@pytest.mark.training
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

    with Simulator(
        net, minibatch_size=minibatch_size, unroll_simulation=1, seed=seed
    ) as sim:
        x = np.asarray(
            [
                [[0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        y = np.asarray([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]])

        sim.compile(
            tf.optimizers.SGD(0.1, momentum=0.9, nesterov=True), loss=tf.losses.mse
        )
        sim.fit({inp: x}, {p: y}, epochs=500, verbose=0)

        sim.step(data={inp: x})

        assert np.allclose(sim.data[p], y, atol=1e-3)


@pytest.mark.training
def test_train_errors(Simulator):
    net, a, p = dummies.linear_net()

    n_steps = 20
    with Simulator(net) as sim:
        sim.compile(
            optimizer=tf.optimizers.SGD(0),
            loss=lambda y_true, y_pred: tf.losses.mse(y_true[:, -1], y_pred[:, -1]),
        )

        # error for mismatched batch size
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim.fit({a: np.ones((2, n_steps, 1))}, {p: np.ones((1, n_steps, 1))})

        # no error for mismatched n_steps
        sim.fit({a: np.ones((1, n_steps + 5, 1))}, {p: np.ones((1, n_steps, 1))})

    # error when calling train after closing
    with pytest.raises(SimulatorClosed, match="call fit"):
        sim.fit(n_steps=1)

    with Simulator(net, unroll_simulation=2) as sim:
        # error when data n_steps does not match unroll
        with pytest.raises(ValidationError, match="must be evenly divisible"):
            sim.fit({a: np.ones((1, 1, 1))}, {p: np.ones((1, 1, 1))})

        # error when n_steps does not evenly divide by truncation
        # TODO: support truncation
        # with pytest.raises(ValidationError):
        #     sim.train(
        #         {a: np.ones((1, 4, 1)), p: np.ones((1, 4, 1))}, None, truncation=3
        #     )


@pytest.mark.training
def test_train_no_data(Simulator):
    net, _, p = dummies.linear_net()

    with Simulator(net) as sim:

        sim.compile(
            tf.optimizers.SGD(0.1), loss={p: lambda y_true, y_pred: abs(2.0 - y_pred)}
        )
        sim.fit(n_steps=5, y=np.zeros((1, 5, 1)), epochs=10)

        sim.step()
        assert np.allclose(sim.data[p], 2)


def test_evaluate_errors(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([1])
        p = nengo.Probe(inp)

    n_steps = 20
    with Simulator(net, unroll_simulation=1) as sim:
        sim.compile(loss=lambda y_true, y_pred: 1.0)

        # check that valid inputs pass
        assert np.allclose(
            sim.evaluate(y={p: np.zeros((1, n_steps, 1))}, n_steps=n_steps)["loss"], 1
        )

        # error for incorrect n_steps (when explicitly specified)
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim.evaluate(
                {inp: np.ones((1, n_steps + 1, 1))},
                {p: np.ones((1, n_steps, 1))},
                n_steps=n_steps,
            )

        # no error for mismatched n_steps (between inputs and targets)
        sim.evaluate({inp: np.ones((1, n_steps, 1))}, {p: np.ones((1, n_steps + 1, 1))})

        # error for mismatched batch size (between inputs and targets)
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim.evaluate({inp: np.ones((1, n_steps, 1))}, {p: np.ones((2, n_steps, 1))})

        # must specify n_steps if no input data
        with pytest.raises(ValidationError, match="either input data or n_steps"):
            sim.evaluate(y={p: np.zeros((4, n_steps, 1))})

    # error when calling evaluate after close
    with pytest.raises(SimulatorClosed, match="call evaluate"):
        sim.evaluate(y={p: np.zeros((1, n_steps, 1))}, n_steps=n_steps)

    with Simulator(net, unroll_simulation=2) as sim:
        # error when data n_steps does not match unroll
        with pytest.raises(ValidationError, match="must be evenly divisible"):
            sim.evaluate({inp: np.ones((1, 1, 1))}, {p: np.ones((1, 1, 1))})


def test_generate_inputs(Simulator, seed):
    n_steps = 3
    minibatch_size = 2

    with nengo.Network() as net:
        proc = nengo.processes.WhiteNoise(seed=seed)
        inp = [
            nengo.Node([1]),
            nengo.Node(np.sin),
            nengo.Node(proc),
            nengo.Node([2]),
            nengo.Node(nengo.processes.WhiteNoise()),
        ]

        p = [nengo.Probe(x) for x in inp]

    with Simulator(
        net, minibatch_size=minibatch_size, unroll_simulation=n_steps
    ) as sim:
        feed = sim._generate_inputs(
            data={inp[0]: np.zeros((minibatch_size, n_steps, 1))}, n_steps=n_steps
        )

        assert len(sim.tensor_graph.invariant_inputs) == len(inp)
        assert len(feed) == len(inp) + 1

        sim.reset()
        sim.run_steps(n_steps, data={inp[0]: np.zeros((minibatch_size, n_steps, 1))})

        vals = [
            np.zeros((minibatch_size, n_steps, 1)),
            np.tile(np.sin(sim.trange())[None, :, None], (minibatch_size, 1, 1)),
            np.tile(proc.run_steps(3)[None, :, :], (minibatch_size, 1, 1)),
            np.ones((minibatch_size, n_steps, 1)) * 2,
        ]
        for i, x in enumerate(vals):
            assert np.allclose(feed[f"node_{i}" if i > 0 else "node"], x)
            assert np.allclose(sim.data[p[i]], x)

        # check that unseeded process was different in each minibatch item
        assert not np.allclose(
            feed[f"node_{len(inp) - 1}"][0], feed[f"node_{len(inp) - 1}"][1]
        )

        with pytest.raises(SimulationError, match="automatically add n_steps"):
            sim._generate_inputs(data=range(5), n_steps=1)

        with pytest.raises(ValidationError, match="not a valid input"):
            sim._generate_inputs(data={p[0]: np.zeros((minibatch_size, n_steps, 1))})


@pytest.mark.parametrize("include_state", (True, False))
def test_save_load_params(Simulator, include_state, tmp_path):
    def get_network(seed):
        with nengo.Network(seed=seed) as net:
            configure_settings(simplifications=[])

            inp = nengo.Node([0])
            out = nengo.Node(size_in=1)
            ens = nengo.Ensemble(10, 1, neuron_type=dummies.DeterministicLIF())
            nengo.Connection(inp, ens)
            conn = nengo.Connection(ens, out)
            p = nengo.Probe(out)

            configure_settings(trainable=None)
            net.config[ens].trainable = False

        return net, ens, conn, p

    net0, ens0, conn0, p0 = get_network(0)

    with Simulator(net0) as sim_save:
        weights0, enc0, bias0 = sim_save.data.get_params(
            (conn0, "weights"), (ens0, "encoders"), (ens0, "bias")
        )

        sim_save.run_steps(10)

        sim_save.save_params(tmp_path, include_state=include_state)

        sim_save.run_steps(10)

    with pytest.raises(SimulatorClosed):
        sim_save.save_params(None)
    with pytest.raises(SimulatorClosed):
        sim_save.load_params(None)

    net1, ens1, conn1, p1 = get_network(1)

    with Simulator(net1) as sim_load:
        weights1, enc1, bias1 = sim_load.data.get_params(
            (conn1, "weights"), (ens1, "encoders"), (ens1, "bias")
        )
        assert not np.allclose(weights0, weights1)
        assert not np.allclose(enc0, enc1)
        assert not np.allclose(bias0, bias1)

        pre_model = sim_load.keras_model

        sim_load.load_params(tmp_path, include_state=include_state)

        weights2, enc2, bias2 = sim_load.data.get_params(
            (conn1, "weights"), (ens1, "encoders"), (ens1, "bias")
        )

        # check if params match
        assert np.allclose(weights0, weights2)
        assert np.allclose(enc0, enc2)
        assert np.allclose(bias0, bias2)

        # check if a new model was created or one was modified in-place
        assert sim_load.keras_model is pre_model

        # check if things still run correctly
        sim_load.run_steps(10)

        # check if simulation state resumed correctly
        if include_state:
            # state saved, so we should match the point at which that state was saved
            assert np.allclose(sim_load.data[p1], sim_save.data[p0][10:])
        else:
            # state not saved, but other seeded params are, so we should match the first
            # timesteps of `sim_save` (despite the networks not having the same seeds)
            assert np.allclose(sim_load.data[p1], sim_save.data[p0][:10])

    with Simulator(nengo.Network()) as sim:
        with pytest.raises(SimulationError, match="!= number of variables"):
            sim.load_params(tmp_path)


def test_save_load_params_deprecation(Simulator, tmp_path):
    with nengo.Network() as net:
        a = nengo.Node([1])
        p = nengo.Probe(a, synapse=0.1)

    with Simulator(net) as sim0:
        sim0.run_steps(5)

        with pytest.warns(
            DeprecationWarning, match="include_non_trainable is deprecated"
        ):
            sim0.save_params(tmp_path / "tmp", include_non_trainable=True)

        sim0.run_steps(5)

    with Simulator(net) as sim1:
        with pytest.warns(
            DeprecationWarning, match="include_non_trainable is deprecated"
        ):
            sim1.load_params(tmp_path / "tmp", include_non_trainable=True)

        sim1.run_steps(5)

    assert np.allclose(sim0.data[p][-5:], sim1.data[p])


def test_model_passing(Simulator, seed):
    # make sure that passing a built model to the Simulator works properly

    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([1])
        ens = nengo.Ensemble(20, 1)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    model = nengo.builder.Model()
    model.build(net)

    ops = model.operators.copy()

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


@pytest.mark.parametrize("device", ["/cpu:0", "/gpu:0", None])
def test_devices(Simulator, device, seed, caplog):
    if device == "/gpu:0" and not utils.tf_gpu_installed:
        pytest.skip("This test requires GPU support")

    caplog.set_level(logging.INFO)

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

        if device == "/cpu:0":
            assert "Running on CPU" in caplog.text
        elif device == "/gpu:0":
            assert "Running on GPU" in caplog.text
        elif utils.tf_gpu_installed:
            assert "Running on CPU/GPU" in caplog.text
        else:
            # device is None but gpu not installed
            assert "Running on CPU" in caplog.text


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


@pytest.mark.training
def test_tensorboard(Simulator, tmp_path):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate())
        c = nengo.Connection(a, b, transform=1)
        c0 = nengo.Connection(a, b)
        p = nengo.Probe(b)
        p2 = nengo.Probe(c)

    # check that training summaries are output properly
    n_epochs = 3
    with Simulator(net) as sim:

        log_dir = tmp_path / "a_run"

        sim.compile(
            tf.optimizers.SGD(0.0),
            loss={p: lambda y_true, y_pred: y_pred, p2: tf.losses.mse},
        )

        sim.fit(
            {a: np.zeros((1, 10, 1))},
            {p: np.zeros((1, 10, 1)), p2: np.zeros((1, 10, 1))},
            epochs=n_epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0),
                callbacks.NengoSummaries(
                    log_dir=log_dir / "nengo",
                    sim=sim,
                    objects=[b, b.neurons, c],
                ),
            ],
        )

    # look up name of event file
    event_dir = (
        log_dir / "train"
        if version.parse(tf.__version__) < version.parse("2.3.0rc0") or eager_enabled()
        else log_dir
    )
    event_file = [
        x
        for x in event_dir.glob("events.out.tfevents*")
        if x.suffix != ".profile-empty"
    ]
    assert len(event_file) == 1
    event_file = event_file[0]
    assert event_file.exists()

    summaries = ["epoch_loss", "epoch_probe_loss", "epoch_probe_1_loss"]
    # metadata stuff in event file
    meta_steps = (
        2
        if version.parse(tf.__version__) < version.parse("2.3.0rc0") or eager_enabled()
        else 3
    )
    with contextlib.suppress() if eager_enabled() else context.eager_mode():
        for i, record in enumerate(tf.data.TFRecordDataset(str(event_file))):
            event = event_pb2.Event.FromString(record.numpy())

            if i >= meta_steps:
                curr_step = (i - meta_steps) // len(summaries)
                assert event.step == curr_step

                assert (
                    event.summary.value[0].tag
                    == summaries[(i - meta_steps) % len(summaries)]
                )

    assert i == len(summaries) * n_epochs + (  # pylint: disable=undefined-loop-variable
        meta_steps - 1
    )

    # look up name of event file
    event_file = list((log_dir / "nengo").glob("*.v2"))
    assert len(event_file) == 1
    event_file = event_file[0]
    assert event_file.exists()

    summaries = [
        "Ensemble_None_encoders",
        "Ensemble.neurons_None_bias",
        "Connection_None_weights",
    ]
    with contextlib.suppress() if eager_enabled() else context.eager_mode():
        for i, record in enumerate(tf.data.TFRecordDataset(str(event_file))):
            event = event_pb2.Event.FromString(record.numpy())

            if i < 1:
                # metadata stuff
                continue

            curr_step = (i - 1) // len(summaries)
            assert event.step == curr_step
            assert event.summary.value[0].tag == summaries[(i - 1) % len(summaries)]

    assert i == len(summaries) * n_epochs  # pylint: disable=undefined-loop-variable

    # check for error on invalid object
    with pytest.raises(ValidationError, match="Unknown summary object"):
        callbacks.NengoSummaries(log_dir=log_dir / "nengo", sim=sim, objects=[a])

    if version.parse(nengo.__version__) >= version.parse("3.1.0"):
        with pytest.raises(ValidationError, match="does not have any weights"):
            callbacks.NengoSummaries(log_dir=log_dir / "nengo", sim=sim, objects=[c0])


@pytest.mark.parametrize("mode", ("predict", "train"))
@pytest.mark.training
def test_profile(Simulator, mode, tmp_path, pytestconfig):
    if (
        pytestconfig.getoption("--graph-mode")
        and version.parse(tf.__version__) >= version.parse("2.4.0rc0")
        and mode == "predict"
    ):
        pytest.skip(
            "TensorFlow bug, see https://github.com/tensorflow/tensorflow/issues/44563"
        )

    net, a, p = dummies.linear_net()

    with Simulator(net) as sim:
        # note: TensorFlow bug if using profile_batch=1, see
        # https://github.com/tensorflow/tensorflow/issues/37543
        callback = callbacks.TensorBoard(
            log_dir=str(tmp_path / "profile"), profile_batch=2
        )

        if mode == "predict":
            sim.predict(np.zeros((2, 5, 1)), callbacks=[callback])
        else:
            sim.compile(tf.optimizers.SGD(1), loss=tf.losses.mse)
            sim.fit(
                {a: np.zeros((2, 5, 1))}, {p: np.zeros((2, 5, 1))}, callbacks=[callback]
            )

        path = tmp_path / "profile"
        if version.parse(tf.__version__) < version.parse("2.3.0rc0"):
            path /= "train"
        assert path.exists()


def test_dt_readonly(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        nengo.Probe(a)

    with Simulator(net) as sim:
        with pytest.raises(ReadonlyError):
            sim.dt = 1


def test_probe_data():
    sim = dummies.Simulator()
    a = dummies.Probe(add_to_container=False)
    b = dummies.Probe(add_to_container=False)
    sim.model.params = {
        a: [np.zeros((5, 1, 3)), np.ones((5, 1, 3))],
        b: [np.ones((1, 1, 3)), np.zeros((1, 1, 3))],
    }
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
    "pre_val", [1, lambda t: 1, nengo.processes.Piecewise({0: [1]})]
)
@pytest.mark.parametrize(
    "post_val", [2, lambda t: 2, nengo.processes.Piecewise({0: [2]})]
)
def test_node_output_change(Simulator, pre_val, post_val, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(pre_val)
        p = nengo.Probe(inp)

    with Simulator(net, unroll_simulation=1) as sim:
        sim.step()
        inp.output = post_val
        sim.step()

        assert np.allclose(sim.data[p], 1.0)

        sim.reset()
        sim.step()

        assert np.allclose(sim.data[p], 1.0)


@pytest.mark.training
def test_check_gradients_error(Simulator):
    # check_gradients detects nans in gradient
    with nengo.Network() as net:
        x = nengo.Node([0])
        y = Layer(lambda x: 1 / x)(x)
        nengo.Probe(y)

    with Simulator(net) as sim:
        with pytest.raises(SimulationError, match="NaNs detected"):
            sim.check_gradients()

    # check_gradients detects errors in gradient (in this case caused by the
    # fact that nengo.Triangle doesn't have a TensorFlow implementation)
    with nengo.Network() as net:
        x = nengo.Node([0])
        nengo.Probe(x, synapse=nengo.Triangle(0.1))

    with Simulator(net) as sim:
        with pytest.raises(SimulationError, match="Gradient check failed"):
            sim.check_gradients()


def test_check_data(Simulator):
    with nengo.Network() as net:
        inpa = nengo.Node([0, 0], label="inpa")
        inpb = nengo.Node([0], label="inpb")
        nengo.Node(size_in=2, label="n")
        nengo.Probe(inpa, label="pa")
        nengo.Probe(inpb, label="pb")
        nengo.Ensemble(10, 1)

    with nengo.Network():
        inp2 = nengo.Node([0, 0], label="inp2")
        nengo.Probe(inp2, label="p2")

    with Simulator(net, minibatch_size=3, unroll_simulation=1) as sim:
        zeros1 = np.zeros((3, 1, 1))
        zeros2 = np.zeros((3, 1, 2))
        n_steps = np.ones((3, 1))

        # make sure that valid inputs pass
        sim._check_data({"inpa": zeros2, "inpb": zeros1, "n_steps": n_steps})
        sim._check_data({"pa": zeros2, "pb": zeros1}, nodes=False)

        # invalid input objects
        with pytest.raises(ValidationError, match="not a valid node name"):
            sim._check_data({"inp2": zeros2, "n_steps": n_steps})
        with pytest.raises(ValidationError, match="not a valid node name"):
            sim._check_data({"n": zeros2, "n_steps": n_steps})

        # invalid target object
        with pytest.raises(ValidationError, match="not a valid probe name"):
            sim._check_data({"p2": zeros2}, nodes=False)

        # mismatched input data
        with pytest.raises(ValidationError, match="different batch size"):
            sim._check_data(
                {
                    "inpa": zeros2,
                    "inpb": np.zeros((sim.minibatch_size * 2, 1, 1)),
                    "n_steps": n_steps,
                }
            )
        with pytest.raises(ValidationError, match="different number of timesteps"):
            sim._check_data(
                {"inpa": zeros2, "inpb": np.zeros((3, 2, 1)), "n_steps": n_steps}
            )

        # mismatched target data
        with pytest.raises(ValidationError, match="different batch size"):
            sim._check_data(
                {"pa": zeros2, "pb": np.zeros((sim.minibatch_size * 2, 1, 1))},
                nodes=False,
            )
        # no error for different n_steps
        sim._check_data({"pa": zeros2, "pb": np.zeros((3, 2, 1))}, nodes=False)

        # data that doesn't match explicit validation value
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim._check_data({"inpa": zeros2, "n_steps": np.ones((2, 1))}, batch_size=2)
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim._check_data(
                {"inpa": zeros2, "n_steps": np.ones_like(n_steps) * 2}, n_steps=2
            )
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim._check_data({"pa": zeros2}, batch_size=2, nodes=False)
        with pytest.raises(ValidationError, match="does not match expected size"):
            sim._check_data({"pa": zeros2}, n_steps=2, nodes=False)

        # data with batch size < minibatch_size
        with pytest.raises(ValidationError, match=r"Batch size of data \(1\)"):
            sim._check_data({"inpa": zeros2[[0]], "n_steps": n_steps})
        with pytest.raises(ValidationError, match=r"Batch size of data \(1\)"):
            sim._check_data({"pa": zeros2[[0]]}, nodes=False)

        # data with incorrect rank
        with pytest.raises(ValidationError, match="should have rank 3"):
            sim._check_data({"inpa": zeros2[0], "n_steps": n_steps})
        with pytest.raises(ValidationError, match="should have rank 3"):
            sim._check_data({"pa": zeros2[0]}, nodes=False)

        # no n_steps
        with pytest.raises(ValidationError, match="Must specify 'n_steps'"):
            sim._check_data({})

        # n_steps wrong shape
        with pytest.raises(ValidationError, match="wrong shape"):
            sim._check_data({"n_steps": np.asarray(10)})

        # non-constant n_steps
        with pytest.raises(ValidationError, match="have the same value"):
            sim._check_data({"n_steps": np.asarray([[0], [1], [2]])})

        # n_steps mismatch
        with pytest.raises(ValidationError, match="does not match"):
            sim._check_data({"n_steps": np.asarray([[10], [10], [10]])}, n_steps=5)

    with net:
        configure_settings(use_loop=False)

    with Simulator(net, unroll_simulation=1) as sim:
        # n_steps doesn't match loop length
        with pytest.raises(ValidationError, match="use_loop=False"):
            sim._check_data({"n_steps": np.asarray([[2]])}, n_steps=2)


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


@pytest.mark.training
def test_train_state_save(Simulator):
    with nengo.Network() as net:
        u = nengo.Node([1])
        o = nengo.Node(size_in=1)
        nengo.Connection(u, o)
        p = nengo.Probe(o, synapse=0.1)

    with Simulator(net) as sim:
        sim.run_steps(20)

    with Simulator(net) as sim2:
        sim2.run_steps(10)

        sim2.compile(tf.optimizers.SGD(0), loss=tf.losses.mse)
        sim2.fit({u: np.ones((4, 10, 1))}, {p: np.ones((4, 10, 1))})

        sim2.evaluate({u: np.ones((4, 10, 1))}, {p: np.ones((4, 10, 1))})

        sim2.run_steps(10)

    assert np.allclose(sim.data[p], sim2.data[p])


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
        b = nengo.Ensemble(
            N,
            d,
            gain=gain,
            bias=bias,
            encoders=enc * 2,
            radius=3,
            normalize_encoders=False,
        )
        c = nengo.Ensemble(N, d, neuron_type=nengo.Direct())
        conn0 = nengo.Connection(u, a)
        conn = nengo.Connection(
            a.neurons,
            b,
            transform=rng.uniform(-1, 1, size=(d, N)),
            learning_rule_type=nengo.PES(),
        )

        p = nengo.Probe(b)

    with Simulator(net) as sim:
        # check dict functions
        assert u in sim.data
        assert a in sim.data
        assert b in sim.data
        assert c in sim.data
        assert conn in sim.data
        assert conn0 in sim.data
        assert p in sim.data
        assert net in sim.data
        assert conn.learning_rule in sim.data
        assert len(sim.data) == 10  # 9 objects + probe connection
        for k, k2 in zip(sim.data, sim.data.keys()):
            assert k is k2

        # check gain/bias
        assert np.allclose(gain, sim.data[a].gain)
        assert np.allclose(bias, sim.data[a].bias)

        # check max_rates/intercepts
        max_rates, intercepts = a.neuron_type.max_rates_intercepts(gain, bias)
        assert np.allclose(max_rates, sim.data[a].max_rates)
        assert np.allclose(intercepts, sim.data[a].intercepts)

        # check encoders/scaled_encoders
        assert np.allclose(enc, sim.data[a].encoders)
        assert np.allclose(enc * gain[:, None] / a.radius, sim.data[a].scaled_encoders)

        # make sure that the inferences still work with non-normalized encoders
        assert np.allclose(enc * 2, sim.data[b].encoders)
        assert np.allclose(gain, sim.data[b].gain)
        assert np.allclose(bias, sim.data[b].bias)
        assert np.allclose(
            enc * 2 * gain[:, None] / b.radius, sim.data[b].scaled_encoders
        )

        # check connection weights
        transform = conn.transform.init
        assert np.allclose(transform, sim.data[conn].weights)

        # check that batch dimension eliminated
        assert sim.data[conn].weights.shape == (d, N)

        # check that values can be updated live
        sig = sim.model.sig[a]["encoders"]
        tensor_sig = sim.tensor_graph.signals[sig]

        base = sim.tensor_graph.base_params[tensor_sig.key]
        tf.keras.backend.set_value(
            base, np.ones(base.shape, dtype=base.dtype.as_numpy_dtype())
        )

        assert np.allclose(sim.data[a].scaled_encoders, 1)
        assert np.allclose(sim.data[a].gain, np.sqrt(2) * a.radius)
        assert np.allclose(sim.data[a].encoders, 1 / np.sqrt(2))

        # check that direct mode ensemble parameters are handled correctly
        assert np.allclose(sim.data[c].encoders, np.eye(d))
        assert sim.data[c].gain is None
        assert sim.data[c].bias is None

    # check minibatch dimension moved to front
    with Simulator(net, minibatch_size=3) as sim:
        assert sim.data[conn].weights.shape == (3, d, N)

    # reverts back to init after simulator close, and warns
    with pytest.warns(UserWarning):
        assert np.allclose(sim.data[a].encoders, enc)

    with pytest.raises(ValidationError):
        _ = sim.data[nengo.Ensemble(10, 1, add_to_container=False)]


@pytest.mark.training
def test_learning_rate_schedule(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1, neuron_type=nengo.LIFRate())
        nengo.Connection(a, b)
        p = nengo.Probe(b)

    with Simulator(net) as sim:
        vals = [1.0, 0.1, 0.001]

        def schedule(epoch, lr):
            if epoch >= 9:
                return vals[2]
            if epoch >= 4:
                return vals[1]
            return vals[0]

        sim.compile(tf.optimizers.SGD(vals[0]), loss=tf.losses.mse)

        for i in range(3):
            assert np.allclose(
                tf.keras.backend.get_value(sim.keras_model.optimizer.lr), vals[i]
            )
            sim.fit(
                {a: np.zeros((1, 10, 1))},
                {p: np.zeros((1, 10, 1))},
                epochs=(i + 1) * 5,
                initial_epoch=i * 5,
                callbacks=[tf.keras.callbacks.LearningRateScheduler(schedule)],
            )


@pytest.mark.training
def test_multiple_objective(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].synapse = None

        a = nengo.Node([0])

        # note: b is configured this way so that the output, and therefore
        # loss, will be equal to the input, so we can control it easily
        b = nengo.Ensemble(
            100,
            1,
            neuron_type=nengo.RectifiedLinear(),
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
            encoders=nengo.dists.Choice([[1]]),
        )

        c = nengo.Ensemble(10, 1)

        nengo.Connection(a, b)
        nengo.Connection(a, c)

        p_b = nengo.Probe(b)
        p_c = nengo.Probe(c)

    with Simulator(net, unroll_simulation=1) as sim:
        data = (
            {a: np.ones((10, 1, 1))},
            {p_b: np.zeros((10, 1, 1)), p_c: np.zeros((10, 1, 1))},
        )
        objective = {
            p_b: lambda y_true, y_pred: y_pred,
            p_c: lambda y_true, y_pred: y_pred * 0,
        }

        sim.compile(tf.optimizers.SGD(1.0), loss=objective)

        loss = sim.evaluate(*data)
        assert np.allclose(loss["loss"], 1, atol=1e-3)

        b_bias = np.copy(sim.data[b].bias)
        c_bias = sim.data[c].bias
        sim.fit(*data, epochs=10)
        assert np.allclose(sim.data[c].bias, c_bias)
        assert not np.allclose(sim.data[b].bias, b_bias)


def test_get_nengo_params(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].neuron_type = dummies.DeterministicLIF()

        a = nengo.Ensemble(12, 3, label="a")
        b = nengo.Ensemble(10, 4, label="b", radius=2)
        n = nengo.Node([1])
        c = nengo.Connection(a.neurons[:5], b[:2], transform=Uniform(-1, 1), label="c")
        d = nengo.Connection(
            a,
            b.neurons,
            function=lambda x: np.ones(5),
            transform=Uniform(-1, 1),
            label="d",
        )
        e = nengo.Connection(n, b, transform=Uniform(-1, 1), label="e")
        f = nengo.Ensemble(5, 1, label="a")
        g = nengo.Ensemble(11, 1, neuron_type=nengo.Direct(), label="g")
        h = nengo.Connection(
            a.neurons,
            b,
            transform=nengo.Convolution(1, (2, 2, 3), padding="same"),
            label="h",
        )
        p = nengo.Probe(b.neurons)

    with Simulator(net, seed=seed) as sim:
        # check that we get an error for non-ensemble/connection objects
        with pytest.raises(ValueError):
            sim.get_nengo_params(n)

        # check that we get an error for duplicate labels
        with pytest.raises(ValueError):
            sim.get_nengo_params([a, f], as_dict=True)

        # check that we get an error for direct ensembles
        with pytest.raises(ValueError):
            sim.get_nengo_params(g)

        # check that single objects are returned as single dicts
        params = sim.get_nengo_params(d)
        assert params["transform"] is default_transform

        fetches = [a.neurons, b, c, d, e, h]

        params = sim.get_nengo_params(fetches, as_dict=True)
        sim.run_steps(100)

    with nengo.Network(seed=seed + 1) as net:
        net.config[nengo.Ensemble].neuron_type = dummies.DeterministicLIF()

        a2 = nengo.Ensemble(12, 3, **params["a"])
        b2 = nengo.Ensemble(10, 4, radius=2, **params["b"])
        n2 = nengo.Node([1])
        nengo.Connection(a2.neurons[:5], b2[:2], **params["c"])
        nengo.Connection(a2, b2.neurons, **params["d"])
        nengo.Connection(n2, b2, **params["e"])
        nengo.Connection(a2.neurons, b2, **params["h"])
        p2 = nengo.Probe(b2.neurons)

    with Simulator(net, seed=seed) as sim2:
        sim2.run_steps(100)

        assert np.allclose(sim.data[p], sim2.data[p2])


@pytest.mark.parametrize("use_scipy", (True, False))
def test_get_nengo_params_sparse(Simulator, use_scipy, rng):
    if use_scipy:
        scipy_sparse = pytest.importorskip("scipy.sparse")

    with nengo.Network() as net:
        node0 = nengo.Node([1] * 10)
        node1 = nengo.Node(size_in=10)
        data = rng.uniform(-1, 1, size=10)
        if use_scipy:
            conn = nengo.Connection(
                node0,
                node1,
                synapse=None,
                transform=nengo.Sparse(
                    (10, 10),
                    init=scipy_sparse.coo_matrix(
                        (data, (np.arange(10), np.arange(10)))
                    ),
                ),
            )
        else:
            conn = nengo.Connection(
                node0,
                node1,
                synapse=None,
                transform=nengo.Sparse(
                    (10, 10), indices=[[i] * 2 for i in range(10)], init=data
                ),
            )

        p0 = nengo.Probe(node1)

    with Simulator(net) as sim0:
        params = sim0.get_nengo_params(conn)
        assert np.allclose(params["transform"].init.data, data)

        sim0.run_steps(10)

    with nengo.Network() as net:
        node0 = nengo.Node([1] * 10)
        node1 = nengo.Node(size_in=10)
        nengo.Connection(node0, node1, synapse=None, **params)
        p1 = nengo.Probe(node1)

    with Simulator(net) as sim1:
        sim1.run_steps(10)

    assert np.allclose(sim0.data[p0], sim1.data[p1])


@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.training
def test_direct_grads(Simulator, mixed):
    net, a, p = dummies.linear_net()

    if mixed:
        with net:
            c = nengo.Node(size_in=1)
            nengo.Connection(a, c, synapse=None, transform=1)
            p2 = nengo.Probe(c)

    with Simulator(net, minibatch_size=1) as sim:
        n_steps = 10

        def direct(y_true, y_pred):
            return tf.reduce_sum(y_true * y_pred)

        obj = {p: direct}
        if mixed:
            obj[p2] = tf.losses.mse

        sim.compile(tf.optimizers.SGD(0.45), loss=obj)

        for _ in range(10):
            sim.run_steps(n_steps)

            data = {p: (2.0 / n_steps) * (sim.data[p] - 2)}

            if mixed:
                data[p2] = np.ones((1, n_steps, 1)) * 2
                assert np.allclose(sim.data[p], sim.data[p2])

            sim.fit({a: np.ones((1, n_steps, 1))}, data)
            sim.reset(include_trainable=False, include_processes=False)

        sim.run_steps(n_steps)
        assert np.allclose(sim.data[p], 2)
        if mixed:
            assert np.allclose(sim.data[p2], 2)


@pytest.mark.training
def test_non_differentiable(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Node(lambda t, x: x, size_in=1)
        c = nengo.Connection(a, b, transform=1)
        p = nengo.Probe(b)

    with Simulator(net) as sim:
        w0 = sim.data[c].weights
        sim.compile(tf.optimizers.SGD(100), loss=tf.losses.mse)
        sim.fit({a: np.ones((1, 10, 1))}, {p: np.ones((1, 10, 1))})

        # TODO: find another way to detect non-differentiable elements in graph
        # note: the challenge is that our stateful ops tend to mask the
        # backpropagating None gradients, because the overwritten parts of
        # the variable end up with a gradient of zero (regardless of what the
        # output gradient is)
        assert np.allclose(sim.data[c].weights, w0)


@pytest.mark.parametrize(
    "net",
    (
        nengo.networks.BasalGanglia(4),
        nengo.networks.CircularConvolution(32, 4),
        nengo.networks.Oscillator(0.01, 10, 100),
        nengo.networks.InputGatedMemory(32, 4),
    ),
)
def test_freeze_network(Simulator, net):
    with Simulator(net) as sim:
        sim.run(0.1)
        sim.freeze_params(net)

    with nengo.Simulator(net) as sim2:
        sim2.run(0.1)

    for p in net.all_probes:
        assert np.allclose(sim.data[p], sim2.data[p])


def test_freeze_obj(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(10, 1)
        b = nengo.Node(size_in=1)
        c = nengo.Connection(a, b)
        p = nengo.Probe(b)

    with nengo.Network(seed=seed + 1):
        d = nengo.Ensemble(10, 1)

    with Simulator(net) as sim:
        # check for error with object outside network
        with pytest.raises(ValueError):
            sim.freeze_params(d)

        # check for error with wrong object type
        with pytest.raises(TypeError):
            sim.freeze_params(p)

        sim.freeze_params([a, c])
        sim.run_steps(10)

    with nengo.Simulator(net) as sim2:
        sim2.run_steps(10)

    assert np.allclose(sim.data[p], sim2.data[p])

    # check that we get an error if the simulator is closed
    with pytest.raises(SimulatorClosed):
        sim.freeze_params(net)


@pytest.mark.training
def test_freeze_train(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, transform=1, synapse=None)
        p = nengo.Probe(b)

    with Simulator(net, unroll_simulation=1) as sim:
        sim.step()
        assert np.allclose(sim.data[p], 1)

        sim.compile(tf.optimizers.SGD(0.5), loss=tf.losses.mse)
        sim.fit({a: np.ones((1, 1, 1))}, {p: np.ones((1, 1, 1)) * 2}, epochs=10)

        sim.step()
        assert np.allclose(sim.data[p][-1], 2)

        sim.freeze_params(net)

    with nengo.Simulator(net) as sim:
        sim.step()
        assert np.allclose(sim.data[p], 2)


@pytest.mark.parametrize("neuron_type", (nengo.SpikingRectifiedLinear(), nengo.LIF()))
def test_inference_only(Simulator, neuron_type, seed):
    with nengo.Network(seed=seed) as net:
        configure_settings(inference_only=False)

        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1, neuron_type=neuron_type)
        c = nengo.Connection(a, b, synapse=None, transform=1)
        p = nengo.Probe(b)

    with Simulator(net) as sim:
        sim.run(0.1)

        assert sim.model.sig[c]["weights"].trainable

    with net:
        configure_settings(inference_only=True)

    with Simulator(net) as sim2:
        sim2.run(0.1)

        # check that inference-only mode produces the same output
        assert np.allclose(sim.data[p], sim2.data[p])

        # check that parameters aren't trainable
        assert not sim2.model.sig[c]["weights"].trainable

        # validation checks (can't do train/gradients in inference-only mode)
        with pytest.raises(SimulationError, match="inference_only=True"):
            sim2.fit(n_steps=10)

        with pytest.raises(SimulationError, match="inference_only=True"):
            sim2.check_gradients()


@pytest.mark.training
def test_synapse_warning(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Connection(a, b, synapse=1)
        p = nengo.Probe(b)
        p2 = nengo.Probe(b)

    def does_warn(n_steps=1, as_dict=True):
        with Simulator(net, unroll_simulation=1) as sim:
            sim.compile(tf.optimizers.SGD(0), loss={p: tf.losses.mse})
            with pytest.warns(None) as rec:
                if as_dict:
                    sim.fit(
                        {a: np.zeros((1, n_steps, 1))}, {p: np.zeros((1, n_steps, 1))}
                    )
                else:
                    sim.fit(n_steps=n_steps)
        return any("contains synaptic filters" in str(w.message) for w in rec)

    # warning from connection
    assert does_warn()

    # warning from probe
    c.synapse = None
    p.synapse = 1
    assert does_warn()

    # no warning for >1 step training
    assert not does_warn(n_steps=2)

    # no warning from non-target probe
    p.synapse = None
    p2.synapse = 1
    assert not does_warn()

    # warning when explicitly specifying n_steps
    c.synapse = 1
    assert does_warn(as_dict=True)


@pytest.mark.training
def test_concat_hang(Simulator, pytestconfig):
    with nengo.Network() as net:
        a = nengo.Node([0])
        x = nengo.Node(size_in=1)
        nengo.Connection(a, x)
        p = nengo.Probe(x)

    with Simulator(net) as sim:
        sim.compile(tf.optimizers.SGD(1), loss=tf.losses.mse)
        sim.fit({a: np.zeros((1, 5, 1))}, {p: np.zeros((1, 5, 1))})


def test_tf_seed(Simulator, seed):
    with nengo.Network() as net:
        a = TensorNode(lambda t: tf.random.uniform((1, 1), dtype=t.dtype))
        p = nengo.Probe(a)

    with Simulator(net, seed=seed) as sim0:
        sim0.step()

    if not eager_enabled():
        # this is necessary to reset the graph seed
        tf.keras.backend.clear_session()

    with Simulator(net, seed=seed) as sim1:
        sim1.step()

    assert np.allclose(sim0.data[p], sim1.data[p])


@pytest.mark.xfail(reason="TensorFlow does not support resetting RNG")
def test_tf_rng_reset(Simulator, seed):
    # TODO: support this using tf.random.experimental; see
    #  https://github.com/nengo/nengo-dl/issues/98

    with nengo.Network() as net:
        a = TensorNode(lambda t: tf.random.uniform((1, 1), dtype=t.dtype))
        p = nengo.Probe(a)

    with Simulator(net, seed=seed) as sim:
        sim.step()

        data = sim.data[p]

        sim.reset(seed=seed + 1)
        sim.step()
        assert not np.allclose(sim.data[p], data)

        sim.reset(seed=seed)
        sim.step()
        assert np.allclose(sim.data[p], data)


def test_pickle_error(Simulator):
    with pytest.raises(NotImplementedError, match="does not support pickling"):
        pickle.dumps(Simulator(nengo.Network()))


def test_tensorflow_gpu_warning(Simulator, pytestconfig):
    with pytest.warns(None) as recwarns:
        with Simulator(nengo.Network()):
            pass

    recwarns = [w for w in recwarns if "No GPU support detected" in str(w.message)]

    assert len(recwarns) == (
        1
        if not utils.tf_gpu_installed and pytestconfig.getoption("--device") is None
        else 0
    )


def test_train_loss_deprecation(Simulator):
    with Simulator(dummies.linear_net()[0]) as sim:
        with pytest.raises(SimulationError, match="train has been deprecated"):
            sim.train()

        with pytest.raises(SimulationError, match="loss has been deprecated"):
            sim.loss()


def test_io_names(Simulator):
    with nengo.Network() as net:
        nodes = [
            nengo.Node([0]),
            nengo.Node([0]),
            nengo.Node([0], label="aa"),
            nengo.Node([0], label="aa"),
            nengo.Node([0], label="Aa"),
            nengo.Node([0], label="aA"),
            nengo.Node([0]),
        ]

        for n in nodes:
            nengo.Probe(n, label=None if n.label is None else "p_" + n.label)

    with Simulator(net) as sim:
        assert sim.keras_model.input_names == [
            "node",
            "node_1",
            "aa",
            "aa_1",
            "Aa_2",
            "aA_3",
            "node_2",
            "n_steps",
        ]
        assert sim.keras_model.output_names == [
            "probe",
            "probe_1",
            "p_aa",
            "p_aa_1",
            "p_Aa_2",
            "p_aA_3",
            "probe_2",
            "steps_run",
        ]
        sim.step()

        with pytest.raises(ValidationError, match="not an input Node"):
            sim.get_name(nengo.Node([0], add_to_container=False))

        with pytest.raises(ValidationError, match="from a different network"):
            sim.get_name(nengo.Probe(nodes[0], add_to_container=False))

        with pytest.raises(ValidationError, match="unknown type"):
            sim.get_name(nengo.Ensemble(10, 1, add_to_container=False))


def test_log_filter(Simulator, caplog):
    with nengo.Network() as net:
        a = nengo.Node([0])
        p = nengo.Probe(a)

    with Simulator(net) as sim:
        sim.compile(loss={p: "mse"})
        assert "missing from loss dictionary" not in caplog.text

        sim.tensor_graph.add_weight()
        assert "constraint is deprecated" not in caplog.text

        tf.compat.v1.keras.backend.get_session()
        assert "tf.keras.backend.get_session" not in caplog.text

        # make sure there were no other deprecation warnings
        filt = TFLogFilter(err_on_deprecation=True)
        for rec in caplog.records:
            filt.filter(rec)

    # make sure the deprecation checking works
    logger = logging.getLogger("test")
    logger.addFilter(filt)
    with pytest.raises(Exception, match="Deprecation warning"):
        logger.warning("a thing is deprecated")


@pytest.mark.training
def test_uneven_batch_size(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node(np.zeros(2))
        nengo.Probe(inp)

    with Simulator(net, minibatch_size=16) as sim:
        with pytest.warns(UserWarning, match="not evenly divisible"):
            sim.predict(np.zeros((20, 5, 2)))

        sim.compile(optimizer=tf.optimizers.SGD(0), loss=tf.losses.mse)

        with pytest.warns(UserWarning, match="not evenly divisible"):
            sim.fit(np.zeros((20, 5, 2)), np.zeros((20, 5, 2)))


def test_progress_bar(Simulator, capsys, monkeypatch):
    class StdProgressBar(utils.ProgressBar):  # pylint: disable=too-many-ancestors
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # manually set fd to stdout (passing `fd=sys.stdout` in init is not the
            # same, because it wraps it in some stream wrapper that is slightly
            # different). we want to use sys.stdout exactly so that stdout capturing
            # (e.g. in capsys) works correctly.
            self.fd = sys.stdout

    monkeypatch.setattr(utils, "ProgressBar", StdProgressBar)

    net, _, _ = dummies.linear_net()

    with Simulator(net, progress_bar=True) as sim:
        # default to displaying build information
        build_output = capsys.readouterr().out
        assert "Building" in build_output
        assert "Optimization" in build_output
        assert "Construction" in build_output

        # display simulation progress
        sim.run_steps(10)
        assert "Simulating" in capsys.readouterr().out

        # simulation progress explicitly disabled
        sim.run_steps(10, progress_bar=False)
        assert capsys.readouterr().out == ""

    with Simulator(net, progress_bar=False) as sim:
        # no build information
        assert capsys.readouterr().out == ""

        # defaults to false
        sim.run_steps(10)
        assert capsys.readouterr().out == ""

        # can be explicitly enabled
        sim.run_steps(10, progress_bar=True)
        assert "Simulating" in capsys.readouterr().out


def test_soft_reset(Simulator):
    with Simulator(nengo.Network()) as sim:
        sim.run_steps(10)
        assert sim.n_steps == 10
        with pytest.warns(DeprecationWarning, match="use Simulator.reset"):
            sim.soft_reset()
        assert sim.n_steps == 0


def test_sim_close(Simulator):
    with Simulator(nengo.Network()) as sim:
        assert sim.tensor_graph

    with pytest.raises(SimulatorClosed, match="access Simulator.tensor_graph"):
        assert sim.tensor_graph

    with pytest.raises(SimulatorClosed, match="simulator is closed"):
        with sim:
            pass


def test_logging(Simulator, caplog):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(inp, ens)
        nengo.Probe(ens)

    # run a simulation with logging to verify that there are no errors
    with caplog.at_level(logging.NOTSET):
        with Simulator(net) as sim:
            sim.run_steps(10)

    for rec in caplog.records:
        assert rec.getMessage(), f"Record {rec} has empty message"


def test_floatx_context(Simulator):
    with nengo.Network() as net:
        configure_settings(dtype="float64")

        def fail_func(t):
            assert tf.keras.backend.floatx() == "float64"
            assert False, "intentional failure"

        nengo.Node(fail_func, size_out=1)

    assert tf.keras.backend.floatx() == "float32"
    sim = Simulator(net)
    with pytest.raises(AssertionError, match="intentional failure"):
        sim.step()
    assert tf.keras.backend.floatx() == "float32"
    sim.close()
