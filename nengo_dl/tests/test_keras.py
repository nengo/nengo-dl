# pylint: disable=missing-docstring

import nengo
from nengo.exceptions import BuildError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import config, dists


@pytest.mark.parametrize("minibatch_size", (None, 1, 3))
def test_tensorgraph_layer(Simulator, seed, minibatch_size):
    n_steps = 100

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(lambda t: np.sin(20 * np.pi * t))
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        p_a = nengo.Probe(a)
        p_b = nengo.Probe(b)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(n_steps)

    with Simulator(net, minibatch_size=minibatch_size) as layer_sim:
        node_inputs, steps_input = layer_sim.tensor_graph.build_inputs()
        inputs = list(node_inputs.values()) + [steps_input]
        outputs = layer_sim.tensor_graph(inputs)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        inputs = layer_sim._generate_inputs(n_steps=n_steps)

        output_vals = keras_model.predict(inputs)

        assert len(output_vals) == 3
        assert np.allclose(output_vals[0], sim.data[p_a])
        assert np.allclose(output_vals[1], sim.data[p_b])
        assert np.allclose(output_vals[2], n_steps)


def test_predict(Simulator, seed):
    n_steps = 100

    with nengo.Network(seed=seed) as net:
        a = nengo.Node([2], label="a")
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        p = nengo.Probe(b)

    with Simulator(net, minibatch_size=4) as sim:
        a_vals = np.ones((12, n_steps, 1))
        n_batches = a_vals.shape[0] // sim.minibatch_size

        sim.run_steps(n_steps)
        data_noinput = sim.data[p]
        sim.soft_reset(include_probes=True)

        sim.run_steps(n_steps, data={a: a_vals[:4]})
        data_tile = np.tile(sim.data[p], (n_batches, 1, 1))
        sim.soft_reset()

        # no input (also checking batch_size is ignored)
        with pytest.warns(UserWarning, match="Batch size is determined statically"):
            output = sim.predict(n_steps=n_steps, batch_size=-1)
        assert np.allclose(output[p], data_noinput)

        # numpy input (single batch)
        output = sim.predict_on_batch(a_vals[:4])
        assert np.allclose(output[p], sim.data[p])

        # numpy input (multiple batches)
        output = sim.predict(a_vals)
        assert np.allclose(output[p], data_tile)

        # tf input
        # TODO: this will work in eager mode
        # output = sim.predict(tf.constant(a_vals))
        # assert np.allclose(output[p], data_tile)

        # dict input
        for key in [a, "a"]:
            output = sim.predict({key: a_vals})
            assert np.allclose(output[p], data_tile)

        # generator input
        output = sim.predict(
            (
                [
                    a_vals[i * sim.minibatch_size : (i + 1) * sim.minibatch_size],
                    np.ones((sim.minibatch_size, 1), dtype=np.int32) * n_steps,
                ]
                for i in range(n_batches)
            ),
            steps=n_batches,
        )
        assert np.allclose(output[p], data_tile)

    # dataset input
    # TODO: this crashes if placed on GPU (but not in eager mode)
    with Simulator(net, minibatch_size=4, device="/cpu:0") as sim:
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "a": tf.constant(a_vals),
                "n_steps": tf.ones((12, 1), dtype=np.int32) * n_steps,
            }
        ).batch(sim.minibatch_size)

        output = sim.predict(dataset)
        assert np.allclose(output[p], data_tile)


def test_evaluate(Simulator):
    minibatch_size = 3
    n_steps = 10
    n_batches = 2

    with nengo.Network() as net:
        inp0 = nengo.Node([0])
        inp1 = nengo.Node([0])
        p0 = nengo.Probe(inp0)
        p1 = nengo.Probe(inp1)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        # single probe
        sim.compile(loss={"probe": tf.losses.mse})
        targets = np.ones((minibatch_size, n_steps, 1))
        with pytest.warns(UserWarning, match="Batch size is determined statically"):
            loss = sim.evaluate(n_steps=n_steps, y=targets, batch_size=-1)
        assert np.allclose(loss["loss"], 1)
        assert np.allclose(loss["probe_loss"], 1)
        assert "probe_1_loss" not in loss

        # multiple probes
        sim.compile(loss=tf.losses.mse)
        loss = sim.evaluate(n_steps=n_steps, y={p0: targets, p1: targets})
        assert np.allclose(loss["loss"], 2)
        assert np.allclose(loss["probe_loss"], 1)
        assert np.allclose(loss["probe_1_loss"], 1)

        # default inputs
        loss = sim.evaluate(
            y={
                p0: np.zeros((minibatch_size, n_steps, 1)),
                p1: np.zeros((minibatch_size, n_steps, 1)),
            },
            n_steps=n_steps,
        )
        assert np.allclose(loss["loss"], 0)
        assert np.allclose(loss["probe_loss"], 0)
        assert np.allclose(loss["probe_1_loss"], 0)

        # list inputs
        inputs = np.ones((minibatch_size * n_batches, n_steps, 1))
        targets = inputs.copy()
        loss = sim.evaluate(x=[inputs, inputs * 2], y={p0: targets, p1: targets})
        assert np.allclose(loss["loss"], 1)
        assert np.allclose(loss["probe_loss"], 0)
        assert np.allclose(loss["probe_1_loss"], 1)

        # tensor inputs
        # TODO: this will work in eager mode
        # loss = sim.evaluate(
        #     x=[tf.constant(inputs), tf.constant(inputs * 2)],
        #     y={p0: tf.constant(targets), p1: tf.constant(targets)},
        # )
        # assert np.allclose(loss["loss"], 1)
        # assert np.allclose(loss["probe_loss"], 0)
        # assert np.allclose(loss["probe_1_loss"], 1)

        gen = (
            (
                {
                    "node": np.ones((minibatch_size, n_steps, 1)),
                    "node_1": np.ones((minibatch_size, n_steps, 1)) * 2,
                    "n_steps": np.ones((minibatch_size, 1)) * n_steps,
                },
                {
                    "probe": np.ones((minibatch_size, n_steps, 1)),
                    "probe_1": np.ones((minibatch_size, n_steps, 1)),
                },
            )
            for _ in range(n_batches)
        )

        loss = sim.evaluate(gen, steps=n_batches)
        assert np.allclose(loss["loss"], 1)
        assert np.allclose(loss["probe_loss"], 0)
        assert np.allclose(loss["probe_1_loss"], 1)

        # check custom objective
        def constant_error(y_true, y_pred):
            return tf.constant(3.0)

        sim.compile(loss={p0: constant_error})
        assert np.allclose(
            sim.evaluate(
                y={p0: np.zeros((minibatch_size, n_steps, 1))}, n_steps=n_steps
            )["loss"],
            3,
        )

        # test metrics
        sim.compile(
            loss=tf.losses.mse,
            metrics={p0: constant_error, p1: [constant_error, "mae"]},
        )
        output = sim.evaluate(
            y={
                p0: np.ones((minibatch_size, n_steps, 1)),
                p1: np.ones((minibatch_size, n_steps, 1)) * 2,
            },
            n_steps=n_steps,
        )
        assert np.allclose(output["loss"], 5)
        assert np.allclose(output["probe_loss"], 1)
        assert np.allclose(output["probe_1_loss"], 4)
        assert np.allclose(output["probe_constant_error"], 3)
        assert np.allclose(output["probe_1_constant_error"], 3)
        assert "probe_mae" not in output
        assert np.allclose(output["probe_1_mae"], 2)


@pytest.mark.training
def test_fit(Simulator, seed):
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
        nengo.Connection(inp, ens.neurons, transform=dists.Glorot())
        nengo.Connection(ens.neurons, out.neurons, transform=dists.Glorot())

        nengo.Probe(out.neurons)

    with Simulator(
        net, minibatch_size=minibatch_size, unroll_simulation=1, seed=seed
    ) as sim:
        x = np.asarray([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]])
        y = np.asarray([[[0.1]], [[0.9]], [[0.9]], [[0.1]]])

        sim.compile(optimizer=tf.optimizers.Adam(0.01), loss=tf.losses.mse)
        # note: batch_size should be ignored
        with pytest.warns(UserWarning, match="Batch size is determined statically"):
            history = sim.fit(
                [x[..., [0]], x[..., [1]]],
                y,
                validation_data=([x[..., [0]], x[..., [1]]], y),
                epochs=200,
                verbose=0,
                batch_size=-1,
            )
        assert history.history["loss"][-1] < 5e-4
        assert history.history["val_loss"][-1] < 5e-4

        # check that validation_sample_weights work correctly
        history = sim.fit(
            [x[..., [0]], x[..., [1]]],
            y,
            validation_data=([x[..., [0]], x[..., [1]]], y, np.zeros(y.shape[0])),
            epochs=1,
            verbose=0,
        )
        assert np.allclose(history.history["val_loss"][-1], 0)

        # TODO: this will work in eager mode
        # sim.reset()
        # history = sim.fit(
        #     [tf.constant(x[..., [0]]), tf.constant(x[..., [1]])],
        #     tf.constant(y),
        #     epochs=200,
        #     verbose=0,
        # )
        # assert history.history["loss"][-1] < 5e-4

        sim.reset()
        history = sim.fit(
            (
                ((x[..., [0]], x[..., [1]], np.ones((4, 1), dtype=np.int32)), y)
                for _ in range(200)
            ),
            epochs=20,
            steps_per_epoch=10,
            verbose=0,
        )
        assert history.history["loss"][-1] < 5e-4

    # TODO: this crashes if placed on GPU (but not in eager mode)
    with Simulator(
        net,
        minibatch_size=minibatch_size,
        unroll_simulation=1,
        seed=seed,
        device="/cpu:0",
    ) as sim:
        sim.compile(optimizer=tf.optimizers.Adam(0.01), loss=tf.losses.mse)

        history = sim.fit(
            tf.data.Dataset.from_tensors(
                ((x[..., [0]], x[..., [1]], np.ones((4, 1), dtype=np.int32)), y)
            ),
            validation_data=tf.data.Dataset.from_tensors(
                ((x[..., [0]], x[..., [1]], np.ones((4, 1), dtype=np.int32)), y)
            ),
            epochs=200,
            verbose=0,
        )
        assert history.history["loss"][-1] < 5e-4
        assert history.history["val_loss"][-1] < 5e-4


@pytest.mark.training
def test_learning_phase(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(
            1, 1, gain=[0], bias=[1], neuron_type=nengo.SpikingRectifiedLinear()
        )
        nengo.Connection(inp, ens, synapse=None)
        p = nengo.Probe(ens.neurons)

    with tf.keras.backend.learning_phase_scope(1):
        with Simulator(net) as sim:
            sim.run_steps(10)
            assert np.allclose(sim.data[p], 1)

    with Simulator(net) as sim:
        sim.run_steps(10)
        assert np.allclose(sim.data[p], 0)


def test_learning_phase_warning(Simulator):
    with nengo.Network() as net:
        config.configure_settings(inference_only=True)
    with pytest.raises(BuildError, match="inference_only=True"):
        with tf.keras.backend.learning_phase_scope(1):
            with Simulator(net):
                pass
