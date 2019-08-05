# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras


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
        steps_input, node_inputs = layer_sim.tensor_graph.build_inputs()
        inputs = [steps_input] + list(node_inputs.values())
        outputs = layer_sim.tensor_graph(inputs)
        keras_model = keras.Model(inputs=inputs, outputs=outputs)

        inputs = layer_sim._generate_inputs(n_steps=n_steps)

        output_vals = keras_model.predict(inputs)

        assert len(output_vals) == 3
        assert np.allclose(output_vals[0], n_steps)
        assert np.allclose(output_vals[1], sim.data[p_a])
        assert np.allclose(output_vals[2], sim.data[p_b])


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
        with pytest.warns(UserWarning, reason="Batch size is determined statically"):
            output = sim.predict(n_steps=n_steps, batch_size=-1)
        assert np.allclose(output[p], data_noinput)

        # numpy input (single batch)
        output = sim.predict_on_batch(a_vals[:4])
        assert np.allclose(output[p], sim.data[p])

        # numpy input (multiple batches)
        output = sim.predict(a_vals)
        assert np.allclose(output[p], data_tile)

        # tf input
        # TODO: get this working
        # output = sim.predict(tf.constant(a_vals), steps=n_batches)
        # assert np.allclose(output[p], data_tile)

        # dict input
        for key in [a, "a"]:
            output = sim.predict({key: a_vals})
            assert np.allclose(output[p], data_tile)

        # generator input
        for func in ["predict", "predict_generator"]:
            output = getattr(sim, func)(
                (
                    [
                        np.ones((sim.minibatch_size, 1), dtype=np.int32) * n_steps,
                        a_vals[i * sim.minibatch_size : (i + 1) * sim.minibatch_size],
                    ]
                    for i in range(n_batches)
                ),
                steps=n_batches,
            )
            assert np.allclose(output[p], data_tile)

    # dataset input
    # TODO: why does this crash if placed on gpu?
    with Simulator(net, minibatch_size=4, device="/cpu:0") as sim:
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "n_steps": tf.ones((12, 1), dtype=np.int32) * n_steps,
                "a": tf.constant(a_vals),
            }
        ).batch(sim.minibatch_size)

        output = sim.predict(dataset)
        assert np.allclose(output[p], data_tile)
