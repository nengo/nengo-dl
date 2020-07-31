# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import losses


@pytest.mark.parametrize("axis, order", [(None, 1), (None, "euclidean"), (1, 2)])
def test_regularize(axis, order, rng):
    x_init = rng.randn(2, 3, 4, 5)

    x = tf.constant(x_init)

    reg = losses.Regularize(order=order, axis=axis)(None, x)

    reg_val = tf.keras.backend.get_value(reg)

    if order == "euclidean":
        order = 2

    if axis is None:
        truth = np.reshape(x_init, x_init.shape[:2] + (-1,))
        axis = 2
    else:
        truth = x_init
        axis += 2
    truth = np.mean(np.linalg.norm(truth, ord=order, axis=axis))

    assert np.allclose(reg_val, truth)


@pytest.mark.parametrize("mode", ("activity", "weights"))
@pytest.mark.training
def test_regularize_train(Simulator, mode, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([1])
        b = nengo.Ensemble(
            30,
            1,
            neuron_type=nengo.Sigmoid(tau_ref=1),
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )
        c = nengo.Connection(
            a, b.neurons, synapse=None, transform=nengo.dists.Uniform(-0.1, 0.1)
        )

        if mode == "weights":
            p = nengo.Probe(c, "weights")
        else:
            p = nengo.Probe(b.neurons)

        # default output required so that there is a defined gradient for all
        # parameters
        default_p = nengo.Probe(b)

    with Simulator(net) as sim:
        sim.compile(
            tf.optimizers.RMSprop(0.01 if mode == "weights" else 0.1),
            loss={p: losses.Regularize(), default_p: lambda y_true, y_pred: 0 * y_pred},
        )
        sim.fit(
            n_steps=5,
            y={
                p: np.zeros((1, 5, p.size_in)),
                default_p: np.zeros((1, 5, default_p.size_in)),
            },
            epochs=100,
        )

        sim.step()
        assert np.allclose(sim.data[p], 0, atol=1e-2)


def test_nan_mse():
    x = np.arange(10, dtype=np.float32)
    y = np.ones(10) * np.nan
    y[3:5] = 0

    loss = losses.nan_mse(y, x)

    assert tf.keras.backend.get_value(loss) == (3 ** 2 + 4 ** 2) / 10
