# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import objectives


@pytest.mark.parametrize("axis, weight, order",
                         [(None, 0.6, 1),
                          (None, 0.7, "euclidean"),
                          (1, None, 2)])
def test_regularize(axis, weight, order, rng, sess):
    x_init = rng.randn(2, 3, 4, 5)

    x = tf.constant(x_init)

    reg = objectives.Regularize(weight=weight, order=order, axis=axis)(x)

    sess.run(tf.global_variables_initializer())
    reg_val = sess.run(reg)

    if order == "euclidean":
        order = 2

    if weight is None:
        weight = 1

    if axis is None:
        truth = np.reshape(x_init, x_init.shape[:2] + (-1,))
        axis = 2
    else:
        truth = x_init
        axis += 2
    truth = np.mean(np.linalg.norm(truth, ord=order, axis=axis))
    truth *= weight

    assert np.allclose(reg_val, truth)


@pytest.mark.training
@pytest.mark.parametrize("mode", ("activity", "weights"))
def test_regularize_train(Simulator, mode, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([1])
        b = nengo.Ensemble(30, 1, neuron_type=nengo.Sigmoid(tau_ref=1),
                           gain=nengo.dists.Choice([1]),
                           bias=nengo.dists.Choice([0]))
        c = nengo.Connection(a, b.neurons, synapse=None,
                             transform=nengo.dists.Uniform(-0.1, 0.1))

        if mode == "weights":
            p = nengo.Probe(c, "weights")
        else:
            p = nengo.Probe(b.neurons)

    with Simulator(net) as sim:
        sim.train(
            5, tf.train.RMSPropOptimizer(0.01 if mode == "weights" else 0.1),
            objective={p: objectives.Regularize()}, n_epochs=100)

        sim.step()
        assert np.allclose(sim.data[p], 0, atol=1e-2)
