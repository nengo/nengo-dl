# pylint: disable=missing-docstring

import nengo
from nengo.exceptions import SimulationError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import utils


def test_sanitize_name():
    assert utils.sanitize_name(0) == "0"
    assert utils.sanitize_name("a b") == "a_b"
    assert utils.sanitize_name("a:b") == "a_b"

    assert utils.sanitize_name(r"Aa0.-/\,?^&*") == r"Aa0.-/"


def test_function_names():
    def my_func(x):
        return x

    class MyFunc:
        def __call__(self, x):
            return x

    assert utils.function_name(my_func) == "my_func"
    assert utils.function_name(MyFunc()) == "MyFunc"


def test_align_func():
    def my_func():
        return [0, 1, 2, 3]

    x = utils.align_func((4,), tf.float32)(my_func)()
    assert x.shape == (4,)
    assert x.dtype == np.float32
    assert np.allclose(x, [0, 1, 2, 3])

    x = utils.align_func((2, 2), np.int64)(my_func)()
    assert x.shape == (2, 2)
    assert x.dtype == np.int64
    assert np.allclose(x, [[0, 1], [2, 3]])


@pytest.mark.parametrize("shuffle", (False, True))
def test_minibatch_generator(shuffle):
    data = {"a": np.arange(100)[:, None], "b": np.arange(1, 101)[:, None]}

    x_all = []
    y_all = []
    for _, d in utils.minibatch_generator(data, 10, shuffle=shuffle):
        x_all += [d["a"]]
        y_all += [d["b"]]

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    if shuffle:
        assert not np.allclose(x_all, data["a"])
        assert not np.allclose(y_all, data["b"])
        x_all = np.sort(x_all, axis=0)
        y_all = np.sort(y_all, axis=0)

    assert np.allclose(x_all, data["a"])
    assert np.allclose(y_all, data["b"])

    x_all = []
    y_all = []
    with pytest.warns(UserWarning):
        for _, d in utils.minibatch_generator(data, 12, shuffle=shuffle):
            assert d["a"].shape[0] == 12
            assert d["b"].shape[0] == 12
            x_all += [d["a"]]
            y_all += [d["b"]]
    x_all = np.sort(np.concatenate(x_all))
    y_all = np.sort(np.concatenate(y_all))

    assert len(x_all) == 96
    assert len(y_all) == 96

    if shuffle:
        assert not np.allclose(x_all, np.arange(96)[:, None])
        assert not np.allclose(y_all, np.arange(1, 97)[:, None])
    else:
        assert np.allclose(x_all, np.arange(96)[:, None])
        assert np.allclose(y_all, np.arange(1, 97)[:, None])


@pytest.mark.parametrize("truncation", (None, 3, 5))
def test_truncation(truncation):
    data = {"a": np.random.randn(2, 10), "b": np.random.randn(2, 10)}

    duration = 10 if truncation is None else truncation

    with pytest.warns(None) as w:
        for i, (o, d) in enumerate(utils.minibatch_generator(
                data, 2, shuffle=False, truncation=truncation)):
            assert np.allclose(d["a"], data["a"][:, o:o + duration])
            assert np.allclose(d["b"], data["b"][:, o:o + duration])

    # pylint: disable=undefined-loop-variable
    assert i == 10 // duration - (10 % duration == 0)

    assert len(w) == (1 if truncation == 3 else 0)

    # check truncation with n_steps input
    for i, (o, d) in enumerate(utils.minibatch_generator(
            10, None, truncation=3)):
        assert o == i * 3
        if i < 3:
            assert d == 3
        else:
            assert d == 1


def test_print_op(capsys, sess):
    x = tf.constant(0)
    y = utils.print_op(x, "hello")
    z = y + 0

    sess.run(z)

    out, _ = capsys.readouterr()

    assert out == "hello 0\n"


def test_find_non_differentiable():
    x = tf.constant(0)
    y = utils.print_op(x, "test")
    z = y + 1

    with pytest.raises(SimulationError):
        utils.find_non_differentiable([x], [z])

    x = tf.constant(0)
    y = x * 2
    z = y + 1

    utils.find_non_differentiable([x], [z])


def test_progress_bar():
    progress = utils.ProgressBar("test", max_value=10).start()

    assert progress.max_steps == progress.max_value == 10

    sub = progress.sub().start()

    # check that starting a new subprocess closes the first one
    sub2 = progress.sub().start()
    assert sub.finished

    # check that iterable wrapping works properly
    for _ in progress(range(11)):
        pass

    assert progress.value == 11

    # check that closing the parent process closes the sub
    assert sub2.finished
    assert progress.finished


@pytest.mark.parametrize("axis, weight, order",
                         [(None, 0.6, 1),
                          (None, 0.7, "euclidean"),
                          (1, None, 2)])
def test_regularize(axis, weight, order, rng, sess):
    x_init = rng.randn(2, 3, 4, 5)

    x = tf.constant(x_init)

    reg = utils.Regularize(weight=weight, order=order, axis=axis)(x)

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
        b = nengo.Ensemble(30, 1, neuron_type=nengo.RectifiedLinear(),
                           gain=nengo.dists.Choice([1]),
                           bias=nengo.dists.Choice([0.1]))
        c = nengo.Connection(a, b.neurons, synapse=None,
                             transform=nengo.dists.Uniform(-0.1, 0.1))

        if mode == "weights":
            p = nengo.Probe(c, "weights")
        else:
            p = nengo.Probe(b.neurons)

    with Simulator(net) as sim:
        sim.train(
            1, tf.train.RMSPropOptimizer(0.05),
            objective={p: utils.Regularize()}, n_epochs=10)

        sim.step()
        assert np.allclose(sim.data[p], 0, atol=1e-2)
