from nengo import ensemble, Network, Connection, Ensemble
from nengo.exceptions import (SimulationError, ValidationError,
                              NetworkContextError)
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
    inputs = {"a": np.arange(100)[:, None]}
    targets = {"b": np.arange(1, 101)[:, None]}

    x_all = []
    y_all = []
    for _, x, y in utils.minibatch_generator(inputs, targets, 10,
                                             shuffle=shuffle):
        x_all += [x["a"]]
        y_all += [y["b"]]

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    if shuffle:
        assert not np.allclose(x_all, inputs["a"])
        assert not np.allclose(y_all, targets["b"])
        x_all = np.sort(x_all, axis=0)
        y_all = np.sort(y_all, axis=0)

    assert np.allclose(x_all, inputs["a"])
    assert np.allclose(y_all, targets["b"])

    x_all = []
    y_all = []
    with pytest.warns(UserWarning):
        for _, x, y in utils.minibatch_generator(inputs, targets, 12,
                                                 shuffle=shuffle):
            assert x["a"].shape[0] == 12
            assert y["b"].shape[0] == 12
            x_all += [x["a"]]
            y_all += [y["b"]]
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
    inputs = {"a": np.random.randn(2, 10)}
    targets = {"b": np.random.randn(2, 10)}

    duration = 10 if truncation is None else truncation

    with pytest.warns(None) as w:
        for i, (o, x, y) in enumerate(utils.minibatch_generator(
                inputs, targets, 2, shuffle=False, truncation=truncation)):
            assert np.allclose(x["a"], inputs["a"][:, o:o + duration])
            assert np.allclose(y["b"], targets["b"][:, o:o + duration])

    assert i == 10 // duration - (10 % duration == 0)

    assert len(w) == (1 if truncation == 3 else 0)


def test_print_op(capsys):
    with tf.Session() as sess:
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


def test_configure_trainable():
    with Network() as net:
        conf = net.config
        utils.configure_settings(trainable=None)

    assert conf[Ensemble].trainable is None
    assert conf[Connection].trainable is None
    assert conf[ensemble.Neurons].trainable is None

    # check that we can set trainable after it is set up for configuration
    conf[Ensemble].trainable = True

    # check that boolean value is enforced
    with pytest.raises(ValidationError):
        conf[Ensemble].trainable = 5

    assert conf[Ensemble].trainable is True

    # check that calling configure again overrides previous changes
    with net:
        utils.configure_settings(trainable=None)

    assert conf[Ensemble].trainable is None

    # check that non-None defaults work
    with net:
        utils.configure_settings(trainable=False)

    assert conf[Ensemble].trainable is False

    # check that calling configure outside network context is an error
    with pytest.raises(NetworkContextError):
        utils.configure_settings(trainable=None)


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


def test_session_config(Simulator):
    with Network() as net:
        utils.configure_settings(session_config={
            "graph_options.optimizer_options.opt_level": 21,
            "gpu_options.allow_growth": True})

    with Simulator(net) as sim:
        assert sim.sess._config.graph_options.optimizer_options.opt_level == 21
        assert sim.sess._config.gpu_options.allow_growth
