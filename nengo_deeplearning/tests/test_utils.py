import numpy as np
import pytest
import tensorflow as tf

from nengo_deeplearning import utils


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
    assert utils.function_name(MyFunc) == "MyFunc"


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


def test_cast_dtype():
    assert utils.cast_dtype(np.float32, tf.float32) == tf.float32
    assert utils.cast_dtype(np.int32, tf.float32) == tf.int32
    assert utils.cast_dtype(tf.float64, tf.float32) == tf.float32


@pytest.mark.parametrize("shuffle", (False, True))
def test_minibatch_generator(shuffle):
    inputs = {"a": np.arange(100)}
    targets = {"b": np.arange(100) + 1}

    x_all = []
    y_all = []
    for x, y in utils.minibatch_generator(inputs, targets, 10,
                                          shuffle=shuffle):
        x_all += [x["a"]]
        y_all += [y["b"]]

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    if shuffle:
        assert not np.allclose(x_all, np.arange(100))
        assert not np.allclose(y_all, np.arange(100) + 1)
        x_all = np.sort(x_all)
        y_all = np.sort(y_all)

    assert np.allclose(x_all, np.arange(100))
    assert np.allclose(y_all, np.arange(100) + 1)
