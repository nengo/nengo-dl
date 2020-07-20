# pylint: disable=missing-docstring

import numpy as np
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

    x = utils.align_func(tf.float32)(my_func)()
    assert x.shape == (4,)
    assert x.dtype == np.float32
    assert np.allclose(x, [0, 1, 2, 3])

    x = utils.align_func(np.int64)(my_func)()
    assert x.dtype == np.int64
    assert np.allclose(x, [[0, 1, 2, 3]])


def test_progress_bar():
    progress = utils.ProgressBar("test", max_value=10).start()

    assert progress.max_steps == progress.max_value == 10

    sub = progress.sub().start()

    # check that starting a new subprocess closes the first one
    sub2 = progress.sub().start()
    assert sub.finished

    # check that iterable wrapping works properly
    counter = 0
    for _ in progress(range(15)):
        counter += 1

    # note: progress value cut-off at 10 (the max value we specified)
    assert counter == 15
    assert progress.value == 10

    # check that closing the parent process closes the sub
    assert sub2.finished
    assert progress.finished


def test_gpu_check():
    gpus_available = tf.config.list_physical_devices("GPU")

    assert utils.tf_gpu_installed == (len(gpus_available) > 0)
