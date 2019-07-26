# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import tensorflow_patch
from nengo_dl.compat import tf_compat


@pytest.fixture
def undo_patch(request):
    tensorflow_patch.undo_patch()

    yield

    tensorflow_patch.patch_dynamic_stitch_grad()


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_dynamic_stitch_fail(sess):
    test_dynamic_stitch(sess)


def test_dynamic_stitch(sess):
    x = tf.zeros((1, 3))
    y = tf.dynamic_stitch([[0], [0]], [x, tf.ones((1, 3))])
    z = tf.gather(y, [0])

    with sess.as_default():
        analytic, numeric = tf_compat.test.compute_gradient(x, (1, 3), z, (1, 3))

    assert np.allclose(analytic, numeric)
