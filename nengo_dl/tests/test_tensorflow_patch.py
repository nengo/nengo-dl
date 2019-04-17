# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import tensorflow_patch
from nengo_dl.compat import tf_compat, RefVariable


@pytest.fixture
def undo_patch(request):
    tensorflow_patch.undo_patch()

    yield

    tensorflow_patch.patch_dynamic_stitch_grad()
    tensorflow_patch.patch_state_grads()


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_dynamic_stitch_fail(sess):
    test_dynamic_stitch(sess)


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_state_grads_fail(sess):
    test_state_grads(sess)


def test_dynamic_stitch(sess):
    x = tf.zeros((1, 3))
    y = tf.dynamic_stitch([[0], [0]], [x, tf.ones((1, 3))])
    z = tf.gather(y, [0])

    with sess.as_default():
        analytic, numeric = tf_compat.test.compute_gradient(
            x, (1, 3), z, (1, 3))

    assert np.allclose(analytic, numeric)


def test_state_grads(sess):
    v = RefVariable([0., 0., 0.])
    x = tf.ones((3,))

    y0 = tf_compat.assign(v, x)
    y1 = tf_compat.assign_add(v, x)

    grad0 = tf.gradients(ys=y0, xs=[v, x])
    grad1 = tf.gradients(ys=y1, xs=[v, x])

    grad_vals = sess.run((grad0, grad1))

    assert np.allclose(grad_vals[0][0], 0)
    assert np.allclose(grad_vals[0][1], 1)
    assert np.allclose(grad_vals[1][0], 1)
    assert np.allclose(grad_vals[1][1], 1)

    v = RefVariable([0., 0., 0.])
    x = tf.ones((1,))
    y0 = tf_compat.scatter_update(v, [0], x)
    y1 = tf_compat.scatter_add(v, [0], x)

    grad0 = tf.gradients(ys=y0, xs=[v._ref(), x])
    grad1 = tf.gradients(ys=y1, xs=[v._ref(), x])

    grad_vals = sess.run((grad0, grad1))

    assert np.allclose(grad_vals[0][0], [0, 1, 1])
    assert np.allclose(grad_vals[0][1], 1)
    assert np.allclose(grad_vals[1][0], 1)
    assert np.allclose(grad_vals[1][1], 1)
