import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import tensorflow_patch


@pytest.fixture
def undo_patch(request):
    tensorflow_patch.undo_patch()

    yield

    tensorflow_patch.patch_dynamic_stitch_grad()
    tensorflow_patch.patch_state_grads()


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_dynamic_stitch_fail():
    test_dynamic_stitch()


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_state_grads_fail():
    test_state_grads()


def test_dynamic_stitch():
    x = tf.zeros((1, 3))
    y = tf.dynamic_stitch([[0], [0]], [x, tf.ones((1, 3))])
    z = tf.gather(y, [0])

    with tf.Session():
        analytic, numeric = tf.test.compute_gradient(x, (1, 3), z, (1, 3))

        assert np.allclose(analytic, numeric)


def test_state_grads():
    with tf.Session() as sess:
        v = tf.Variable([0., 0., 0.])
        x = tf.ones((3,))

        y0 = tf.assign(v, x)
        y1 = tf.assign_add(v, x)

        grad0 = tf.gradients(y0, [v, x])
        grad1 = tf.gradients(y1, [v, x])

        grad_vals = sess.run((grad0, grad1))

        assert np.allclose(grad_vals[0][0], 0)
        assert np.allclose(grad_vals[0][1], 1)
        assert np.allclose(grad_vals[1][0], 1)
        assert np.allclose(grad_vals[1][1], 1)

    with tf.Session() as sess:
        v = tf.Variable([0., 0., 0.])
        x = tf.ones((1,)) * 2
        y0 = tf.scatter_update(v, [0], x)
        y1 = tf.scatter_add(v, [0], x)
        y2 = tf.scatter_mul(v, [0], x)

        grad0 = tf.gradients(y0, [v._ref(), x])
        grad1 = tf.gradients(y1, [v._ref(), x])
        grad2 = tf.gradients(y2, [v._ref(), x])

        assert grad2[1] is None
        grad2 = [grad2[0]]

        grad_vals = sess.run((grad0, grad1, grad2))

        assert np.allclose(grad_vals[0][0], [0, 1, 1])
        assert np.allclose(grad_vals[0][1], 1)
        assert np.allclose(grad_vals[1][0], 1)
        assert np.allclose(grad_vals[1][1], 1)
        assert np.allclose(grad_vals[2][0], [2, 1, 1])
