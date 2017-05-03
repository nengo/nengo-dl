import traceback

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_grad

from nengo_dl import tensorflow_patch


@pytest.fixture
def undo_patch(request):
    # TODO: ideally we would just reload(tensorflow) here, but that crashes atm
    ops._gradient_registry._registry["DynamicStitch"] = {
        "type": data_flow_grad._DynamicStitchGrads,
        "location": traceback.extract_stack()}
    ops._gradient_registry._registry["ScatterUpdate"] = None
    ops._gradient_registry._registry["ScatterAdd"] = None
    ops._gradient_registry._registry["Assign"] = None
    ops._gradient_registry._registry["AssignAdd"] = None

    yield

    tensorflow_patch.patch_dynamic_stitch_grad()
    tensorflow_patch.patch_state_grads()


@pytest.mark.xfail
@pytest.mark.usefixtures("undo_patch")
def test_dynamic_stitch_fail():
    x = tf.zeros((1, 3))
    y = tf.dynamic_stitch([[0], [0]], [x, tf.ones((1, 3))])

    with tf.Session():
        analytic, numeric = tf.test.compute_gradient(x, (1, 3), y, (1, 3))

        assert np.allclose(analytic, numeric)


def test_dynamic_stitch():
    x = tf.zeros((1, 3))
    y = tf.dynamic_stitch([[0], [0]], [x, tf.ones((1, 3))])

    with tf.Session():
        analytic, numeric = tf.test.compute_gradient(x, (1, 3), y, (1, 3))

        assert np.allclose(analytic, numeric)


def test_state_grads():
    v = tf.Variable([0., 0., 0.])
    x = tf.ones((3,))

    y0 = tf.assign(v, x)
    y1 = tf.assign_add(v, x)

    with tf.Session() as sess:
        # TODO: the ._ref() is necessary due to something in tensorflow 1.0.0,
        # can remove if we upgrade requirements
        grad0 = tf.gradients(y0, [v._ref(), x])
        grad1 = tf.gradients(y1, [v._ref(), x])

        grad_vals = sess.run((grad0, grad1))

        assert np.allclose(grad_vals[0][0], 0)
        assert np.allclose(grad_vals[0][1], 1)
        assert np.allclose(grad_vals[1][0], 1)
        assert np.allclose(grad_vals[1][1], 1)

    x = tf.ones((1,))
    y0 = tf.scatter_update(v, [0], x)
    y1 = tf.scatter_add(v, [0], x)

    with tf.Session() as sess:
        grad0 = tf.gradients(y0, [v._ref(), x])
        grad1 = tf.gradients(y1, [v._ref(), x])
        grad_vals = sess.run((grad0, grad1))

        assert np.allclose(grad_vals[0][0], [0, 1, 1])
        assert np.allclose(grad_vals[0][1], 1)
        assert np.allclose(grad_vals[1][0], 1)
        assert np.allclose(grad_vals[1][1], 1)
