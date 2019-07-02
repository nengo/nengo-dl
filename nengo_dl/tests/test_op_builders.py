# pylint: disable=missing-docstring

from nengo import builder
from nengo.builder import signal, operator
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl.op_builders import sparse_matmul
from nengo_dl.utils import tf_gpu_installed


def test_elementwise_inc(Simulator):
    # note: normally the op_builders are just tested as part of the nengo
    # tests.  but in this particular case, there are no nengo tests that
    # have a scalar, non-1 transform.  those all get optimized out during
    # the graph optimization, so we don't end up with any tests of
    # elementwiseinc where A is a scalar. so that's what this is for.

    model = builder.Model()

    a = signal.Signal([2.0])
    x = signal.Signal([[3.0]])
    y = signal.Signal([[1.0]])
    op = operator.ElementwiseInc(a, x, y)
    model.add_op(op)

    with Simulator(None, model=model) as sim:
        sim.sess.run(sim.tensor_graph.steps_run,
                     feed_dict={sim.tensor_graph.step_var: 0,
                                sim.tensor_graph.stop_var: 5})


@pytest.mark.parametrize("device", ("/cpu:0", "/gpu:0", None))
@pytest.mark.parametrize("dtype", (tf.float32, tf.float64))
def test_sparse_matmul(sess, dtype, device):
    if device == "/gpu:0" and not tf_gpu_installed:
        pytest.skip("Can't test GPU device without GPU support")

    with tf.device(device):
        A = tf.ones(3, dtype=dtype)
        idxs = tf.constant([[0, 0], [1, 1], [2, 2]])
        shape = (3, 3)
        X = tf.ones((3, 1), dtype=dtype) * 2

        with pytest.warns(None) as recwarns:
            Y = sparse_matmul(idxs, A, shape, X)

    if dtype == tf.float64 and (device == "/gpu:0"
                                or (device is None and tf_gpu_installed)):
        assert len(
            [w for w in recwarns if "sparse_matmul" in str(w.message)]) == 1

        # need to go back past cast
        assert Y.op.inputs[0].op.inputs[1].dtype == tf.float32
    else:
        assert len(
            [w for w in recwarns if "sparse_matmul" in str(w.message)]) == 0
        assert Y.op.inputs[1].dtype == dtype

    assert Y.dtype == dtype

    assert np.allclose(sess.run(Y), np.ones(3) * 2)
