# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf
from nengo import builder
from nengo.builder import operator, signal

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
        sim.run_steps(5)


@pytest.mark.parametrize("device", ("/cpu:0", "/gpu:0", None))
@pytest.mark.parametrize("dtype", (tf.float32, tf.float64))
def test_sparse_matmul(dtype, device):
    if device == "/gpu:0" and not tf_gpu_installed:
        pytest.skip("Can't test GPU device without GPU support")

    with tf.device(device):
        A = tf.ones(3, dtype=dtype)
        idxs = tf.constant([[0, 0], [1, 1], [2, 2]])
        shape = (3, 3)
        X = tf.ones((3, 1), dtype=dtype) * 2

        with pytest.warns(None) as recwarns:
            Y = sparse_matmul(idxs, A, shape, X)

    if dtype == tf.float64 and (
        device == "/gpu:0" or (device is None and tf_gpu_installed)
    ):
        assert len([w for w in recwarns if "sparse_matmul" in str(w.message)]) == 1
    else:
        assert len([w for w in recwarns if "sparse_matmul" in str(w.message)]) == 0

    assert Y.dtype == dtype

    assert np.allclose(tf.keras.backend.get_value(Y), np.ones(3) * 2)


def test_merged_simpyfunc(Simulator):
    with nengo.Network() as net:
        # nodes get time + x
        node0 = nengo.Node(lambda t, x: x + t, size_in=1)
        node1 = nengo.Node(lambda t, x: x + 2 * t, size_in=1)

        # direct ensembles won't get time as input
        ens0 = nengo.Ensemble(10, 1, neuron_type=nengo.Direct())
        nengo.Connection(ens0, node0, function=lambda x: x + 1)
        ens1 = nengo.Ensemble(10, 1, neuron_type=nengo.Direct())
        nengo.Connection(ens1, node1, function=lambda x: x + 2)

        p0 = nengo.Probe(node0)
        p1 = nengo.Probe(node1)

    with nengo.Simulator(net) as canonical:
        canonical.run_steps(10)

    with Simulator(net) as sim:
        assert (
            len(
                [
                    ops
                    for ops in sim.tensor_graph.plan
                    if isinstance(ops[0], operator.SimPyFunc)
                ]
            )
            == 2
        )

        sim.run_steps(10)

    assert np.allclose(canonical.data[p0], sim.data[p0])
    assert np.allclose(canonical.data[p1], sim.data[p1])
