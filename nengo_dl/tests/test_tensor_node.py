import nengo
from nengo.exceptions import ValidationError, SimulationError
import numpy as np
import pytest
import tensorflow as tf

import nengo_dl


def test_validation():
    with nengo.Network() as net:
        with pytest.raises(ValidationError):
            nengo_dl.TensorNode([0])

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda t: t, size_out=0)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda a, b, c: a)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda x: None)

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda x: [0])

        with pytest.raises(ValidationError):
            nengo_dl.TensorNode(lambda t: tf.zeros((2, 2, 2)))

        n = nengo_dl.TensorNode(lambda t: tf.zeros((5, 2)))
        assert n.size_out == 2

        n = nengo_dl.TensorNode(lambda t: tf.zeros((5, 2)), size_out=4)
        assert n.size_out == 4

    with nengo.Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.step()


def test_node():
    minibatch_size = 3
    with nengo.Network() as net:
        node0 = nengo_dl.TensorNode(lambda t: tf.tile(tf.reshape(t, (1, -1)),
                                                      (minibatch_size, 1)))
        node1 = nengo_dl.TensorNode(lambda t, x: tf.sin(x), size_in=1)
        nengo.Connection(node0, node1, synapse=None)

        p0 = nengo.Probe(node0)
        p1 = nengo.Probe(node1)

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p0], sim.trange()[None, :, None])
    assert np.allclose(sim.data[p1], np.sin(sim.trange()[None, :, None]))


def test_pre_build():
    class TestFunc:
        def pre_build(self, size_in, size_out):
            self.weights = tf.Variable(tf.ones((size_in[1], size_out[1])))

        def __call__(self, t, x):
            return tf.matmul(x, self.weights)

    with nengo.Network() as net:
        inp = nengo.Node([1, 1])
        test = nengo_dl.TensorNode(TestFunc(), size_in=2, size_out=3)
        nengo.Connection(inp, test, synapse=None)
        p = nengo.Probe(test)

    with nengo_dl.Simulator(net) as sim:
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])

        sim.reset()
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])
