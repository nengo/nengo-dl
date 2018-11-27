# pylint: disable=missing-docstring

import nengo
from nengo.exceptions import ValidationError, SimulationError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import TensorNode, configure_settings, reshaped, tensor_layer


def test_validation(Simulator):
    with nengo.Network() as net:
        # not a callable
        with pytest.raises(ValidationError):
            TensorNode([0])

        # size out < 1
        with pytest.raises(ValidationError):
            TensorNode(lambda t: t, size_out=0)

        # wrong call signature
        with pytest.raises(ValidationError):
            TensorNode(lambda a, b, c: a)

        # returning None
        with pytest.raises(ValidationError):
            TensorNode(lambda x: None)

        # returning non-tensor
        with pytest.raises(ValidationError):
            TensorNode(lambda x: [0])

        # returning wrong number of dimensions
        with pytest.raises(ValidationError):
            TensorNode(lambda t: tf.zeros((2, 2, 2)))

        # correct output
        n = TensorNode(lambda t: tf.zeros((5, 2)))
        assert n.size_out == 2

    # can't run tensornode in regular Nengo simulator
    with nengo.Simulator(net) as sim:
        with pytest.raises(SimulationError):
            sim.step()


    # these tensornodes won't be validated at creation, because size_out
    # is specified. instead the validation occurs when the network is built

    # None output
    with nengo.Network() as net:
        TensorNode(lambda t: None, size_out=2)
    with pytest.raises(ValidationError):
        Simulator(net)

    # wrong number of dimensions
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((1, 2, 2)), size_out=2)
    with pytest.raises(ValidationError):
        Simulator(net)

    # wrong minibatch size
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2)), size_out=2)
    with pytest.raises(ValidationError):
        Simulator(net, minibatch_size=2)

    # wrong output d
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2)), size_out=3)
    with pytest.raises(ValidationError):
        Simulator(net, minibatch_size=3)

    # wrong dtype
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2), dtype=tf.int32), size_out=2)
    with pytest.raises(ValidationError):
        Simulator(net, minibatch_size=3)

    # make sure that correct output _does_ pass
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2), dtype=t.dtype), size_out=2)
    with Simulator(net, minibatch_size=3):
        pass


def test_node(Simulator):
    minibatch_size = 3
    with nengo.Network() as net:
        node0 = TensorNode(lambda t: tf.tile(tf.reshape(t, (1, -1)),
                                             (minibatch_size, 1)))
        node1 = TensorNode(lambda t, x: tf.sin(x), size_in=1)
        nengo.Connection(node0, node1, synapse=None)

        p0 = nengo.Probe(node0)
        p1 = nengo.Probe(node1)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p0], sim.trange()[None, :, None])
    assert np.allclose(sim.data[p1], np.sin(sim.trange()[None, :, None]))


def test_pre_build(Simulator):
    class TestFunc:
        def pre_build(self, size_in, size_out):
            self.weights = tf.Variable(tf.ones((size_in[1], size_out[1])))

        def __call__(self, t, x):
            return tf.matmul(x, tf.cast(self.weights, x.dtype))

    class TestFunc2:
        def pre_build(self, size_in, size_out):
            assert size_in is None
            assert size_out == (1, 1)

        def __call__(self, t):
            return tf.reshape(t, (1, 1))

    with nengo.Network() as net:
        inp = nengo.Node([1, 1])
        test = TensorNode(TestFunc(), size_in=2, size_out=3)
        nengo.Connection(inp, test, synapse=None)
        p = nengo.Probe(test)

        test2 = TensorNode(TestFunc2(), size_out=1)
        p2 = nengo.Probe(test2)

    with Simulator(net) as sim:
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])
        assert np.allclose(sim.data[p2], [[0.001]])

        sim.reset()
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])
        assert np.allclose(sim.data[p2], [[0.001]])


def test_post_build(Simulator):
    class TestFunc:
        def pre_build(self, size_in, size_out):
            self.weights = tf.Variable(tf.zeros((size_in[1], size_out[1])))

        def post_build(self, sess, rng):
            assert isinstance(rng, np.random.RandomState)
            init_op = tf.assign(self.weights, tf.ones((2, 3)))
            sess.run(init_op)

        def __call__(self, t, x):
            return tf.matmul(x, tf.cast(self.weights, x.dtype))

    with nengo.Network() as net:
        inp = nengo.Node([1, 1])
        test = TensorNode(TestFunc(), size_in=2, size_out=3)
        nengo.Connection(inp, test, synapse=None)
        p = nengo.Probe(test)

    with Simulator(net) as sim:
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])

        sim.reset()
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])


def test_reshaped(sess):
    x = tf.zeros((5, 12))

    @reshaped((4, 3))
    def my_func(_, a):
        with tf.control_dependencies([tf.assert_equal(tf.shape(a),
                                                      (5, 4, 3))]):
            return tf.identity(a)

    y = my_func(None, x)

    sess.run(y)


def test_tensor_layer(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node(np.arange(12))

        # check that connection arguments work
        layer0 = tensor_layer(inp, tf.identity, transform=2)

        assert isinstance(layer0, TensorNode)
        p0 = nengo.Probe(layer0)

        # check that arguments are passed to layer function
        layer1 = tensor_layer(
            layer0, lambda x, axis: tf.reduce_sum(x, axis=axis), axis=1,
            shape_in=(2, 6))
        assert layer1.size_out == 6
        p1 = nengo.Probe(layer1)

        # check that ensemble layers work
        layer2 = tensor_layer(layer1, nengo.RectifiedLinear(), gain=[1] * 6,
                              bias=[-20] * 6)
        assert isinstance(layer2, nengo.ensemble.Neurons)
        assert np.allclose(layer2.ensemble.gain, 1)
        assert np.allclose(layer2.ensemble.bias, -20)
        p2 = nengo.Probe(layer2)

        # check that size_in can be inferred from transform
        layer3 = tensor_layer(layer2, lambda x: x,
                              transform=np.ones((1, 6)))
        assert layer3.size_in == 1

        # check that size_in can be inferred from shape_in
        layer4 = tensor_layer(
            layer3, lambda x: x, transform=nengo.dists.Uniform(-1, 1),
            shape_in=(2,))
        assert layer4.size_in == 2

    with Simulator(net, minibatch_size=2) as sim:
        sim.step()

    x = np.arange(12) * 2
    assert np.allclose(sim.data[p0], x)

    x = np.sum(np.reshape(x, (2, 6)), axis=0)
    assert np.allclose(sim.data[p1], x)

    x = np.maximum(x - 20, 0)
    assert np.allclose(sim.data[p2], x)


def test_reuse_vars(Simulator):
    def my_func(_, x):
        # note: the control dependencies thing is due to some weird tensorflow
        # issue with creating variables inside while loops
        with tf.control_dependencies(None):
            w = tf.get_variable("weights", initializer=tf.constant(2.0))

        return x * tf.cast(w, x.dtype)

    with nengo.Network() as net:
        configure_settings(trainable=False)

        inp = nengo.Node([1])
        node = TensorNode(my_func, size_in=1)
        node2 = TensorNode(
            lambda _, x: tf.layers.dense(
                x, units=10, use_bias=False,
                kernel_initializer=tf.constant_initializer(3)),
            size_in=1, size_out=10)
        p = nengo.Probe(node)
        p2 = nengo.Probe(node2)
        nengo.Connection(inp, node, synapse=None)
        nengo.Connection(inp, node2, synapse=None)

    with Simulator(net, unroll_simulation=5) as sim:
        sim.run_steps(5)
        assert np.allclose(sim.data[p], 2)
        assert np.allclose(sim.data[p2], 3)

        with sim.tensor_graph.graph.as_default():
            vars = tf.trainable_variables()

        assert len(vars) == 2
        assert vars[0].get_shape() == ()
        assert sim.sess.run(vars[0]) == 2
        assert vars[1].get_shape() == (1, 10)
        assert np.allclose(sim.sess.run(vars[1]), 3)
