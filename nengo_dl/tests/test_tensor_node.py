# pylint: disable=missing-docstring

import re
from functools import partial

import nengo
import numpy as np
import pytest
import tensorflow as tf
from nengo.exceptions import SimulationError, ValidationError

from nengo_dl import Layer, TensorNode, configure_settings, tensor_layer


def test_validation(Simulator):
    with nengo.Network() as net:
        # not a callable
        with pytest.raises(ValidationError, match="function or Keras Layer"):
            TensorNode([0])

        # size out < 1
        with pytest.raises(ValidationError, match="must be >= 1"):
            TensorNode(lambda t: t, shape_out=(0,))

        # wrong call signature
        with pytest.raises(ValidationError, match="function produced an error"):
            TensorNode(lambda a, b, c: a)

        # wrong call signature
        with pytest.raises(ValidationError, match="signature produced an error"):
            TensorNode(tf.keras.layers.Lambda(lambda a, b, c: a))

        # returning None
        with pytest.raises(ValidationError, match="must return a Tensor"):
            TensorNode(lambda x: None, shape_in=(1,), pass_time=False)

        # returning non-tensor
        with pytest.raises(ValidationError, match="must return a Tensor"):
            TensorNode(lambda x: [0], shape_in=(1,), pass_time=False)

        # no input
        with pytest.raises(ValidationError, match="either shape_in or pass_time"):
            TensorNode(None, pass_time=False)

        # correct output
        n = TensorNode(lambda t: tf.zeros((5, 2)))
        assert n.size_out == 2

    # can't run tensornode in regular Nengo simulator
    with nengo.Simulator(net) as sim:
        with pytest.raises(SimulationError, match="Cannot call TensorNode output"):
            sim.step()

    # these tensornodes won't be validated at creation, because size_out
    # is specified. instead the validation occurs when the network is built

    # None output
    with nengo.Network() as net:
        TensorNode(lambda t: None, shape_out=(2,))
    with pytest.raises(ValidationError, match="must return a Tensor"):
        Simulator(net)

    # wrong number of dimensions
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((1, 2, 2)), shape_out=(2,))
    with pytest.raises(ValidationError, match="should have size"):
        Simulator(net)

    # wrong minibatch size
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2)), shape_out=(2,))
    with pytest.raises(ValidationError, match="should have batch size"):
        Simulator(net, minibatch_size=2)

    # wrong output d
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2)), shape_out=(3,))
    with pytest.raises(ValidationError, match="should have size"):
        Simulator(net, minibatch_size=3)

    # wrong dtype
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2), dtype=tf.int32), shape_out=(2,))
    with pytest.raises(ValidationError, match="should have dtype"):
        Simulator(net, minibatch_size=3)

    # make sure that correct output _does_ pass
    with nengo.Network() as net:
        TensorNode(lambda t: tf.zeros((3, 2), dtype=t.dtype), shape_out=(2,))
    with Simulator(net, minibatch_size=3):
        pass


def test_node(Simulator):
    minibatch_size = 3
    with nengo.Network() as net:
        node0 = TensorNode(
            lambda t: tf.tile(tf.reshape(t, (1, -1)), (minibatch_size, 1))
        )
        node1 = TensorNode(lambda t, x: tf.sin(x), shape_in=(1,))
        nengo.Connection(node0, node1, synapse=None)

        p0 = nengo.Probe(node0)
        p1 = nengo.Probe(node1)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(10)

    assert np.allclose(sim.data[p0], sim.trange()[None, :, None])
    assert np.allclose(sim.data[p1], np.sin(sim.trange()[None, :, None]))


def test_pre_build(Simulator):
    class TestLayer(tf.keras.layers.Layer):
        def __init__(self, size_out):
            super().__init__()

            self.size_out = size_out

        def build(self, shape_in):
            super().build(shape_in)

            self.w = self.add_weight(
                initializer=tf.initializers.ones(),
                shape=(shape_in[-1], self.size_out),
                name="weights",
            )

        def call(self, x):
            return tf.matmul(x, tf.cast(self.w, x.dtype))

        def compute_output_shape(self, _):
            return tf.TensorShape((None, self.size_out))

    class TestLayer2(tf.keras.layers.Layer):
        def build(self, shape_in):
            # TODO: add this check back in once
            #  https://github.com/tensorflow/tensorflow/issues/32786 is fixed
            # assert shape_in == [(), (1, 1, 1)]
            pass

        def call(self, inputs):
            t, x = inputs
            assert t.shape == ()
            assert x.shape == (1, 1, 1)
            return tf.reshape(t, (1, 1))

    with nengo.Network() as net:
        inp = nengo.Node([1, 1])
        test = TensorNode(TestLayer(3), shape_in=(2,), pass_time=False)
        nengo.Connection(inp, test, synapse=None)
        p = nengo.Probe(test)

        test2 = TensorNode(TestLayer2(), shape_in=(1, 1), pass_time=True)
        p2 = nengo.Probe(test2)

    with Simulator(net) as sim:
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])
        assert np.allclose(sim.data[p2], sim.trange()[:, None])

        sim.reset()
        sim.step()
        assert np.allclose(sim.data[p], [[2, 2, 2]])
        assert np.allclose(sim.data[p2], sim.trange()[:, None])


def test_tensor_layer(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node(np.arange(12))

        # check that connection arguments work
        layer0 = Layer(tf.identity)(inp, transform=2)

        assert isinstance(layer0, TensorNode)
        p0 = nengo.Probe(layer0)

        # check that arguments can be passed to layer function
        layer1 = Layer(partial(lambda x, axis: tf.reduce_sum(x, axis=axis), axis=1))(
            layer0, shape_in=(2, 6)
        )
        assert layer1.size_out == 6
        p1 = nengo.Probe(layer1)

        class TestFunc:
            def __init__(self, axis):
                self.axis = axis

            def __call__(self, x):
                return tf.reduce_sum(x, axis=self.axis)

        layer1b = Layer(TestFunc(axis=1))(layer0, shape_in=(2, 6))
        assert layer1b.size_out == 6

        # check that ensemble layers work
        layer2 = Layer(nengo.RectifiedLinear())(layer1, gain=[1] * 6, bias=[-20] * 6)
        assert isinstance(layer2, nengo.ensemble.Neurons)
        assert np.allclose(layer2.ensemble.gain, 1)
        assert np.allclose(layer2.ensemble.bias, -20)
        p2 = nengo.Probe(layer2)

        # check that size_in can be inferred from transform
        layer3 = Layer(lambda x: x)(layer2, transform=np.ones((1, 6)))
        assert layer3.size_in == 1

        # check that size_in can be inferred from shape_in
        layer4 = Layer(lambda x: x)(
            layer3, transform=nengo.dists.Uniform(-1, 1), shape_in=(2,)
        )
        assert layer4.size_in == 2

        # check that conn is marked non-trainable
        with nengo.Network():
            _, conn = Layer(tf.identity)(inp, return_conn=True)
        assert not net.config[conn].trainable

    with Simulator(net, minibatch_size=2) as sim:
        sim.step()

    x = np.arange(12) * 2
    assert np.allclose(sim.data[p0], x)

    x = np.sum(np.reshape(x, (2, 6)), axis=0)
    assert np.allclose(sim.data[p1], x)

    x = np.maximum(x - 20, 0)
    assert np.allclose(sim.data[p2], x)


def test_reuse_vars(Simulator, pytestconfig):
    class MyLayer(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.w = self.add_weight(
                initializer=tf.initializers.constant(2.0), name="weights"
            )

        def call(self, x):
            return x * tf.cast(self.w, x.dtype)

    with nengo.Network() as net:
        configure_settings(trainable=False)

        inp = nengo.Node([1])
        node = TensorNode(MyLayer(), shape_in=(1,), pass_time=False)
        nengo.Connection(inp, node, synapse=None)

        node2 = Layer(
            tf.keras.layers.Dense(
                units=10,
                use_bias=False,
                kernel_initializer=tf.initializers.constant(3),
                dtype=pytestconfig.getoption("--dtype"),
            )
        )(inp)

        p = nengo.Probe(node)
        p2 = nengo.Probe(node2)

    with Simulator(net, unroll_simulation=5) as sim:
        sim.run_steps(5)
        assert np.allclose(sim.data[p], 2)
        assert np.allclose(sim.data[p2], 3)

        # note: when inference-only=True the weights will be marked as non-trainable
        if sim.tensor_graph.inference_only:
            assert len(sim.tensor_graph.saved_state) == 2
            assert len(sim.keras_model.non_trainable_variables) == 2
            assert len(sim.keras_model.trainable_variables) == 0
            vars = sim.keras_model.non_trainable_variables
        else:
            assert len(sim.tensor_graph.saved_state) == 2
            assert len(sim.keras_model.non_trainable_variables) == 0
            assert len(sim.keras_model.trainable_variables) == 2
            vars = sim.keras_model.trainable_variables

        assert len(vars) == 2
        assert vars[0].shape == ()
        assert tf.keras.backend.get_value(vars[0]) == 2
        assert vars[1].shape == (1, 10)
        assert np.allclose(tf.keras.backend.get_value(vars[1]), 3)


def test_tensor_layer_deprecation(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        with pytest.warns(DeprecationWarning, match="nengo_dl.Layer instead"):
            out = tensor_layer(inp, lambda x: x + 1)
        p = nengo.Probe(out)

    with Simulator(net) as sim:
        sim.run_steps(5)

    assert np.allclose(sim.data[p], 1)


def test_nested_layer(Simulator, pytestconfig):
    class MyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.layer = tf.keras.layers.Dense(
                10,
                kernel_initializer=tf.initializers.ones(),
                dtype=pytestconfig.getoption("--dtype"),
            )

        def build(self, input_shapes):
            super().build(input_shapes)

            if not self.layer.built:
                self.layer.build(input_shapes)

        def call(self, inputs):
            return self.layer(inputs)

        def compute_output_shape(self, input_shape):
            return self.layer.compute_output_shape(input_shape)

    with nengo.Network() as net:
        inp = nengo.Node([1])
        node = Layer(MyLayer())(inp)
        p = nengo.Probe(node)

    # do this twice to test layer rebuilding
    for _ in range(2):
        with Simulator(net) as sim:
            sim.run_steps(5)

        assert np.allclose(sim.data[p], 1)


def test_unspecified_shape():
    with nengo.Network():
        inp = nengo.Node([0])

        with pytest.raises(ValidationError, match="must be an int"):
            TensorNode(lambda x: x, shape_in=(None, 1))

        with pytest.raises(ValidationError, match="must be an int"):
            TensorNode(lambda x: x, shape_out=(None, 1))

        with pytest.raises(ValidationError, match="must be an int"):
            Layer(lambda x: x)(inp, shape_in=(None, 1))

        with pytest.raises(ValidationError, match="must be an int"):
            Layer(lambda x: x)(inp, shape_out=(None, 1))


def test_str():
    with nengo.Network():

        def test_func(x):
            return x

        node = nengo.Node([0])
        layer_w_func = Layer(test_func)
        assert str(layer_w_func) == "Layer(test_func)"
        assert "TensorNode (unlabeled)" in str(layer_w_func(node))
        # nengo <= 3.1 uses double quotes around the name, after uses single quotes
        assert re.compile("<TensorNode ['\"]test_func['\"]>").match(
            str(layer_w_func(node, label="test_func"))
        )

        class TestLayer(tf.keras.layers.Layer):
            pass

        assert str(Layer(TestLayer())) == "Layer(test_layer)"

        assert str(Layer(tf.keras.layers.Dense(units=10))) == "Layer(dense)"


@pytest.mark.training
def test_training_arg(Simulator):
    class TrainingLayer(tf.keras.layers.Layer):
        def __init__(self, expected):
            super().__init__()

            self.expected = expected

        def call(self, inputs, training=None):
            tf.assert_equal(tf.cast(training, tf.bool), self.expected)
            return tf.reshape(inputs, (1, 1))

    with nengo.Network() as net:
        node = TensorNode(TrainingLayer(expected=False), shape_in=None, shape_out=(1,))
        nengo.Probe(node)

    with Simulator(net) as sim:
        sim.predict(n_steps=10)

    with Simulator(net) as sim:
        sim.compile(optimizer=tf.optimizers.SGD(0), loss=tf.losses.mse)
        node.tensor_func.expected = True
        sim.fit(n_steps=10, y=np.zeros((1, 1, 1)))

    with net:
        configure_settings(learning_phase=True)
    with Simulator(net) as sim:
        sim.predict(n_steps=10)


def test_wrapped_model(Simulator, pytestconfig):
    inp0 = tf.keras.Input((1,))
    out0 = tf.keras.layers.Dense(units=10, dtype=pytestconfig.getoption("--dtype"))(
        inp0
    )
    model0 = tf.keras.Model(inp0, out0)

    model0.compile(loss="mse", metrics=["accuracy"])

    class KerasWrapper(tf.keras.layers.Layer):
        def __init__(self, model):
            super().__init__()

            self.model = model

        def build(self, input_shape):
            super().build(input_shape)

            self.model = tf.keras.models.clone_model(self.model)

        def call(self, inputs):
            return self.model(inputs)

    with nengo.Network() as net:
        layer = KerasWrapper(model0)
        keras_node = TensorNode(layer, shape_in=(1,), shape_out=(10,), pass_time=False)
        nengo.Probe(keras_node)

    with Simulator(net) as sim:
        # this caused an error at one point, so testing it here (even though it
        # seems irrelevant)
        sim.compile(loss="mse", metrics=["accuracy"])

        # assert layer.model.layers[0] in sim.tensor_graph.layers
        assert layer.model.weights[0] in sim.keras_model.weights
