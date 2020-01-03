# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization

from nengo_dl import config, converter, utils


def _test_convert(inputs, outputs, allow_fallback=False, inp_vals=None):
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    conv = converter.Converter(model, allow_fallback=allow_fallback)

    assert conv.verify(training=False, inputs=inp_vals)
    assert conv.verify(training=True, inputs=inp_vals)


def test_dense(seed):
    tf.random.set_seed(seed)

    inp = x = tf.keras.Input(shape=(2, 2))
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=5, activation="relu")(x)
    x = tf.keras.layers.Dense(units=6, bias_initializer=tf.initializers.ones())(x)
    x = tf.keras.layers.Dense(units=7, use_bias=False)(x)

    _test_convert(inp, x)


def test_conv(seed):
    tf.random.set_seed(seed)

    inp = x = tf.keras.Input(shape=(16, 4))
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=2, padding="same")(x)
    x = tf.keras.layers.Reshape((4, 4, 8))(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation="sigmoid")(x)
    x = tf.keras.layers.Reshape((4, 4, 2))(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=1, strides=2)(x)
    x = tf.keras.layers.Reshape((2, 2, 2, 4))(x)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=2)(x)
    if utils.tf_gpu_installed:
        x = tf.keras.layers.Reshape((2, 4, 4))(x)
        x = tf.keras.layers.Conv2D(
            filters=4, kernel_size=3, data_format="channels_first"
        )(x)

    _test_convert(inp, x)


def test_activation():
    inp = x = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    x = tf.keras.layers.Activation(tf.nn.sigmoid)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    _test_convert(inp, x)

    inp = x = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

    model = tf.keras.Model(inp, x)

    with pytest.raises(TypeError, match="Unsupported activation type"):
        converter.Converter(model, allow_fallback=False)

    with pytest.warns(UserWarning, match="falling back to a TensorNode"):
        conv = converter.Converter(model, allow_fallback=True)
    assert conv.verify(training=False)
    assert conv.verify(training=True)


@pytest.mark.training
def test_fallback(Simulator):
    inp = x = tf.keras.Input(shape=(2, 2))

    class MyLayer(tf.keras.layers.Layer):
        def build(self, input_shapes):
            super().build(input_shapes)
            self.kernel = self.add_weight(
                shape=(), initializer=tf.initializers.RandomUniform()
            )

        def call(self, inputs):
            assert inputs.shape[1:] == (2, 2)
            return tf.reshape(
                inputs * tf.cast(self.kernel, inputs.dtype), shape=(-1, 4)
            )

    layer = MyLayer()
    x = layer(x)
    x = tf.keras.layers.Reshape((2, 2))(x)
    x = layer(x)
    x = tf.keras.layers.Reshape((2, 2))(x)
    x = layer(x)

    model = tf.keras.Model(inp, x)
    conv = converter.Converter(model, allow_fallback=True)

    with Simulator(conv.net) as sim:
        # check that weights are being shared correctly
        assert len(sim.keras_model.trainable_weights) == 1
        assert sim.keras_model.trainable_weights[0].shape == ()

    assert conv.verify(training=False)
    assert conv.verify(training=True)


def test_add():
    inp = [tf.keras.Input(shape=(2, 2)), tf.keras.Input(shape=(2, 2))]
    out = tf.keras.layers.Add()(inp)

    _test_convert(inp, out)


def test_average():
    inp = [tf.keras.Input(shape=(2, 2)), tf.keras.Input(shape=(2, 2))]
    out = tf.keras.layers.Average()(inp)

    _test_convert(inp, out)


def test_concatenate(rng):
    inp = [
        tf.keras.Input(shape=(1, 4)),
        tf.keras.Input(shape=(2, 4)),
        tf.keras.Input(shape=(3, 5)),
    ]
    x = tf.keras.layers.Concatenate(axis=1)(inp[:2])
    x = tf.keras.layers.Concatenate(axis=-1)([x, inp[2]])

    _test_convert(
        inp,
        x,
        inp_vals=[
            rng.uniform(size=(5, 1, 4)),
            rng.uniform(size=(5, 2, 4)),
            rng.uniform(size=(5, 3, 5)),
        ],
    )

    inp = [tf.keras.Input(shape=(1,)), tf.keras.Input(shape=(1,))]
    x = tf.keras.layers.Concatenate(axis=0)(inp)
    model = tf.keras.Model(inp, x)

    with pytest.raises(TypeError, match="concatenate along batch dimension"):
        converter.Converter(model, allow_fallback=False)


def test_zero_padding():
    inp = x = tf.keras.Input(shape=(14, 2))
    x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
    x = tf.keras.layers.Reshape((4, 4, 2))(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((2, 0), (0, 4)))(x)
    x = tf.keras.layers.Reshape((2, 2, 2, 12))(x)
    x = tf.keras.layers.ZeroPadding3D(padding=(1, 0, 3))(x)
    if utils.tf_gpu_installed:
        x = tf.keras.layers.Reshape((12, 8, 8))(x)
        x = tf.keras.layers.ZeroPadding2D(padding=1, data_format="channels_first")(x)

    _test_convert(inp, x)


def test_batch_normalization(rng):
    inp = x = tf.keras.Input(shape=(4, 4))
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.BatchNormalization(axis=2)(x)
    x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    inp_vals = [rng.uniform(size=(2, 4, 4))]

    # test using tensornode fallback
    # TODO: there is some bug with using batchnormalization layers inside
    #  nengo_dl.Layers in general (unrelated to converting)
    # conv = convert.Converter(allow_fallback=True, inference_only=False)
    # with pytest.warns(UserWarning, match="falling back to nengo_dl.Layer"):
    #     net = conv.convert(model)
    #
    # assert conv.verify(model, net, training=False, inputs=inp_vals)
    # assert conv.verify(model, net, training=True, inputs=inp_vals)

    # test actually converting to nengo objects
    conv = converter.Converter(model, allow_fallback=False, inference_only=True)

    assert conv.verify(training=False, inputs=inp_vals)
    with pytest.raises(ValueError, match="number of trainable parameters"):
        # we don't expect the verification to pass for training=True, since we froze
        # the batch normalization in the nengo network (but not the keras model)
        conv.verify(training=True, inputs=inp_vals)

    # error if inference_only=False
    with pytest.raises(TypeError, match="unless inference_only=True"):
        converter.Converter(model, allow_fallback=False, inference_only=False)


def test_relu():
    inp = tf.keras.Input(shape=(10,))
    out = tf.keras.layers.ReLU()(inp)

    _test_convert(inp, out, inp_vals=[np.linspace(-1, 1, 10)[None, :]])


def test_avg_pool(rng):
    n_filters = 6
    inp = tf.keras.Input(shape=(64, n_filters))
    outputs = [tf.keras.layers.AvgPool1D()(inp)]

    x = tf.keras.layers.Reshape((8, 8, n_filters))(inp)
    outputs.append(tf.keras.layers.AvgPool2D()(x))

    x = tf.keras.layers.Reshape((8, 8, n_filters))(inp)
    outputs.append(tf.keras.layers.AvgPool2D(pool_size=3, strides=2)(x))

    x = tf.keras.layers.Reshape((4, 4, 4, n_filters))(inp)
    outputs.append(tf.keras.layers.AvgPool3D()(x))

    if utils.tf_gpu_installed:
        x = tf.keras.layers.Reshape((n_filters, 8, 8))(inp)
        outputs.append(
            tf.keras.layers.AvgPool2D(
                pool_size=3, strides=2, data_format="channels_first"
            )(x)
        )

    _test_convert(inp, outputs, inp_vals=[rng.uniform(size=(3, 64, n_filters))])


def test_global_avg_pool(rng):
    n_filters = 6
    inp = tf.keras.Input(shape=(64, n_filters))
    out0 = tf.keras.layers.GlobalAvgPool1D()(inp)

    x = tf.keras.layers.Reshape((8, 8, n_filters))(inp)
    out1 = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Reshape((n_filters, 8, 8))(inp)
    out2 = tf.keras.layers.GlobalAvgPool2D(data_format="channels_first")(x)

    x = tf.keras.layers.Reshape((4, 4, 4, n_filters))(inp)
    out3 = tf.keras.layers.GlobalAvgPool3D()(x)

    _test_convert(
        inp, [out0, out1, out2, out3], inp_vals=[rng.uniform(size=(3, 64, n_filters))]
    )


@pytest.mark.training
def test_densenet(Simulator, seed):
    tf.random.set_seed(seed)
    model = tf.keras.applications.densenet.DenseNet121(
        weights=None, include_top=False, input_shape=(112, 112, 3)
    )

    conv = converter.Converter(
        model, allow_fallback=False, max_to_avg_pool=True, inference_only=True
    )

    keras_params = 0
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            for w in layer._trainable_weights:
                keras_params += np.prod(w.shape)

    # note: we don't expect any of the verification checks to pass, due to the
    # max_to_avg_pool swap, so just checking that the network structure has been
    # recreated
    with conv.net:
        # undo the inference_only=True so that parameters will be marked as
        # trainable (so that the check below will work)
        config.configure_settings(inference_only=False)

    with Simulator(conv.net) as sim:
        assert keras_params == sum(
            np.prod(w.shape) for w in sim.keras_model.trainable_weights
        )


def test_nested_network(seed):
    tf.random.set_seed(seed)

    inp = x = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(units=10)(x)

    sub_sub_inp = tf.keras.Input(shape=(10, 1, 1))
    sub_sub_out = tf.keras.layers.Flatten()(sub_sub_inp)
    sub_sub_model = tf.keras.Model(sub_sub_inp, sub_sub_out)

    sub_inp0 = tf.keras.Input(shape=(10, 1))
    sub_inp1 = tf.keras.Input(shape=(10, 1))
    sub_add = tf.keras.layers.Add()([sub_inp0, sub_inp1])
    sub_out0 = tf.keras.layers.Reshape((10,))(sub_add)
    sub_out1 = tf.keras.layers.Reshape((10, 1, 1))(sub_add)
    sub_out1 = sub_sub_model(sub_out1)
    sub_model = tf.keras.Model(
        [sub_inp0, sub_inp1], [sub_out0, sub_out1], name="submodel"
    )

    x = sub_model([x, x])
    x = sub_model(x)

    _test_convert(inp, x)


def test_repeated_layers(seed):
    inp = x = tf.keras.layers.Input(shape=(4,))
    layer = tf.keras.layers.Dense(units=4)
    x = layer(x)
    x = layer(x)
    x = layer(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    conv = converter.Converter(model, allow_fallback=False, split_shared_weights=True)

    assert conv.verify(training=False)
    with pytest.raises(ValueError, match="number of trainable parameters"):
        # we don't expect the verification to pass for training=True, since we
        # split up the shared weights in the nengo network
        conv.verify(training=True)

    # error if split_shared_weights=False
    with pytest.raises(
        ValueError, match="not supported unless split_shared_weights=True"
    ):
        converter.Converter(model, split_shared_weights=False)


def test_fan_in_out():
    a = tf.keras.layers.Input(shape=(1,))
    b = tf.keras.layers.Reshape((1, 1))
    c = tf.keras.layers.Reshape((1, 1, 1))
    d = tf.keras.layers.Reshape((1, 1, 1))

    x = d(a)
    y = d(x)
    z = d(y)

    outputs = [b(a), c(a), d(a), x, y, z]

    _test_convert(a, outputs)


def test_sequential(seed):
    tf.random.set_seed(seed)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(4,)))
    model.add(tf.keras.layers.Dense(32))

    conv = converter.Converter(model, allow_fallback=False)
    assert conv.verify(training=False)
    assert conv.verify(training=True)


@pytest.mark.parametrize(
    "keras_activation, nengo_activation, swap",
    [
        (tf.nn.sigmoid, nengo.RectifiedLinear(), {tf.nn.sigmoid: tf.nn.relu}),
        (
            tf.nn.relu,
            nengo.SpikingRectifiedLinear(),
            {tf.nn.relu: nengo.SpikingRectifiedLinear()},
        ),
        (tf.nn.relu, nengo.LIF(), {nengo.RectifiedLinear(): nengo.LIF()}),
    ],
)
def test_activation_swap(Simulator, keras_activation, nengo_activation, swap, rng):
    inp = x = tf.keras.Input(shape=(100,))
    x = tf.keras.layers.Activation(activation=keras_activation)(x)
    x = tf.keras.layers.Dense(
        units=100,
        activation=keras_activation,
        kernel_initializer=tf.initializers.constant(np.eye(100)),
    )(x)
    model = tf.keras.Model(inp, x)

    conv = converter.Converter(model, allow_fallback=False, swap_activations=swap)

    with nengo.Network() as net:
        net.config[nengo.Ensemble].neuron_type = nengo_activation
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        inp = nengo.Node(np.zeros(100))
        ens0 = nengo.Ensemble(100, 1)
        nengo.Connection(inp, ens0.neurons)

        ens1 = nengo.Ensemble(100, 1)
        nengo.Connection(ens0.neurons, ens1.neurons)

        p = nengo.Probe(ens1.neurons)

    inp_vals = rng.uniform(size=(20, 50, 100))

    with Simulator(net) as sim0:
        data0 = sim0.predict(inp_vals)

    with Simulator(conv.net) as sim1:
        data1 = sim1.predict(inp_vals)

    assert np.allclose(data0[p], data1[conv.outputs[model.outputs[0]]])


def test_max_pool(rng):
    inp = x = tf.keras.Input(shape=(4, 4, 2))
    x = tf.keras.layers.MaxPool2D()(x)

    model = tf.keras.Model(inp, x)

    with pytest.warns(UserWarning, match="consider setting max_to_avg_pool=True"):
        conv = converter.Converter(model, max_to_avg_pool=False)

    assert conv.verify(training=False)
    assert conv.verify(training=True)

    # can convert to avg pool, but then we don't expect output to match
    conv = converter.Converter(model, max_to_avg_pool=True, allow_fallback=False)
    with pytest.raises(ValueError, match="does not match output"):
        conv.verify(training=False, inputs=[rng.uniform(size=(2, 4, 4, 2))])


def test_unsupported_args():
    inp = x = tf.keras.Input(shape=(4, 1))
    x = tf.keras.layers.Conv1D(1, 1, kernel_regularizer=tf.keras.regularizers.l1(0.1))(
        x
    )

    model = tf.keras.Model(inp, x)

    with pytest.raises(
        TypeError,
        match="kernel_regularizer has value .* != None.*unless inference_only=True",
    ):
        converter.Converter(model, allow_fallback=False)

    with pytest.warns(
        UserWarning,
        match="kernel_regularizer has value .* != None.*unless inference_only=True",
    ):
        conv = converter.Converter(model, allow_fallback=True)
    assert conv.verify(training=False)
    assert conv.verify(training=True)

    inp = x = tf.keras.Input(shape=(4, 1))
    x = tf.keras.layers.Conv1D(1, 1, dilation_rate=(2,))(x)

    model = tf.keras.Model(inp, x)

    with pytest.raises(TypeError, match=r"dilation_rate has value \(2,\) != \(1,\)"):
        converter.Converter(model, allow_fallback=False)

    with pytest.warns(UserWarning, match=r"dilation_rate has value \(2,\) != \(1,\)"):
        conv = converter.Converter(model, allow_fallback=True)
    assert conv.verify(training=False)
    assert conv.verify(training=True)


def test_custom_converter():
    class AddOne(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs + 1

    @converter.Converter.register(AddOne)
    class ConvertAddOne(converter.LayerConverter):
        def convert(self, node_id):
            output = self.add_nengo_obj(node_id)
            self.add_connection(node_id, output)

            bias = nengo.Node([1] * output.size_in)
            conn = nengo.Connection(bias, output, synapse=None)
            self.converter.net.config[conn].trainable = False

            return output

    inp = x = tf.keras.Input(shape=(2, 2))
    x = AddOne()(x)

    _test_convert(inp, x)

    with pytest.warns(UserWarning, match="already has a converter"):
        converter.Converter.register(AddOne)(ConvertAddOne)


def test_input():
    inp = x = tf.keras.Input(shape=(None, None, 2))
    model = tf.keras.Model(inp, x)

    with pytest.raises(ValueError, match="must be fully specified"):
        converter.Converter(model)


def test_nested_input():
    subinputs = (
        tf.keras.Input(shape=(2,), name="sub0"),
        tf.keras.Input(shape=(2,), name="sub1"),
    )
    x_0 = x = tf.keras.layers.Activation(tf.nn.relu)(subinputs[0])
    x = tf.keras.layers.Concatenate()([x, subinputs[1]])
    submodel = tf.keras.Model(subinputs, (x_0, x))

    inputs = (
        tf.keras.Input(shape=(2,), name="in0"),
        tf.keras.Input(shape=(2,), name="in1"),
    )
    x_0, x = submodel(inputs)
    x = tf.keras.layers.Concatenate()([x, x_0])

    _test_convert(inputs, x)
