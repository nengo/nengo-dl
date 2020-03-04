Converter
=========

NengoDL allows Keras models to be integrated directly into a NengoDL network, through
the use of `nengo_dl.TensorNode` (see :doc:`this example <examples/tensorflow-models>`).
However, rather than keeping a model defined in Keras/TensorFlow, sometimes we may
want to convert it to a native Nengo model
(composed only of Nengo objects, such as `nengo.Ensemble` and `nengo.Connection`). One
reason we might want to do this is if we want to be able to run the model on different
Nengo backends. Keras models can only run in the NengoDL Simulator, but a native Nengo
model can run in any Nengo Simulator (including backends for specialized neuromorphic
hardware).  Another reason we might want to convert a Keras model to a Nengo network
is so that we can use Nengo's spiking neuron types rather than the standard
Keras/TensorFlow activation functions.

Whatever the motivation, the `nengo_dl.Converter` tool is designed to automate the
translation from Keras to Nengo as much as possible.  To use it, simply instantiate a
Keras model and then pass it to the Converter:

.. testcode::

    inp = tf.keras.Input(shape=(28, 28, 3))
    conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(inp)
    flat = tf.keras.layers.Flatten()(conv)
    dense = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(flat)

    model = tf.keras.Model(inputs=inp, outputs=dense)

    converter = nengo_dl.Converter(model)

The `.Converter` object will have a ``.net`` attribute which is the converted Nengo
network. This can then be passed to a Nengo Simulator. For example, we could run that
converted Keras network in the standard Nengo (not NengoDL) Simulator:

.. testcode::

    with nengo.Simulator(converter.net) as sim:
        sim.step()

.. testoutput::
    :hide:

    ...

The converter also has ``.inputs`` and ``.outputs`` attributes which can be used
to look up the Nengo Nodes/Probes corresponding to the Keras model inputs/outputs.
For example, to see the output of the converted network:

.. testcode::

    print(sim.data[converter.outputs[model.output]])

.. testoutput::

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

`nengo_dl.Converter` has many options that control different aspects of the conversion
process.  For example, we could change all the ``tf.nn.relu`` activation functions
in the Keras model to spiking neurons:

.. testcode::

    converter = nengo_dl.Converter(
        model, swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()})

We will not go into all the available options here; see the
`API docs <nengo_dl.Converter>` for more details.

Keep in mind that the Converter is designed to automate the translation process as much
as possible, but it will not work for all possible Keras networks. In general, the
Converter will fall back to using TensorNodes for any elements that cannot be converted
to native Nengo objects (set ``allow_fallback=False`` if you would like this to be an
error instead).  The `.Converter.verify` function can be used to check that the output
of the Nengo network matches the output of the Keras model.

Extending the converter
-----------------------

Since we know that it will not be possible to automatically translate all possible
Keras models, the Converter has been designed to be easily extensible. So if you have
a model containing elements that the Converter does not know how to translate,
you can augment the Converter with custom conversion logic.

For example, suppose we have a model containing a custom Keras layer:

.. testcode::

    class AddOne(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs + 1

    inp = tf.keras.Input(shape=(1,))
    dense = tf.keras.layers.Dense(units=10)(inp)
    addone = AddOne()(dense)

    model = tf.keras.Model(inputs=inp, outputs=addone)

`.Converter` would fail to convert this model to native Nengo objects, since it does
not know how to translate the ``AddOne`` layer:

.. testcode::

    converter = nengo_dl.Converter(model, allow_fallback=False)

.. testoutput::

    Traceback (most recent call last):
    ...
    TypeError: Unable to convert layer add_one to native Nengo objects


We could set ``allow_fallback=True`` to use a `.TensorNode` to implement the ``AddOne``
layer, but suppose we want to use native Nengo objects instead. We need to make a custom
`nengo_dl.converter.LayerConverter` subclass, which contains the logic for translating
an ``AddOne`` layer.  Note that this may require some knowledge of how Keras layers work
under the hood, which is not extensively documented. Your best bet may be to look at
the existing :ref:`LayerConverter classes <layer-converter-api>` to find a similar
Layer type to start from.

As an example, here is how we might translate the ``AddOne`` layer:

.. testcode::

    @nengo_dl.Converter.register(AddOne)
    class ConvertAddOne(nengo_dl.converter.LayerConverter):
        def convert(self, node_id):
            # create a Nengo object representing the output of this layer node
            output = self.add_nengo_obj(node_id)

            # connect up the input of the layer node
            self.add_connection(node_id, output)

            # create a node to output a constant 1 vector
            bias = nengo.Node([1] * output.size_in)

            # connect up the bias node to the output (thereby adding one to the
            # input values)
            conn = nengo.Connection(bias, output, synapse=None)

            # mark the above connection as non-trainable (since we didn't have any
            # trainable parameters in the AddOne layer, we don't want any in the
            # converted Nengo equivalent either)
            self.converter.net.config[conn].trainable = False

            return output

And now if we try to convert the original Keras model, we can see that it is
successfully transformed into a native Nengo network:

.. testcode::

    converter = nengo_dl.Converter(model, allow_fallback=False)
    assert converter.verify()
