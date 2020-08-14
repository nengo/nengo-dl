TensorNodes
===========

`TensorNodes <.TensorNode>` allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.  TensorNodes work very similarly to
a regular `~nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

The TensorFlow code is defined in a function or callable class.
This function accepts the current simulation time (if ``pass_time=True``) and/or an
input Tensor ``x`` (if ``node.shape_in`` is specified).  ``x`` will have shape
``(sim.minibatch_size,) + node.shape_in``, and the function should return a
Tensor with shape ``(sim.minibatch_size,) + node.shape_out``.
``node.shape_out`` will be inferred by calling the function once and
checking the output, if it isn't set when the Node is created.

.. testsetup::

    net = nengo.Network()
    net.__enter__()

.. testcode::

    def tensor_func(t, x):
        print(t)  # current simulation time
        print(x)  # input on current timestep

        return x + 1

    my_node = nengo_dl.TensorNode(tensor_func, shape_in=(1,))

.. testoutput::
    :hide:

    ...

TensorNodes can also be used with `Keras Layers
<https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_, by passing an
instantiated Layer to the TensorNode. Since Keras layers typically don't take the
simulation time as input, we can use the ``pass_time=False`` parameter to only pass
``x``.

.. testcode::

    my_node = nengo_dl.TensorNode(tf.keras.layers.Dense(units=10),
                                  shape_in=(1,), pass_time=False)

This also means that we can use custom Keras layers to implement more complicated
TensorNode behaviour. For example, if a TensorNode requires internal parameter
variables, those can be created inside a Layer's ``build`` function.

.. testcode::

    class MyLayer(tf.keras.layers.Layer):
        def build(self, input_shapes):
            self.w = self.add_weight()

        def call(self, inputs):
            return inputs * self.w

    my_node = nengo_dl.TensorNode(MyLayer(), shape_in=(1,), pass_time=False)

See the
`TensorFlow documentation
<https://www.tensorflow.org/tutorials/customization/custom_layers>`_ for more details
on creating custom Layers.

Once created, a TensorNode can then be used in a Nengo network just like any other
Nengo object (for example, it can receive input from Connections or have its output
recorded via Probes)

.. testcode::

    inp = nengo.Node(output=np.sin)
    conn = nengo.Connection(inp, my_node)
    probe = nengo.Probe(my_node)

NengoDL also provides another syntax for creating TensorNodes, designed for users more
familiar with the `Keras functional API
<https://www.tensorflow.org/guide/keras/functional>`_.  This is the `.Layer`
class. Under the hood, this is just a different way of creating TensorNodes, it simply
combines the creation of a TensorNode and a Connection from some input object to that
TensorNode in a single step.

For example, in Keras we would create a Layer like

.. testcode::

    x = tf.keras.Input(shape=(1,))
    y = tf.keras.layers.Dense(units=10)(x)

The equivalent, using ``nengo_dl.Layer``, would be

.. testcode::

    x = nengo.Node([0])
    y = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)

Which, under the hood, is equivalent to

.. testcode::

    x = nengo.Node([0])
    y = nengo_dl.TensorNode(
        tf.keras.layers.Dense(units=10), pass_time=False, shape_in=(1,))
    nengo.Connection(x, y, synapse=None)

See the :ref:`TensorNode API <tensornode-api>` for more details, or the
examples below for demonstrations of using TensorNodes in practice.

.. testcleanup::

    net.__exit__(None, None, None)

Examples
--------

* :doc:`examples/from-nengo`
* :doc:`examples/from-tensorflow`
* :doc:`examples/tensorflow-models`
