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

.. code-block:: python

    def tensor_func(t, x):
        print(t)  # current simulation time
        print(x)  # input on current timestep

        return x + 1

    my_node = nengo_dl.TensorNode(tensor_func, size_in=1)

TensorNodes can also be used with `Keras Layers
<https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_, by passing an
instantiated Layer to the TensorNode. Since Keras layers typically don't take the
simulation time as input, we can use the ``pass_time=False`` parameter to only pass
``x``.

.. code-block:: python

    my_node = nengo_dl.TensorNode(tf.keras.layers.Dense(units=10),
                                  size_in=1, pass_time=False)

This also means that we can use custom Keras layers to implement more complicated
TensorNode behaviour. For example, if a TensorNode requires internal parameter
variables, those can be created inside a Layer's ``build`` function.

.. code-block:: python

    class MyLayer(tf.keras.layers.Layer):
        def build(self, input_shapes):
            self.w = self.add_weight(...)

        def call(self, inputs):
            return inputs * self.w

    my_node = nengo_dl.TensorNode(MyLayer(), size_in=1, pass_time=False)

See the
`TensorFlow documentation
<https://www.tensorflow.org/tutorials/customization/custom_layers>`_ for more details
on creating custom Layers.

Once created, a TensorNode can then be used in a Nengo network just like any other
Nengo object (for example, it can receive input from Connections or have its output
recorded via Probes)

.. code-block:: python

    conn = nengo.Connection(..., my_node)
    probe = nengo.Probe(my_node)

NengoDL also provides another syntax for creating TensorNodes, designed for users more
familiar with the `Keras functional API
<https://www.tensorflow.org/guide/keras/functional>`_.  This is the `.Layer`
class. Under the hood, this is just a different way of creating TensorNodes, it simply
combines the creation of a TensorNode and a Connection from some input object to that
TensorNode in a single step.

For example, in Keras we would create a Layer like

.. code-block:: python

    x = ...
    y = tf.keras.Layers.Dense(units=10)(x)

The equivalent, using ``nengo_dl.Layer``, would be

.. code-block:: python

    x = ...
    y = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)

Which, under the hood, is equivalent to

.. code-block:: python

    x = ...
    y = nengo_dl.TensorLayer(tf.keras.layers.Dense(units=10), pass_time=False)
    nengo.Connection(x, y, synapse=None)

See the :ref:`TensorNode API <tensornode-api>` for more details, or the
examples below for demonstrations of using TensorNodes in practice.

Examples
--------

* :doc:`examples/from-nengo`
* :doc:`examples/from-tensorflow`
* :doc:`examples/tensorflow-models`
