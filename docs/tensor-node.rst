TensorNodes
===========

TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.  TensorNodes work very similarly to
a regular `~nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

The TensorFlow code is defined in a function or callable class
(``tensor_func``).  This function accepts the current simulation time as
input, or the current simulation time and a Tensor ``x`` if
``node.size_in > 0``.  ``x`` will have shape
``(sim.minibatch_size, node.size_in``), and the function should return a
Tensor with shape ``(sim.minibatch_size, node.size_out)``.
``node.size_out`` will be inferred by calling the function once and
checking the output, if it isn't set when the Node is created.

.. code-block:: python

    def tensor_func(t [, x]):
        print(t)  # current simulation time
        print(x)  # input on current timestep (minibatch_size, node.size_in)

        return x + 1

If ``tensor_func`` has a ``pre_build`` attribute, that function will be
called once when the model is constructed.  This can be used to compute any
constant values or set up variables -- things that don't need to
execute every simulation timestep.

.. code-block:: python

    def pre_build(shape_in, shape_out):
        print(shape_in)  # (minibatch_size, node.size_in)
        print(shape_out)  # (minibatch_size, node.size_out)

If ``tensor_func`` has a ``post_build`` attribute, that function will be
called after the simulator is created and whenever it is reset.  This can
be used to set any random elements in the TensorNode or perform any
post-initialization setup required by the node (e.g., loading pretrained
weights).

.. code-block:: python

    def post_build(sess, rng):
        print(sess)  # the TensorFlow simulation session object
        print(rng)  # random number generator (np.random.RandomState)

`.tensor_layer` is a utility function for constructing TensorNodes,
designed to mimic the layer-based model construction style of many deep
learning packages.  It combines the creation of a `.TensorNode` or
`~nengo.Ensemble` and a `~nengo.Connection` in a single step.

See the :ref:`TensorNode API <tensornode-api>` for more details, or the
examples below for demonstrations of using TensorNodes in practice.

Examples
--------

* :doc:`examples/from-nengo`
* :doc:`examples/from-tensorflow`
* :doc:`examples/pretrained-model`
* :doc:`examples/spiking-mnist`
