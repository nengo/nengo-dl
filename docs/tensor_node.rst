TensorNodes
===========

TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.  TensorNodes work very similarly to
a regular :class:`~nengo:nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

:func:`.tensor_layer` is a utility function for constructing TensorNodes,
designed to mimic the layer-based model construction style of many deep
learning packages.

See the :ref:`TensorNode API <tensornode-api>` for more details, or the
examples below for demonstrations of using TensorNodes in practice.

Examples
--------

* :doc:`examples/pretrained_model`
* :doc:`examples/spiking_mnist`
