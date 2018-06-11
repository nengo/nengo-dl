TensorNodes
===========

TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.  TensorNodes work very similarly to
a regular :class:`~nengo:nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

:func:`.tensor_layer` is a utility function for constructing TensorNodes,
designed to mimic the layer-based model construction style of many deep
learning packages.

Examples
--------

.. toctree::
    examples/pretrained_model
    examples/spiking_mnist

API
---

.. autoclass:: nengo_dl.tensor_node.TensorNode

.. autofunction:: nengo_dl.tensor_node.tensor_layer
