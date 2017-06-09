TensorNodes
===========

TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.  TensorNodes work very similarly to a
regular :class:`~nengo:nengo.Node`, except instead of executing arbitrary
Python code they execute arbitrary TensorFlow code.

See `this example <pretrained_model>`_ for a demonstration of usage.

.. autoclass:: nengo_dl.tensor_node.TensorNode

.. autofunction:: nengo_dl.tensor_node.tensor_layer