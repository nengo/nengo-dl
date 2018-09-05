TensorNodes
===========

To build :class:`.TensorNode` objects we need to define a new Nengo operator
(:class:`.tensor_node.SimTensorNode`), a build function that adds that operator
into a Nengo graph (:func:`.tensor_node.build_tensor_node`), and a NengoDL
build class that maps that new Nengo operator to TensorFlow operations
(:class:`.tensor_node.SimTensorNodeBuilder`).

.. autoclass:: nengo_dl.tensor_node.SimTensorNode

.. autofunction:: nengo_dl.tensor_node.build_tensor_node

.. autoclass:: nengo_dl.tensor_node.SimTensorNodeBuilder
