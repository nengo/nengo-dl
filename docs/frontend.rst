User documentation
==================

There are two classes that users may need to interact with in order
to access the features of NengoDL: :class:`~.simulator.Simulator` and
:class:`~.tensor_node.TensorNode`.  The former is the main access point for
NengoDL, allowing the user to simulate models, or optimize parameters via
:meth:`.Simulator.train`.  :class:`.TensorNode` is used for
inserting TensorFlow code into a Nengo model.

.. toctree::
    :maxdepth: 1

    simulator
    training
    tensor_node
    extra_objects

