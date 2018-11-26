User guide
==========

There are two classes that users may need to interact with in order
to access the features of NengoDL: `~.simulator.Simulator` and
`~.tensor_node.TensorNode`.  The former is the main access point for
NengoDL, allowing the user to simulate models, or optimize parameters via
`.Simulator.train`.  `.TensorNode` is used for
inserting TensorFlow code into a Nengo model.

.. toctree::
    :maxdepth: 1

    simulator
    training
    tensor-node
    config

