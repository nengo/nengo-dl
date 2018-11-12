API reference
=============

This section details the functions, modules, and classes available in
NengoDL.  For a more in-depth description of how to use these objects,
see the :doc:`user-guide`.

Users
-----

These are objects that users may interact with.

.. _sim-api:

Simulator
^^^^^^^^^

The Simulator class is the access point for the main features of
NengoDL, including `running <.Simulator.run_steps>` and
`training <.Simulator.train>` a model.

.. autoclass:: nengo_dl.simulator.Simulator
    :exclude-members: unsupported, dt

.. autoclass:: nengo_dl.simulator.SimulationData
    :special-members:
    :exclude-members: __weakref__

.. _tensornode-api:

TensorNodes
^^^^^^^^^^^

TensorNodes allow parts of a model to be defined using TensorFlow and smoothly
integrated with the rest of a Nengo model.

.. autoclass:: nengo_dl.tensor_node.TensorNode

.. autofunction:: nengo_dl.tensor_node.tensor_layer

.. _config-api:

Configuration system
^^^^^^^^^^^^^^^^^^^^

The configuration system is used to change NengoDL's default behaviour in
various ways.

.. automodule:: nengo_dl.config

Neuron types
^^^^^^^^^^^^

Additions to the `neuron types included with Nengo
<https://www.nengo.ai/nengo/frontend_api.html#neuron-types>`_.

.. automodule:: nengo_dl.neurons

Distributions
^^^^^^^^^^^^^

Additions to the `distributions included with Nengo
<https://www.nengo.ai/nengo/frontend_api.html#distributions>`_.  These
distributions are usually used to initialize weight matrices, e.g.
``nengo.Connection(a.neurons, b.neurons, transform=nengo_dl.dists.Glorot())``.

.. automodule:: nengo_dl.dists


Developers
----------

These objects are only relevant to NengoDL developers.

Builder
^^^^^^^

The Builder is in charge of mapping (groups of) Nengo operators to
the builder objects that know how to translate those operators into a
TensorFlow graph.

.. autoclass:: nengo_dl.builder.Builder

.. autoclass:: nengo_dl.builder.OpBuilder
    :exclude-members: pass_rng

.. autoclass:: nengo_dl.builder.BuildConfig

Operator builders
^^^^^^^^^^^^^^^^^

These objects are used to convert Nengo operators into TensorFlow graph
elements.

Basic operators
***************

.. automodule:: nengo_dl.op_builders

Neuron types
************

.. automodule:: nengo_dl.neuron_builders

Learning rules
**************

.. automodule:: nengo_dl.learning_rule_builders

Processes
*********

.. automodule:: nengo_dl.process_builders

TensorNodes
***********

To build `.TensorNode` objects we need to define a new Nengo operator
(`.tensor_node.SimTensorNode`), a build function that adds that operator
into a Nengo graph (`.tensor_node.build_tensor_node`), and a NengoDL
build class that maps that new Nengo operator to TensorFlow operations
(`.tensor_node.SimTensorNodeBuilder`).

.. autoclass:: nengo_dl.tensor_node.SimTensorNode

.. autofunction:: nengo_dl.tensor_node.build_tensor_node

.. autoclass:: nengo_dl.tensor_node.SimTensorNodeBuilder

Graph construction
^^^^^^^^^^^^^^^^^^

The TensorGraph class manages all the data and build processes associated with
the TensorFlow graph.  The TensorFlow graph is the symbolic description of
the computations in the network, which will be executed by the simulator.

.. automodule:: nengo_dl.tensor_graph

Signals
^^^^^^^

.. autoclass:: nengo_dl.signals.TensorSignal
    :special-members: __getitem__

.. autoclass:: nengo_dl.signals.SignalDict

Graph optimization
^^^^^^^^^^^^^^^^^^

These functions are used to restructure the operator graph so that it can
be simulated more efficiently when converted into a TensorFlow graph.

.. automodule:: nengo_dl.graph_optimizer

Utilities
^^^^^^^^^

.. automodule:: nengo_dl.utils
