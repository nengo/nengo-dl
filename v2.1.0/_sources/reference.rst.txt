API reference
=============

This section details the modules, classes, and functions available in
NengoDL.  It is divided into two sections.  The first section describes the
objects relevant to :ref:`NengoDL users <user-api>`. For a more in-depth
description of how to use these objects, see the :doc:`user-guide`.
The second section describes objects that only
:ref:`NengoDL developers <developer-api>` need to worry about.


.. _user-api:

Users
-----

These objects are the main access points for the user-facing features
of NengoDL.

.. _sim-api:

Simulator
^^^^^^^^^

.. automodule:: nengo_dl.simulator
    :no-members:

    .. autoclass:: nengo_dl.simulator.Simulator
        :exclude-members: unsupported, dt

    .. autoclass:: nengo_dl.simulator.SimulationData
        :special-members:
        :exclude-members: __init__, __weakref__

.. _tensornode-api:

TensorNodes
^^^^^^^^^^^

.. automodule:: nengo_dl.tensor_node
    :no-members:

    .. autoclass:: nengo_dl.tensor_node.TensorNode

    .. autofunction:: nengo_dl.tensor_node.tensor_layer

.. _config-api:

Configuration system
^^^^^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.config

Neuron types
^^^^^^^^^^^^

.. automodule:: nengo_dl.neurons

Distributions
^^^^^^^^^^^^^

.. automodule:: nengo_dl.dists

.. _objective-api:

Objectives
^^^^^^^^^^

.. automodule:: nengo_dl.objectives

.. _developer-api:

Developers
----------

These objects are only relevant to people interested in modifying the 
implementation of NengoDL (e.g., adding a new neuron type).

Builder
^^^^^^^

.. automodule:: nengo_dl.builder

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

Transforms
**********

.. automodule:: nengo_dl.transform_builders

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

.. automodule:: nengo_dl.tensor_graph

Signals
^^^^^^^

.. automodule:: nengo_dl.signals
    :exclude-members: TensorSignal

    .. autoclass:: nengo_dl.signals.TensorSignal
        :special-members: __getitem__

Graph optimization
^^^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.graph_optimizer

Utilities
^^^^^^^^^

.. automodule:: nengo_dl.utils

Benchmarks
^^^^^^^^^^

.. automodule:: nengo_dl.benchmarks

Interface
*********

The benchmark module also includes a command-line interface for building and
running the benchmarks:

.. click:: nengo_dl.benchmarks:main
    :prog: benchmarks
    :show-nested:
