API reference
=============

This section details the modules, classes, and functions available in
NengoDL.  It is divided into two sections.  The first section describes the
objects relevant to :ref:`NengoDL users <user-api>`. More information on these objects
can also be found in the :doc:`user-guide`.
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

    .. autoautosummary:: nengo_dl.Simulator
        :nosignatures:

        nengo_dl.simulator.SimulationData

    .. autoclass:: nengo_dl.Simulator
        :exclude-members: unsupported, dt

    .. autoclass:: nengo_dl.simulator.SimulationData
        :special-members:
        :exclude-members: __init__, __weakref__

.. _tensornode-api:

TensorNodes
^^^^^^^^^^^

.. automodule:: nengo_dl.tensor_node
    :no-members:

    .. autosummary::
        :nosignatures:

        nengo_dl.TensorNode
        nengo_dl.Layer

    .. autoclass:: nengo_dl.TensorNode

    .. autoclass:: nengo_dl.Layer
        :special-members:
        :exclude-members: __init__, __weakref__

Converter
^^^^^^^^^

`nengo_dl.Converter` can be used to automatically convert a Keras model to a native
Nengo Network. This can be useful if, for example, you want to run a model in different
Nengo Simulator backends (which will only support the core Nengo objects).

See `the documentation <https://www.nengo.ai/nengo-dl/converter.html>`__ for more
details.

.. autosummary::
    :nosignatures:

    nengo_dl.Converter

.. autoclass:: nengo_dl.Converter

.. _config-api:

Configuration system
^^^^^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.config

    .. autoautosummary:: nengo_dl.config
        :nosignatures:

Neuron types
^^^^^^^^^^^^

.. automodule:: nengo_dl.neurons

    .. autoautosummary:: nengo_dl.neurons
        :nosignatures:

Distributions
^^^^^^^^^^^^^

.. automodule:: nengo_dl.dists

    .. autoautosummary:: nengo_dl.dists
        :nosignatures:

.. _objective-api:

Loss functions
^^^^^^^^^^^^^^

.. automodule:: nengo_dl.losses

    .. autoautosummary:: nengo_dl.losses
        :nosignatures:

Callbacks
^^^^^^^^^

.. automodule:: nengo_dl.callbacks

    .. autoautosummary:: nengo_dl.callbacks
        :nosignatures:

.. _developer-api:

Developers
----------

These objects are only relevant to people interested in modifying the 
implementation of NengoDL (e.g., adding a new neuron type).

Builder
^^^^^^^

.. automodule:: nengo_dl.builder

    .. autoautosummary:: nengo_dl.builder
        :nosignatures:

Operator builders
^^^^^^^^^^^^^^^^^

These objects are used to convert Nengo operators into TensorFlow graph
elements.

Basic operators
***************

.. automodule:: nengo_dl.op_builders

    .. autoautosummary:: nengo_dl.op_builders
        :nosignatures:

Neuron types
************

.. automodule:: nengo_dl.neuron_builders

    .. autoautosummary:: nengo_dl.neuron_builders
        :nosignatures:

Learning rules
**************

.. automodule:: nengo_dl.learning_rule_builders

    .. autoautosummary:: nengo_dl.learning_rule_builders
        :nosignatures:

Processes
*********

.. automodule:: nengo_dl.process_builders

    .. autoautosummary:: nengo_dl.process_builders
        :nosignatures:

Transforms
**********

.. automodule:: nengo_dl.transform_builders

    .. autoautosummary:: nengo_dl.transform_builders
        :nosignatures:

TensorNodes
***********

To build `.TensorNode` objects we need to define a new Nengo operator
(`.tensor_node.SimTensorNode`), a build function that adds that operator
into a Nengo graph (`.tensor_node.build_tensor_node`), and a NengoDL
build class that maps that new Nengo operator to TensorFlow operations
(`.tensor_node.SimTensorNodeBuilder`).

.. autosummary::
    :nosignatures:

    nengo_dl.tensor_node.SimTensorNode
    nengo_dl.tensor_node.build_tensor_node
    nengo_dl.tensor_node.SimTensorNodeBuilder

.. autoclass:: nengo_dl.tensor_node.SimTensorNode

.. autofunction:: nengo_dl.tensor_node.build_tensor_node

.. autoclass:: nengo_dl.tensor_node.SimTensorNodeBuilder

Graph construction
^^^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.tensor_graph

    .. autoautosummary:: nengo_dl.tensor_graph
        :nosignatures:

Signals
^^^^^^^

.. automodule:: nengo_dl.signals
    :exclude-members: TensorSignal

    .. autoautosummary:: nengo_dl.signals
        :nosignatures:

    .. autoclass:: nengo_dl.signals.TensorSignal
        :special-members: __getitem__

Graph optimization
^^^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.graph_optimizer

    .. autoautosummary:: nengo_dl.graph_optimizer
        :nosignatures:

.. _layer-converter-api:

Layer converters
^^^^^^^^^^^^^^^^

.. automodule:: nengo_dl.converter
    :exclude-members: Converter

    .. autoautosummary:: nengo_dl.converter
        :nosignatures:
        :exclude-members: Converter

Utilities
^^^^^^^^^

.. automodule:: nengo_dl.utils
    :exclude-members: MessageBar

    .. autoautosummary:: nengo_dl.utils
        :nosignatures:
        :exclude-members: MessageBar

Benchmarks
^^^^^^^^^^

.. automodule:: nengo_dl.benchmarks

    .. autoautosummary:: nengo_dl.benchmarks
        :nosignatures:

Interface
*********

The benchmark module also includes a command-line interface for building and
running the benchmarks:

.. click:: nengo_dl.benchmarks:main
    :prog: benchmarks
    :show-nested:
