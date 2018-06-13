Builder
=================
The Builder is in charge of mapping (groups of) Nengo operators to
the builder objects that know how to translate those operators into a
TensorFlow graph.

.. autoclass:: nengo_dl.builder.Builder

.. autoclass:: nengo_dl.builder.OpBuilder
    :exclude-members: pass_rng