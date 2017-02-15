Builder
=================
The Builder is in charge of mapping (groups of) Nengo operators to
the builder objects that know how to translate those operators into a
Tensorflow graph.

.. autoclass:: nengo_deeplearning.builder.Builder

.. autoclass:: nengo_deeplearning.builder.OpBuilder
    :exclude-members: pass_rng