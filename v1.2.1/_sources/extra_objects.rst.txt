Extra Nengo objects
===================

NengoDL adds some new Nengo objects that can be used during model construction.
These could be used with any Simulator, not just ``nengo_dl``, but they tend to
be useful for deep learning applications.

Neuron types
------------

Additions to the `neuron types included with Nengo
<https://www.nengo.ai/nengo/frontend_api.html#neuron-types>`_.

.. automodule:: nengo_dl.neurons

Distributions
-------------

Additions to the `distributions included with Nengo
<https://www.nengo.ai/nengo/frontend_api.html#distributions>`_.  These
distributions are usually used to initialize weight matrices, e.g.
``nengo.Connection(a.neurons, b.neurons, transform=nengo_dl.dists.Glorot())``.

.. automodule:: nengo_dl.dists
