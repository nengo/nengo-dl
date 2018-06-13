NengoDL Simulator
=================

This is the class that allows users to access the ``nengo_dl``
backend.  This can be used as a drop-in replacement for ``nengo.Simulator``
(i.e., simply replace any instance of ``nengo.Simulator`` with
``nengo_dl.Simulator`` and everything will continue to function as
normal).

In addition, the Simulator exposes features unique to the
``nengo_dl`` backend, such as :meth:`.Simulator.train`.

.. autoclass:: nengo_dl.simulator.Simulator
    :private-members:
    :exclude-members: unsupported, _generate_inputs, _update_probe_data, dt