Simulator
=========

This is the class that allows users to access the ``nengo_deeplearning``
backend.  This can be used as a drop-in replacement for ``nengo.Simulator``
(i.e., simply replace any instance of ``nengo.Simulator`` with
``nengo_deeplearning.Simulator`` and everything will continue to function as
normal).

In addition, the Simulator exposes features unique to the
``nengo_deeplearning`` backend, such as :meth:`.Simulator.train`.

.. autoclass:: nengo_deeplearning.simulator.Simulator
    :private-members:
    :exclude-members: unsupported, _generate_inputs, _update_probe_data, dt