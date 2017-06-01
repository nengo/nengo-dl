Code examples
=============

Various examples can be found in the ``<nengo_dl>/examples`` directory
(where ``<nengo_dl>`` is the location of the NengoDL package).  To run an
example, e.g. ``ensemble_chain.py``, execute

.. code-block:: bash

    python <nengo_dl>/examples/ensemble_chain.py

Note that some of the examples require ``matplotlib`` to be installed, in
order to view the output plots.

Training a chain of ensembles
-----------------------------

.. literalinclude:: ../examples/ensemble_chain.py

Training with NEF initialization
--------------------------------

.. literalinclude:: ../examples/nef_init.py
