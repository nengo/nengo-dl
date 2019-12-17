Examples
========

These examples can be found in the ``<nengo-dl>/docs/examples`` directory
(where ``<nengo-dl>`` is the location of the NengoDL package).  The examples
are IPython/Jupyter notebooks; if you would like to run them yourself, refer to
the
`Jupyter documentation <https://jupyter-notebook.readthedocs.io/en/latest/>`_.

Alternatively, you can use
`Google Colab <https://colab.research.google.com/github/nengo/nengo-dl/>`_
to run the examples online.  Note that when running on Colab you will need to add the
line ``!pip install nengo-dl[docs]`` at the top of each notebook, in order to install
the necessary requirements.

We recommend starting with the two introductory tutorials.  One is designed for
Nengo users who want to learn about NengoDL, and the other for TensorFlow
users.  If you are not familiar with Nengo or TensorFlow, we would recommend
beginning with the standard
`Nengo documentation <https://www.nengo.ai/nengo>`_, and then
come back here!

.. toctree::
    :maxdepth: 1

    examples/from-nengo
    examples/from-tensorflow

These examples illustrate some different possible use cases for NengoDL:

.. toctree::
    :maxdepth: 1

    examples/tensorflow-models
    examples/spiking-mnist
    examples/lmu
    examples/spa-retrieval
    examples/spa-memory
