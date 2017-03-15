NengoDL users
=============

NengoDL is a simulator for `Nengo <https://pythonhosted.org/nengo/>`_ models.
That means it takes a Nengo network as input, and allows the user to simulate
that network using some underlying computational framework (in this case,
TensorFlow).

In practice, what that means is that the code for constructing a Nengo model
is exactly the same as it would be for the standard Nengo simulator.  All that
changes is that we use a different :class:`.Simulator` class to execute the
model.

For example:

.. code-block:: python

    import nengo
    import nengo_dl
    import numpy as np

    with nengo.Network() as net:
        inp = nengo.Node(output=np.sin)
        ens = nengo.Ensemble(50, 1, neuron_type=nengo.LIF())
        nengo.Connection(inp, ens, synapse=0.1)
        p = nengo.Probe(ens)

    with nengo_dl.Simulator(net) as sim: # this is the only line that changes
        sim.run(1.0)

    print(sim.data[p])

However, NengoDL is not simply a duplicate of the Nengo simulator.  It also
adds a number of unique features, such as:

- optimizing the parameters of a model through deep learning
  training methods
- faster simulation speed, on both CPU and GPU
- inserting networks defined using TensorFlow (such as
  convolutional neural networks) directly into a Nengo model

If you are new to Nengo, you should start by reading the
`Nengo documentation <https://pythonhosted.org/nengo/>`_.

Knowledge of TensorFlow is not required to use NengoDL.  However, if you want
to start constructing your own TensorFlow networks, you can can check out the
`TensorFlow documentation <https://www.tensorflow.org/>`_

These are the only classes that a NengoDL user needs to interact with:

.. toctree::
    :maxdepth: 3

    simulator
    tensor_node

You can read the :doc:`NengoDL developer documentation <backend>` if you want
to know more about how things are working under the hood.