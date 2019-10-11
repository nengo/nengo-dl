.. image:: https://img.shields.io/pypi/v/nengo-dl.svg
  :target: https://pypi.org/project/nengo-dl
  :alt: Latest PyPI version

.. image:: https://img.shields.io/travis/nengo/nengo-dl/master.svg
  :target: https://travis-ci.org/nengo/nengo-dl
  :alt: Travis-CI build status

.. image:: https://ci.appveyor.com/api/projects/status/github/nengo/nengo-dl?branch=master&svg=true
  :target: https://ci.appveyor.com/project/nengo/nengo-dl
  :alt: AppVeyor build status

.. image:: https://img.shields.io/codecov/c/github/nengo/nengo-dl/master.svg
  :target: https://codecov.io/gh/nengo/nengo-dl
  :alt: Test coverage

|

.. image:: https://www.nengo.ai/design/_images/nengo-dl-full-light.svg
  :target: https://www.nengo.ai/nengo-dl
  :alt: NengoDL
  :width: 400px

***********************************
Deep learning integration for Nengo
***********************************

NengoDL is a simulator for `Nengo <https://www.nengo.ai/nengo/>`_ models.
That means it takes a Nengo network as input, and allows the user to simulate
that network using some underlying computational framework (in this case,
`TensorFlow <https://www.tensorflow.org/>`_).

In practice, what that means is that the code for constructing a Nengo model
is exactly the same as it would be for the standard Nengo simulator.  All that
changes is that we use a different Simulator class to execute the
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
  training methods (using the Keras API)
- faster simulation speed, on both CPU and GPU
- inserting networks defined using TensorFlow (such as
  deep learning architectures) directly into a Nengo model

**Documentation**

Check out the `documentation <https://www.nengo.ai/nengo-dl/>`_ for

- `Installation instructions
  <https://www.nengo.ai/nengo-dl/installation.html>`_
- `Details on the unique features of NengoDL
  <https://www.nengo.ai/nengo-dl/user-guide.html>`_
- `Tutorial for new users with a TensorFlow background
  <https://www.nengo.ai/nengo-dl/examples/from-tensorflow.html>`_
- `Tutorial for new users with a Nengo background
  <https://www.nengo.ai/nengo-dl/examples/from-nengo.html>`_
- `More in-depth examples <https://www.nengo.ai/nengo-dl/examples.html>`_
- `API reference <https://www.nengo.ai/nengo-dl/reference.html>`_
