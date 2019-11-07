User guide
==========

The NengoDL API can be thought of as a combination of two high level frameworks:
`Nengo <https://www.nengo.ai/nengo/>`_ and
`Keras <https://www.tensorflow.org/guide/keras>`_.  We can define a network using the
standard Nengo API, such as `nengo.Ensemble` and `nengo.Connection`. This is augmented
by `nengo_dl.Layer`/`nengo_dl.TensorNode`, which allow us to add standard TensorFlow
components, such as Keras Layers, to the Nengo network.
Then we can simulate that network using `nengo_dl.Simulator`, which supports
both the standard Nengo Simulator syntax for running a model as well as the Keras
`compile/fit/evaluate <https://www.tensorflow.org/guide/keras/train_and_evaluate>`_ API
for training.

In this documentation we try not to duplicate information from those two base
frameworks. If you are wondering how to access some functionality in Nengo or Keras,
you should begin by looking up how that thing is done in the base framework. Chances
are, it works the same way in NengoDL! Here we will
focus on the new features introduced by NengoDL on top of those APIs, along with some
practical discussion of how or when to use those features.

There are two classes that users may need to interact with in order
to access the features of NengoDL: `.Simulator` and
`.TensorNode`.  The former is the main access point for
NengoDL, allowing the user to simulate models, or optimize parameters via
`.Simulator.fit`.  `.TensorNode` is used for
inserting TensorFlow code into a Nengo model.

.. toctree::
    :maxdepth: 1

    simulator
    tensor-node
    converter
    config
    tips
