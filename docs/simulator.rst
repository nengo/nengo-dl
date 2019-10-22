NengoDL Simulator
=================

This is the class that allows users to access the ``nengo_dl``
backend.  This can be used as a drop-in replacement for ``nengo.Simulator``
(i.e., simply replace any instance of ``nengo.Simulator`` with
``nengo_dl.Simulator`` and everything will continue to function as
normal).

In addition, the Simulator exposes features unique to the
NengoDL backend.  In many cases these features are accessed through something very
similar to the `Keras API <https://www.tensorflow.org/guide/keras/overview>`_, which
will be familiar to many deep learning practitioners.  For example,
the Simulator has `~.Simulator.predict`, `~.Simulator.fit`, `~.Simulator.compile`, and
`~.Simulator.evaluate` functions, which work in the same way as the corresponding
functions of a `Keras Model
<https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

The full class documentation can be viewed in the
:ref:`API Reference <sim-api>`.

Keras integration
-----------------

Under the hood, the NengoDL Simulator is using a Keras model to perform the simulation.
That means that most things that you would do with a Keras Model
will also work with a NengoDL Simulator.  For example, you can set up a complex
input pipeline using `tf.data <https://www.tensorflow.org/guide/data>`_, and feed those
inputs to the Nengo model through `.Simulator.predict`. Or you could follow all the
steps in `this TensorBoard tutorial
<https://www.tensorflow.org/tensorboard/get_started>`_
(simply replacing the Keras Model with a NengoDL Simulator) in order to view training
metrics from `.Simulator.fit` in TensorBoard.

In general, if you are wondering how to access some functionality in NengoDL, look up
how you would do that thing in Keras, and it probably works the same way! If you find
something that works in Keras but not NengoDL, consider
`asking a question on the forums <https://forum.nengo.ai/>`_ or
`opening a feature request <https://github.com/nengo/nengo-dl/issues>`_.

Choosing which elements to optimize
-----------------------------------

By default, NengoDL will optimize the following elements in a model:

1. Connection weights (neuron--neuron weight matrices or decoders)
2. Ensemble encoders
3. Neuron biases

These elements will *not* be optimized if they are targeted by an online
learning rule.  For example, `nengo.PES` modifies connection
weights as a model is running.  If we also tried to optimize those weights with
some offline training method then those two processes would conflict
with each other, likely resulting in unintended effects.  So NengoDL will
assume that those elements should not be optimized.

Any of these default behaviours can be overridden using the
:ref:`"trainable" config option <config-trainable>`.

Saving and loading parameters
-----------------------------

After optimizing a model we often want to do something with the trained
parameters (e.g., inspect their values, save them to file, reuse them in a
different model).  NengoDL provides a number of methods to access model
parameters, in order to support different use cases.

sim.data
^^^^^^^^

The most basic way to access model parameters is through the
`sim.data <.simulator.SimulationData>`
data structure.  This provides access to the parameters of any Nengo object,
returning them as ``numpy`` arrays.  For example:

.. testcode::

    with nengo.Network() as net:
        node = nengo.Node([0])
        ens = nengo.Ensemble(10, 1)
        conn = nengo.Connection(node, ens)
        probe = nengo.Probe(ens)

    with nengo_dl.Simulator(net) as sim:
        # < run training >

        print(sim.data[conn].weights)  # connection weights
        print(sim.data[ens].bias)  # bias values
        print(sim.data[ens].encoders)  # encoder values
        print(sim.data[ens])  # to see all the parameters for an object

.. testoutput::
    :hide:

    ...

Once we have the parameters as ``numpy`` arrays we can then do whatever
we want with them (e.g., save them to file, or use them as arguments in a
different model).  Thus this method is the most general and flexible, but also
somewhat labour intensive as the user needs to handle all of that processing
themselves for each parameter.

sim.save_params/sim.load_params
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the opposite end of the spectrum, `~.Simulator.save_params`/
`~.Simulator.load_params` can be used to save all the parameters of a
model to file.  This is
convenient if we want to save and resume the state of a model (e.g., run some
training, do some analysis, and then run more training):

.. testcode::

    with nengo_dl.Simulator(net) as sim:
        # < run training >

        sim.save_params("./my_saved_params")

    # < do something else >

    with nengo_dl.Simulator(net) as sim2:
        sim2.load_params("./my_saved_params")
        # sim2 will now match the parameters from sim

We can also use ``save/load_params`` to reuse parameters between models, as
long as the structure of the two models match exactly (for example,
reusing parameters from a rate version of a model in a spiking version).

This method is quick and convenient, but not as flexible as other options.

sim.freeze_params
^^^^^^^^^^^^^^^^^

Rather than saving model parameters to file,
we can store live parameters back into the model definition using
`~.Simulator.freeze_params`.  We can freeze the parameters of individual
Ensembles and Connections, or pass a Network to freeze all the Ensembles and
Connections in that Network.

The main advantage of this approach is
that it makes it easy to reuse a NengoDL model in different Nengo simulators.
For example, we could optimize a model in NengoDL, save the result as a
Nengo network, and then run that model in another Simulator (e.g., one running
on custom neuromorphic hardware).

.. testcode::

    with nengo_dl.Simulator(net) as sim:
        # < run training >

        sim.freeze_params(net)

    # load our optimized network in a different simulator
    with nengo.Simulator(net) as sim2:
        # sim2 will now simulate a model in the default Nengo simulator, but
        # with the same parameters as our optimized Nengo DL model
        sim2.run(1.0)

.. testoutput::
    :hide:

    ...

Examples
--------

* :doc:`examples/from-nengo`
* :doc:`examples/from-tensorflow`
* :doc:`examples/spiking-mnist`
