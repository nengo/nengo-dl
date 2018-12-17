Optimizing a NengoDL model
==========================

Optimizing Nengo models via deep learning training methods is one of the
important features of NengoDL.  This functionality is accessed via the
`.Simulator.train` method.  For example:

.. code-block:: python

    with nengo.Network() as net:
        <construct the model>

    with nengo_dl.Simulator(net, ...) as sim:
        sim.train(<data>, <optimizer>, n_epochs=10, objective=<objective>)

When the ``Simulator`` is first constructed, all the parameters in the model
(e.g., encoders, decoders, connection weights, biases) are initialized based
on the functions/distributions specified during model construction (see the
`Nengo documentation <https://www.nengo.ai/nengo/>`_ for more detail on
how that works).  What the `.Simulator.train` method does is then
further optimize those parameters based on some inputs and desired
outputs.  We'll go through each of those components in more detail
below.

Simulator.train arguments
-------------------------

data
^^^^

The first argument to the `.Simulator.train` function is the training
data.  This generally consists of two components: input values for Nodes, and
target values for Probes.

**inputs**

We can think of a model as computing a function
:math:`y = f(x, \theta)`, where :math:`f` is the model, mapping inputs
:math:`x` to outputs :math:`y` with parameters :math:`\theta`.  These values
are specifying the values for :math:`x`.

In practice what that means is specifying values for the input Nodes in the
model.  A `~nengo.Node` is a Nengo object that inserts values into
a Network, usually used
to define external inputs.  `.Simulator.train` will override the normal
Node values with the training data that is provided.  This is specified as a
dictionary ``{<node>: <array>, ...}``, where ``<node>`` is the input node
for which training data is being defined, and ``<array>`` is a numpy array
containing the training values.  This training array should have shape
``(n_inputs, n_steps, node.size_out)``, where ``n_inputs`` is the number of
training examples, ``n_steps`` is the number of simulation steps to train
across, and ``node.size_out`` is the dimensionality of the Node.

When training a NengoDL model the user must specify the ``minibatch_size``
to use during training, via the ``Simulator(..., minibatch_size=n``) argument.
This defines how many inputs (out of the total ``n_inputs`` defined above) will
be used for each optimization step.

Here is an example illustrating how to define the input values for two
input nodes:

.. code-block:: python

    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Node([1, 2, 3])
        ...

    n_inputs = 1000
    minibatch_size = 20
    n_steps = 10

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.train(data={a: np.random.randn(n_inputs, n_steps, 1),
                        b: np.random.randn(n_inputs, n_steps, 3),
                        ...},
                  ...)

Note that inputs can only be defined for Nodes with no incoming connections
(i.e., Nodes with ``size_in == 0``).  Any Nodes that don't have data provided
will take on the values specified during model construction.

**targets**

Returning to the network equation :math:`y = f(x, \theta)`, the goal in
optimization is usually to find a set of parameter values such that given
inputs :math:`x` and target values :math:`t`, an error value
:math:`e = o(y, t)` is minimized.  These values are specifying those target
values :math:`t`.

This works very similarly to defining inputs, except instead of assigning
input values to Nodes it assigns target values to Probes.  We add
``{<probe>: <array>, ...}`` entries to the ``data`` dictionary, where
``<array>`` has shape ``(n_inputs, n_steps, probe.size_in)``.  Those target
values will be passed to the objective function :math:`g` for each timestep.

For example:

.. code-block:: python

    with nengo.Network() as net:
        ...
        ens = nengo.Ensemble(10, 2)
        p = nengo.Probe(ens)

    n_inputs = 1000
    minibatch_size = 20
    n_steps = 10

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.train(data={..., p: np.random.randn(n_inputs, n_steps, 2)},
                  ...)

Note that these examples use random inputs/targets, for the sake of simplicity.
In practice we would do something like ``targets=my_func(inputs)``, where
``my_func`` is a function specifying what the ideal outputs are for the given
inputs.

optimizer
^^^^^^^^^

The optimizer is the algorithm that defines how to update the
network parameters during training.  Any of the optimization methods
implemented in TensorFlow can be used in NengoDL; more information can be found
in the `TensorFlow documentation
<https://www.tensorflow.org/api_docs/python/tf/train>`_.

An instance of the desired TensorFlow optimizer is created (specifying any
arguments required by that optimizer), and that instance is then passed to
`.Simulator.train`.  For example:

.. code-block:: python

    import tensorflow as tf

    with nengo_dl.Simulator(net, ...) as sim:
        sim.train(optimizer=tf.train.MomentumOptimizer(
            learning_rate=1e-2, momentum=0.9, use_nesterov=True), ...)

objective
^^^^^^^^^

As mentioned, the goal in optimization is to minimize some error value
:math:`e = o(y, t)`.  The objective is the function :math:`o` that computes an
error value :math:`e`, given :math:`y` and :math:`t`.  This argument is
specified as a dictionary mapping Probes to objective functions, indicating how
the output of that probe is mapped to an error value.

The default objective in NengoDL is the standard `mean squared error
<https://en.wikipedia.org/wiki/Mean_squared_error>`_.  This will be used if
the user doesn't specify an objective.

Users can specify a custom objective by creating a function that implements
the :math:`o` function above.  Note that the
objective is defined using TensorFlow operators.  It should accept Tensors
representing outputs and targets as input (each with shape
``(minibatch_size, n_steps, probe.size_in)``) and return a scalar Tensor
representing the error. This example manually computes mean squared error,
rather than using the default:

.. code-block:: python

    import tensorflow as tf

    def my_objective(outputs, targets):
        return tf.reduce_mean((targets - outputs) ** 2)

    with nengo_dl.Simulator(net, ...) as sim:
        sim.train(objective={p: my_objective}, ...)


Some objective functions may not require target values.  In this case the
function can be defined with one argument

.. code-block:: python

    def my_objective(outputs):
        ...


Finally, it is also possible to specify ``None`` as the objective.  This
indicates that the error is being computed outside the simulation by the
modeller.  In this case the modeller should directly specify the output error
gradient as the ``targets`` value.  For example, we could apply the same mean
squared error update this way:

.. code-block:: python

    with nengo_dl.Simulator(net, ...) as sim:
        sim.run(...)
        error = 2 * (sim.data[p] - my_targets)
        sim.train(data={..., p: error}, objective={p: None}, ...)


Note that it is possible to specify multiple objective functions like
``objective={p0: my_objective0, p1: my_objective1}``.  In this case the error
will be summed across the probe objectives to produce an overall error
value to be minimized.
It is also possible to create objective functions that depend on multiple
probe outputs, by specifying ``objective={(p0, p1): my_objective}``.  In this
case, ``my_objective`` will still be passed parameters ``outputs`` and
``targets``, but those parameters will be lists containing the output/target
values for each of the specified probes.

`.Simulator.loss` can be used to check the loss
(error) value for a given objective.

See :ref:`objective-api` for some common objective functions that are
provided with NengoDL for convenience.

.. _truncation:

truncation
^^^^^^^^^^

When optimizing a simulation over time we specify inputs and targets for all
:math:`n` steps of the simulation.  The gradients are computed by running
the simulation forward for :math:`n` steps, comparing the outputs to the
targets we specified, and then propagating the gradients backwards from
:math:`n` to 0.  This is known as `Backpropagation Through Time (BPTT)
<https://en.wikipedia.org/wiki/Backpropagation_through_time>`_.

However, in some cases we may not want to run BPTT over the full :math:`n`
steps (usually because it requires a lot of memory to store all the
intermediate values for :math:`n` steps of gradient calculation).  In this case
we choose some value :math:`m < n`, run the simulation for :math:`m` steps,
backpropagate the gradients over those :math:`m` steps, then run the simulation
for :math:`m` more steps, and so on until we have run for the total :math:`n`
steps.  This is known as Truncated BPTT.

The ``truncation`` argument is used to specify :math:`m`, i.e.
``sim.train(..., truncation=m)``.  If no value is given then full un-truncated
BPTT will be performed.

In general, truncated BPTT will result in worse performance than untruncated
BPTT.  Truncation limits the range of the temporal dynamics that the network
is able to learn.  For example, if we tried to learn a function where input
:math:`x_t` should influence the output at :math:`y_{t+m+1}` that would not
work well, because the errors from step :math:`t+m+1` never make it back to
step :math:`t`.  More generally, a truncated system has less information about
how outputs at :math:`t` will affect future performance, which will limit how
well that system can be optimized.

As mentioned, the main reason to use truncated BPTT is in order to reduce the
memory demands during training.  So if you find yourself running out of memory
while training a model, consider using the ``truncation`` argument (while
ensuring that the value of :math:`m` is still large enough to capture the
temporal dynamics in the task).

.. _summaries:

summaries
^^^^^^^^^

It is often useful to view information about how aspects of a model are
changing over the course of training.  TensorFlow has created `TensorBoard
<https://www.tensorflow.org/guide/summaries_and_tensorboard>`_ to
help visualize this kind of data, and the ``summaries`` argument can be used to
specify the model data that you would like to export for TensorBoard.

It is specified as a list of objects for which we want to collect
data.  The data collected depends on the object: if it is a
`~nengo.Connection` then data will be collected about the
distribution of the connection weights over the course of training; passing an
`~nengo.Ensemble` will collect data about the distribution of
encoders, and `~nengo.ensemble.Neurons` will collect data about
the distribution of biases. Additionally, the string ``"loss"`` can be passed,
in which case the training error for the given objective will be
collected over the course of training.

Alternatively, you can manually create summaries using ``tf.summary.*`` ops for
any Tensors you would like to track (see `the TensorFlow documentation
<https://www.tensorflow.org/api_docs/python/tf/summary>`_), and include those
in the summaries list.

TensorBoard can be used to view the exported data via the command

.. code-block:: bash

    tensorboard --logdir <tensorboard_dir>

where ``tensorboard_dir`` is the value specified on Simulator creation via
``nengo_dl.Simulator(..., tensorboard=tensorboard_dir)``.  After TensorBoard is
running you can view the data by opening a web browser and navigating to
http://localhost:6006.

For details on the usage of TensorBoard, consult the `TensorFlow documentation
<https://www.tensorflow.org/guide/summaries_and_tensorboard>`__.
However, as a brief summary, you will find plots showing the loss values over
the course of training in the ``Scalars`` tab at the top, and plots showing the
distributions of weights/encoders/biases over time in the ``Distributions`` or
``Histograms`` tabs.  If you call ``sim.train`` several times with the same
summaries, each call will result in its own set of plots, with a suffix added
to the label indicating the call number (e.g.
``label, label_1, label_2, ...``). If you run your code multiple times with
the same ``tensorboard_dir``, data will be organized according to run number;
you can turn on/off the plots for different runs using the checkboxes in the
bottom left.

Other parameters
^^^^^^^^^^^^^^^^

- ``n_epochs`` (int): run training for this many passes through the input data
- ``shuffle`` (bool): if ``True`` (default), randomly assign data to different
  minibatches each epoch
- ``profile`` (bool or str): collect profiling information
  (:ref:`as in Simulator.run <sim-profile>`)

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

.. code-block:: python

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

Once we have the parameters as ``numpy`` arrays we can then do whatever
we want with them (e.g., save them to file, or use them as arguments in a
different model).  Thus this method is the most general and flexible, but also
somewhat labour intensive as the user needs to handle all of that processing
themselves for each parameter.

sim.save_params/sim.load_params
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the opposite end of the spectrum, `~.Simulator.save_params`/
`~.Simulator.load_params` can be used to save all the parameters of a
model to file (using TensorFlow's checkpointing system).  This is
convenient if we want to save and resume the state of a model (e.g., run some
training, do some analysis, and then run more training):

.. code-block:: python

    with nengo_dl.Simulator(net) as sim:
        # < run training >

        sim.save_params("./my_saved_params")

    # < do something else >

    with nengo_dl.Simulator(net) as sim2:
        sim2.load_params("./my_saved_params")
        # sim2 will now match the final state of sim

We can also use ``save/load_params`` to reuse parameters between models, as
long as the structure of the two models match exactly (for example,
reusing parameters from a rate version of a model in a spiking version;
see the :doc:`spiking MNIST example <examples/spiking-mnist>`).

This method is quick and convenient, but not as flexible as other options.

sim.freeze_params
^^^^^^^^^^^^^^^^^

Rather than saving model parameters using TensorFlow's checkpoint system,
we can store live parameters back into the model definition using
`~.Simulator.freeze_params`.  We can freeze the parameters of individual
Ensembles and Connections, or pass a Network to freeze all the Ensembles and
Connections in that Network.

The main advantage of this approach is
that it makes it easy to reuse a NengoDL model in different Nengo simulators.
For example, we could optimize a model in NengoDL, save the result as a
Nengo network, and then run that model in another Simulator (e.g., one running
on custom neuromorphic hardware).

.. code-block:: python

    with nengo_dl.Simulator(net) as sim:
        # < run training >

        sim.freeze_params(net)

    # load our optimized network in a different simulator
    with nengo.Simulator(net) as sim2:
        # sim2 will now simulate a model in the default Nengo simulator, but
        # with the same parameters as our optimized nengo_dl model
        sim2.run(1.0)


Examples
--------

* :doc:`examples/from-nengo`
* :doc:`examples/from-tensorflow`
* :doc:`examples/spiking-mnist`
