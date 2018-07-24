Optimizing a NengoDL model
==========================

Optimizing Nengo models via deep learning training methods is one of the
important features of NengoDL.  This functionality is accessed via the
:meth:`.Simulator.train` method.  For example:

.. code-block:: python

    with nengo.Network() as net:
        <construct the model>

    with nengo_dl.Simulator(net, ...) as sim:
        sim.train(<inputs>, <targets>, <optimizer>, n_epochs=10,
                  objective=<objective>)

When the ``Simulator`` is first constructed, all the parameters in the model
(e.g., encoders, decoders, connection weights, biases) are initialized based
on the functions/distributions specified during model construction (see the
`Nengo documentation <https://www.nengo.ai/nengo/>`_ for more detail on
how that works).  What the :meth:`.Simulator.train` method does is then
further optimize those parameters based on some inputs and desired
outputs.  We'll go through each of those components in more detail
below.

Simulator.train arguments
-------------------------

inputs
^^^^^^

The first argument to the :meth:`.Simulator.train` function is the input data.
We can think of a model as computing a function
:math:`y = f(x, \theta)`, where :math:`f` is the model, mapping inputs
:math:`x` to outputs :math:`y` with parameters :math:`\theta`.  This
argument is specifying the values for :math:`x`.

In practice what that means is specifying values for the input Nodes in the
model.  A :class:`~nengo:nengo.Node` is a Nengo object that inserts values into
a Network, usually used
to define external inputs.  :meth:`.Simulator.train` will override the normal
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
        sim.train(inputs={a: np.random.randn(n_inputs, n_steps, 1),
                          b: np.random.randn(n_inputs, n_steps, 3)},
                  ...)

Input values must be provided for at least one Node, but beyond that can be
defined for as many Nodes as desired.  Any Nodes that don't have data provided
will take on the values specified during model construction.  Also note that
inputs can only be defined for Nodes with no incoming connections (i.e., Nodes
with ``size_in == 0``).

targets
^^^^^^^

Returning to the network equation :math:`y = f(x, \theta)`, the goal in
optimization is to find a set of parameter values such that given inputs
:math:`x` the actual network outputs :math:`y` are as close as possible to
some target values :math:`t`.  This argument is specifying those
desired outputs :math:`t`.

This works very similarly to defining inputs, except instead of assigning
input values to Nodes it assigns target values to Probes.  The structure of the
argument is similar -- a dictionary of ``{<probe>: <array>, ...}``, where
``<array>`` has shape ``(n_inputs, n_steps, probe.size_in)``.  Each entry
in the target array defines the desired output for the corresponding entry in
the input array.

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
        sim.train(targets={p: np.random.randn(n_inputs, n_steps, 2)},
                  ...)

Note that these examples use random inputs/targets, for the sake of simplicity.
In practice we would do something like ``targets={p: my_func(inputs)}``, where
``my_func`` is a function specifying what the ideal outputs are for the given
inputs.

optimizer
^^^^^^^^^

The optimizer is the algorithm that defines how to update the
network parameters during training.  Any of the optimization methods
implemented in TensorFlow can be used in NengoDL; more information can be found
in the `TensorFlow documentation
<https://www.tensorflow.org/api_guides/python/train#Optimizers>`_.

An instance of the desired TensorFlow optimizer is created (specifying any
arguments required by that optimizer), and that instance is then passed to
:meth:`.Simulator.train`.  For example:

.. code-block:: python

    import tensorflow as tf

    with nengo_dl.Simulator(net, ...) as sim:
        sim.train(optimizer=tf.train.MomentumOptimizer(
            learning_rate=1e-2, momentum=0.9, use_nesterov=True), ...)

objective
^^^^^^^^^

The goal in optimization is to minimize the error between the network's actual
outputs :math:`y` and the targets :math:`t`.  The objective is the
function :math:`e = o(y, t)` that computes an error value :math:`e`, given
:math:`y` and :math:`t`.

The default objective in NengoDL is the standard `mean squared error
<https://en.wikipedia.org/wiki/Mean_squared_error>`_.  This will be used if
the user doesn't specify an objective.

Users can specify a custom objective by creating a function and passing that
to the ``objective`` argument in :meth:`.Simulator.train`.  Note that the
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
        sim.train(objective=my_objective, ...)


Finally, it is also possible to pass ``None`` as the objective.  This indicates
that the error is being computed outside the simulation by the modeller.  In
this case the modeller should directly specify the output error gradient as the
``targets`` value.  For example, we could apply the same mean squared error
update this way:

.. code-block:: python

    with nengo_dl.Simulator(net, ...) as sim:
        sim.run(...)
        error = 2 * (sim.data[p] - my_targets)
        sim.train(targets=error, objective=None, ...)


If there are multiple output Probes defined in ``targets`` then by default the
same objective will be used for all probes.  This can be overridden by passing
a dictionary with the form
``{my_probe0: my_objective0, my_probe1: my_objective1, ...}`` for the
``objective``, specifying a different
objective for each probe. In either case, the error will then be summed
across the probes to produce an overall error value.

Note that :meth:`.Simulator.loss` can be used to check the loss
(error) value for a given objective.

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
<https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard>`_ to
help visualize this kind of data, and the ``summaries`` argument can be used to
specify the model data that you would like to export for TensorBoard.

It is specified as a list of objects for which we want to collect
data.  The data collected depends on the object: if it is a
:class:`~nengo:nengo.Connection` then data will be collected about the
distribution of the connection weights over the course of training; passing an
:class:`~nengo:nengo.Ensemble` will collect data about the distribution of
encoders, and :class:`~nengo:nengo.ensemble.Neurons` will collect data about
the distribution of biases. Additionally, the string ``"loss"`` can be passed,
in which case the training error for the given objective will be
collected over the course of training.

Alternatively, you can manually create summaries using ``tf.summary.*`` ops for
any Tensors you would like to track (see `the TensorFlow documentation
<https://www.tensorflow.org/api_guides/python/summary>`_), and include those
in the summaries list.

TensorBoard can be used to view the exported data via the command

.. code-block:: bash

    tensorboard --logdir <tensorboard_dir>

where ``tensorboard_dir`` is the value specified on Simulator creation via
``nengo_dl.Simulator(..., tensorboard=tensorboard_dir)``.  After TensorBoard is
running you can view the data by opening a web browser and navigating to
http://localhost:6006.

For details on the usage of TensorBoard, consult the `TensorFlow documentation
<https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard>`__.
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
- ``profile`` (bool or dict): collect profiling information
  (:ref:`as in Simulator.run <sim-profile>`)

Choosing which elements to optimize
-----------------------------------

By default, NengoDL will optimize the following elements in a model:

1. Connection weights (neuron--neuron weight matrices or decoders)
2. Ensemble encoders
3. Neuron biases

These elements will *not* be optimized if they are targeted by an online
learning rule.  For example, :class:`nengo:nengo.PES` modifies connection
weights as a model is running.  If we also tried to optimize those weights with
some offline training method then those two processes would conflict
with each other, likely resulting in unintended effects.  So NengoDL will
assume that those elements should not be optimized.

Any of these default behaviours can be overridden using `Nengo's config system
<https://www.nengo.ai/nengo/config.html>`_.  Specifically, setting the
``trainable`` config attribute for an object will control whether or not it
will be optimized.

:func:`.configure_settings` is a utility function that can be used to add a
configurable ``trainable`` attribute to the objects in a network.  Setting
``trainable=None`` will use the defaults described above, or True/False can
be passed to override the default for all objects in a model.

For example, suppose we only want to optimize one connection in our network,
while leaving everything else unchanged.  This could be achieved via

.. code-block:: python

    with nengo.Network() as net:
        # this adds the `trainable` attribute to all the trainable objects
        # in the network, and initializes it to `False`
        nengo_dl.configure_settings(trainable=False)

        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(size_in=1)

        nengo.Connection(a, b)

        # make this specific connection trainable
        conn = nengo.Connection(b, c)
        net.config[conn].trainable = True

Or if we wanted to disable training for some subnetwork:

.. code-block:: python

    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=None)
        ...
        with nengo.Network() as subnet:
            net.config[subnet].trainable = False
            ...

Note that ``config[nengo.Ensemble].trainable`` controls both encoders and
biases, as both are properties of an Ensemble.  However, it is possible to
separately control the biases via ``config[nengo.ensemble.Neurons].trainable``
or ``config[my_ensemble.neurons].trainable``.

There are two important caveats to keep in mind when configuring ``trainable``,
which differ from the standard config behaviour:

1. ``trainable`` applies to all objects in a network, regardless of whether
   they were created before or after ``trainable`` is set.  For example,

   .. code-block:: python

       with nengo.Network() as net:
           ...
           net.config[nengo.Ensemble].trainable = False
           a = nengo.Ensemble(10, 1)
           ...

   is the same as

   .. code-block:: python

       with nengo.Network() as net:
           ...
           a = nengo.Ensemble(10, 1)
           net.config[nengo.Ensemble].trainable = False
           ...


2. ``trainable`` can only be set on the config of the top-level network.  For
   example,

   .. code-block:: python

       with nengo.Network() as net:
           nengo_dl.configure_settings(trainable=None)

           with nengo.Network() as subnet:
               my_ens = nengo.Ensemble(...)

               # incorrect
               subnet.config[my_ens].trainable = False

               # correct
               net.config[my_ens].trainable = False


Saving and loading parameters
-----------------------------

After optimizing a model we often want to do something with the trained
parameters (e.g., inspect their values, save them to file, reuse them in a
different model).  NengoDL provides a number of methods to access model
parameters, in order to support different use cases.

sim.data
^^^^^^^^

The most basic way to access model parameters is through the
:class:`sim.data <.simulator.SimulationData>`
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

On the opposite end of the spectrum, :meth:`~.Simulator.save_params`/
:meth:`~.Simulator.load_params` can be used to save all the parameters of a
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
see the :doc:`spiking MNIST example <examples/spiking_mnist>`).

This method is quick and convenient, but not as flexible as other options.

sim.freeze_params
^^^^^^^^^^^^^^^^^

Rather than saving model parameters using TensorFlow's checkpoint system,
we can store live parameters back into the model definition using
:meth:`~.Simulator.freeze_params`.  We can freeze the parameters of individual
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

* :doc:`examples/nef_init`
* :doc:`examples/spiking_mnist`
