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
`Nengo documentation <https://pythonhosted.org/nengo/>`_ for more detail on
how that works).  What the :meth:`.Simulator.train` method does is then
further optimize those parameters based on some inputs and desired
outputs.  We'll go through each of those components in more detail
below.

Simulator.train arguments
-------------------------

Inputs
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

Targets
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

Optimizer
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

Objective
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

If there are multiple output Probes defined in ``targets`` then the error
will be computed for each output individually (using the specified objective).
Then the error will be averaged across outputs to produce an overall
error value.

Note that :meth:`.Simulator.loss` can be used to check the loss
(error) value for a given objective.

Other parameters
^^^^^^^^^^^^^^^^

- ``n_epochs`` (int): run training for this many passes through the input data
- ``shuffle`` (bool): if ``True`` (default), randomly assign data to different
  minibatches each epoch
- ``profile`` (bool or str): collect profiling information
  (`:ref:`as in Simulator.run <sim-profile>`)

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

Any of these default behaviours can be overriden using `Nengo's config system
<https://pythonhosted.org/nengo/config.html>`_.  Specifically, setting the
``trainable`` config attribute for an object will control whether or not it
will be optimized.

:func:`.configure_trainable` is a utility function that will add a configurable
``trainable`` attribute to the objects in a network.  It can also
set the initial value of ``trainable`` on all those objects at the same time,
for convenience.

For example, suppose we only want to optimize one
connection in our network, while leaving everything else unchanged.  This
could be achieved via

.. code-block:: python

    with nengo.Network() as net:
        # this adds the `trainable` attribute to all the trainable objects
        # in the network, and initializes it to `False`
        nengo_dl.configure_trainable(net, default=False)

        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(size_in=1)

        nengo.Connection(a, b)

        # make this specific connection trainable
        conn = nengo.Connection(b, c)
        net.config[conn].trainable = True

Or if we wanted to disable training on the overall network, but enable it for
Connections within some subnetwork:

.. code-block:: python

    with nengo.Network() as net:
        nengo_dl.configure_trainable(net, default=False)
        ...
        with nengo.Network() as subnet:
            nengo_dl.configure_trainable(subnet)
            subnet.config[nengo.Connection].trainable = True
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


2. ``trainable`` cannot be set on manually created
   :class:`~nengo:nengo.Config` objects, only ``net.config``.  For
   example, the following would have no effect:

   .. code-block:: python

       with nengo.Config(nengo.Ensemble) as conf:
           conf[nengo.Ensemble].trainable = False


Examples
--------
.. toctree::

   examples/nef_init
   examples/spiking_mnist
