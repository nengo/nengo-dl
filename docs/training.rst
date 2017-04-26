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

Inputs
------

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

When training a NengoDL model there are two :class:`.Simulator` parameters
that must be provided.  The first is ``minibatch_size``, which defines how
many inputs (out of the total ``n_inputs`` defined above) will be used for each
optimization step.  The second is ``step_blocks``, which tells the simulator
the value of ``n_steps`` above, so that the simulation graph is configured
to run the appropriate number of simulation steps.

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

    with nengo_dl.Simulator(net, step_blocks=n_steps, minibatch_size=minibatch_size) as sim:
        sim.train(inputs={a: np.random.randn(n_inputs, n_steps, 1),
                          b: np.random.randn(n_inputs, n_steps, 3)},
                  ...)

Input values must be provided for at least one Node, but beyond that can be
defined for as many Nodes as desired.  Any Nodes that don't have data provided
will take on the values specified during model construction.  Also note that
inputs can only be defined for Nodes with no incoming connections (i.e., Nodes
with ``size_in == 0``).

Targets
-------

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

    with nengo_dl.Simulator(
            net, step_blocks=n_steps, minibatch_size=minibatch_size) as sim:
        sim.train(targets={p: np.random.randn(n_inputs, n_steps, 2)},
                  ...)

Note that these examples use random inputs/targets, for the sake of simplicity.
In practice we would do something like ``targets={p: my_func(inputs)}``, where
``my_func`` is a function specifying what the ideal outputs are for the given
inputs.

Optimizer
---------

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
---------

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
----------------

- ``n_epochs`` (int): run training for this many passes through the input data
- ``shuffle`` (bool): if ``True`` (default), randomly assign data to different
  minibatches each epoch

Examples
--------

Here is a complete example showing how to train a network using NengoDL.  The
function being learned here is not particularly interesting (multiplying by 2),
but it shows how all of the above parts can fit together.

.. code-block:: python

    import nengo
    import nengo_dl
    import numpy as np
    import tensorflow as tf

    with nengo.Network(seed=0) as net:
        # these parameter settings aren't necessary, but they set things up in
        # a more standard machine learning way, for familiarity
        net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
        net.config[nengo.Connection].synapse = None

        # connect up our input node, and 3 ensembles in series
        a = nengo.Node([0.5])
        b = nengo.Ensemble(30, 1)
        c = nengo.Ensemble(30, 1)
        d = nengo.Ensemble(30, 1)
        nengo.Connection(a, b)
        nengo.Connection(b, c)
        nengo.Connection(c, d)

        # define our outputs with a probe on the last ensemble in the chain
        p = nengo.Probe(d)

    n_steps = 5  # the number of simulation steps we want to run our model for
    mini_size = 100  # minibatch size

    with nengo_dl.Simulator(net, step_blocks=n_steps, minibatch_size=mini_size,
                            device="/cpu:0") as sim:
        # create input/target data. this could be whatever we want, but here
        # we'll train the network to output 2x its input
        input_data = np.random.uniform(-1, 1, size=(10000, n_steps, 1))
        target_data = input_data * 2

        # train the model, passing `input_data` to our input node `a` and
        # `target_data` to our output probe `p`. we can use whatever TensorFlow
        # optimizer we want here.
        sim.train({a: input_data}, {p: target_data},
                  tf.train.MomentumOptimizer(5e-2, 0.9), n_epochs=30)

        # run the model to see the results of the training. note that this will
        # use the input values specified in our `nengo.Node` definition
        # above (0.5)
        sim.run_steps(n_steps)

        # so the output should be 1
        assert np.allclose(sim.data[p], 1, atol=1e-2)

        sim.soft_reset(include_probes=True)

        # or if we wanted to see the performance on a test dataset, we could do
        test_data = np.random.uniform(-1, 1, size=(mini_size, n_steps, 1))
        sim.run_steps(n_steps, input_feeds={a: test_data})

        assert np.allclose(test_data * 2, sim.data[p], atol=1e-2)

Limitations
-----------

- Almost all deep learning methods require the network to be differentiable,
  which means that trying to train a network with non-differentiable elements
  will result in an error.  Examples of common non-differentiable
  elements include :class:`nengo:nengo.LIF`,
  :class:`nengo:nengo.Direct`, or processes/neurons that don't have a
  custom TensorFlow implementation (see
  :class:`.processes.SimProcessBuilder`/
  :class:`.neurons.SimNeuronsBuilder`)

- Most TensorFlow optimizers do not have GPU support for networks with
  sparse reads, which are a common element in Nengo models.  If your
  network contains sparse reads then training will have to be
  executed on the CPU (by creating the simulator via
  ``nengo_dl.Simulator(..., device="/cpu:0")``), or is limited to
  optimizers with GPU support (currently this is only
  ``tf.train.GradientDescentOptimizer``). Follow `this issue
  <https://github.com/tensorflow/tensorflow/issues/2314>`_ for updates
  on Tensorflow GPU support.