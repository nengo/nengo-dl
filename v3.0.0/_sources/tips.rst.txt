Tips and tricks
===============

This page includes some tips and tricks to get the most out of NengoDL, or answers to
common questions we see from users.  If you have a question that isn't answered here,
come `post it on the forums <https://forum.nengo.ai/>`_ and we will do our best to help.

Speed improvements
------------------

There are a number of options within NengoDL that can customize the behaviour for a
particular model in order to improve the training or inference speed.

First is the `Simulator unroll_simulation <.Simulator>` option.  This unrolls the
simulation loop, which can improve the speed of models that are simulated over time.
The optimal value for this parameter will depend on the size of the model; there will
likely be some sweet spot in the middle that offers the best performance, with
performance decreasing if ``unroll_simulation`` is too high or too low.  In general,
this parameter just needs to be determined experimentally, by running with different
``unroll_simulation`` values and seeing which one is fastest.

If you know that you will only ever be running your model for a fixed number of
timesteps, you can build the model without the outer simulation loop by setting
``nengo_dl.configure_settings(use_loop=False)``.
See :ref:`the documentation <config-use-loop>` for more details.

If you know that you will never need to preserve simulator state between runs, you
can avoid building and running the state-saving parts of the model by setting
``nengo_dl.configure_settings(stateful=False)``.  In particular, if you see warnings
like "``Method (on_train_batch_end) is slow compared to the batch update``", then this
setting may alleviate that issue.
See :ref:`the documentation <config-stateful>` for more details.

If you know that you will never be calling `.Simulator.fit`, you can disable the
training parts of the model with ``nengo_dl.configure_settings(inference_only=True)``.
See :ref:`the documentation <config-inference-only>` for more details.

For large models with a complex architecture, modifying the graph optimization
settings may offer speed improvements. For example, you could increase the planning
search depth by setting

.. testcode::

    import functools

    from nengo_dl import graph_optimizer

    with nengo.Network():
        nengo_dl.configure_settings(planner=functools.partial(
            graph_optimizer.tree_planner, max_depth=4))
        ...

See :ref:`the documentation <config-planner>` for more details.

Training a spiking deep network
-------------------------------

Training a spiking version of a deep network comes with some important differences
compared to training the corresponding standard deep network.  In general,
we cannot expect to simply copy code that works for a standard deep network, add
spiking neurons, and have everything work the same.  The process of
successfully training a spiking network will depend on the particular task, but here
are some general tips and things to think about.

First, begin with a standard, non-spiking network, and then gradually add complexity.
For example, following the steps in `this example
<https://www.nengo.ai/nengo-dl/examples/tensorflow-models.html>`_, we can begin by
adding the source network to Nengo completely unchanged, and verify that the performance
of the network hasn't changed.  Then we can separate the model into individual layers
(and again verify that the performance hasn't changed).  Finally, we can begin changing
some of the nonlinearities to spiking neurons, and see how that impacts performance.
This process allows us to debug any issues uncovered at each stage one at a time,
rather than attempting to jump into a fully redesigned spiking network.

When debugging spiking performance issues, here are somethings to think about:

1. **Connection/Probe synapses**. Spiking neurons produce discrete, "spiky" output
   signals. Synapses smooth that output, making it look more like the continuous
   signal we would get from a non-spiking nonlinearity.  So by adding synaptic filters
   into our network architecture, we can make the spiking network more like a standard
   rate network.  However, this comes at the cost of introducing temporal filter effects
   (so, for example, we will need to run the network for multiple timesteps in order
   to allow the output to stabilize).  We may also want to train the network without
   synapses, and then add them in for evaluation/inference (see `this example
   <https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html>`__).
2. **Neuron model**. There are many kinds of spiking neurons, just as there are many
   kinds of non-spiking nonlinearities.  The default in Nengo is `~nengo.LIF` neurons,
   which have a very nonlinear response curve (in addition to being spiking). It may
   be better to start with `nengo.SpikingRectifiedLinear`, which will behave more
   similarly to the standard ``relu`` nonlinearity.
3. **Ensemble parameterization**. The default parameters in Nengo are often different
   than the typical defaults in deep learning. In particular, in deep learning
   applications it is often useful to change the Nengo defaults to use a constant
   ``max_rate`` and zero ``intercepts``.  We also typically
   set the ``amplitude`` parameter on the neurons to be equal to ``1/max_rate`` (so
   that the overall output of the neuron will be around the 0--1 range). See
   `this example <https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html>`__ where
   we use both of these techniques.  Again, however, as with any hyperparameters these
   will likely need to be adjusted depending on the application if we want to
   maximize performance.
