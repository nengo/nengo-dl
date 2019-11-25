Configuration options
=====================

NengoDL uses Nengo's `config system <https://www.nengo.ai/nengo/config.html>`__
to allow users to control more fine-grained aspects of the simulation.  In
general, most users will not need to worry about these options, and can leave
them at their default settings.  However, these options may be useful in
some scenarios.

`.configure_settings` is a utility function that can be used to set these
configuration options.  It needs to be called within a Network context, as in:

.. code-block:: python

    with nengo.Network() as net:
        nengo_dl.configure_settings(config_option=config_value, ...)
        ...

Each call to ``configure_settings`` only sets the configuration
options specified in that call.  That is,

.. code-block:: python

    nengo_dl.configure_settings(option0=val0)
    nengo_dl.configure_settings(option1=val1)

is equivalent to

.. code-block:: python

    nengo_dl.configure_settings(option0=val0, option1=val1)

Under the hood, ``configure_settings`` is setting ``Network.config`` attributes on
the top-level network.  All of the same effects could be achieved by setting
those ``Network.config`` attributes directly, ``configure_settings`` simply makes this
easier.

Note that TensorFlow also has its own `config system
<https://www.tensorflow.org/api_docs/python/tf/config>`__, which can be
used as normal with NengoDL to control the underlying TensorFlow behaviour.

.. _config-trainable:

trainable
---------

The ``trainable`` config attribute can be used to control which parts of a
model will be optimized by the `.Simulator.fit` process.
``configure_settings(trainable=None)`` will add a configurable ``trainable``
attribute to the objects in a network.  Setting ``trainable=None`` will use the
default trainability settings, or ``trainable=True/False`` can be used to
override the default for all objects.

Once the ``trainable`` attribute has been added to all the objects in a model,
the ``Network.config`` system can then be used to control the trainability of
individual objects.

For example, suppose we only want to optimize one connection in our network,
while leaving everything else unchanged.  This could be achieved via

.. testcode::

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

.. testcode::

    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=None)
        ...
        with nengo.Network() as subnet:
            nengo_dl.configure_settings(trainable=False)
            ...

Note that ``config[nengo.Ensemble].trainable`` controls both encoders and
biases, as both are properties of an Ensemble.  However, it is possible to
separately control the biases via ``config[nengo.ensemble.Neurons].trainable``
or ``config[my_ensemble.neurons].trainable``.

There is one important caveat to keep in mind when configuring ``trainable``,
which differ from the standard config behaviour. ``trainable`` applies to all objects
in a network, regardless of whether
they were created before or after ``trainable`` is set.  For example,

.. testcode::

    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=None)
        ...
        net.config[nengo.Ensemble].trainable = False
        a = nengo.Ensemble(10, 1)
        ...

is the same as

.. testcode::

    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=None)
        ...
        a = nengo.Ensemble(10, 1)
        net.config[nengo.Ensemble].trainable = False
        ...

Trainability settings are prioritized according to the following rules:

1. Settings in lower subnetworks take priority.

    .. testcode::

        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=True)
            with nengo.Network() as subnet:
                nengo_dl.configure_settings(trainable=False)

                # this ensemble will not be trainable, because settings on
                # `subnet` override settings on `net`
                a = nengo.Ensemble(10, 1)

2. Settings on instances take priority over classes.

    .. testcode::

        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=None)

            a = nengo.Ensemble(10, 1)

            # this will make `a` trainable (even though Ensembles in general
            # are not trainable)
            net.config[a].trainable = True
            net.config[nengo.Ensemble].trainable = False

.. _config-planner:

planner
-------

This option can be used to change the algorithm used for assigning an order
to simulation operations during the graph optimization stage.  For example, we
could disable operator merging by using the ``noop_planner``.

.. testcode::

    from nengo_dl.graph_optimizer import noop_planner

    with nengo.Network() as net:
        nengo_dl.configure_settings(planner=noop_planner)

sorter
------

This option can be used to change the algorithm used for sorting
signals/operators during the graph optimization stage.  For example, we could
disable sorting via

.. testcode::

    from nengo_dl.graph_optimizer import noop_order_signals

    with nengo.Network() as net:
        nengo_dl.configure_settings(sorter=noop_order_signals)

simplifications
---------------

This option can be used to change the simplification transformations applied
during the graph optimization stage.  This takes a list of transformation
functions, where each will be applied in sequence.  For example, we could apply
only two of the default simplifications via

.. testcode::

    from nengo_dl.graph_optimizer import remove_identity_muls, remove_zero_incs

    with nengo.Network() as net:
        nengo_dl.configure_settings(simplifications=[remove_identity_muls,
                                                     remove_zero_incs])

.. _config-inference-only:

inference_only
--------------

By default, NengoDL models are built to support both training and inference.
However, sometimes we may know that we'll only be using a simulation for
inference (for example, if we want to take advantage of the batching/GPU
acceleration of NengoDL, but don't need the ``sim.fit`` functionality).  In
that case we can improve the simulation speed of the model by omitting some
of the aspects related to training.  Setting
``nengo_dl.configure_settings(inference_only=True)`` will cause the network
to be built in inference-only mode.

lif_smoothing
-------------

During training, NengoDL automatically replaces the non-differentiable
spiking `~nengo.LIF` neuron model with the differentiable
`~nengo.LIFRate` approximation.
However, although ``LIFRate`` is generally differentiable, it has a sharp
discontinuity at the firing threshold.  In some cases this can lead to
difficulties during the training process, and performance can be improved by
smoothing the ``LIFRate`` response around the firing threshold.  This is
known as the `~.neurons.SoftLIFRate` neuron model.

``SoftLIFRate`` has a parameter ``sigma`` that controls the degree of smoothing
(``SoftLIFRate`` approaches ``LIFRate`` as ``sigma`` goes to zero).  Setting
``nengo_dl.configure_settings(lif_smoothing=x)`` will cause the ``LIF``
gradients to be approximated by ``SoftLIFRate`` instead of ``LIFRate``, with
``sigma=x``.

dtype
-----

This specifies the floating point precision to be used for the simulator's
internal computations.  It can be either ``"float32"`` or ``"float64"``,
for 32 or 64-bit precision, respectively.  32-bit precision is the default,
as it is faster, will use less memory, and in most cases will not make a
difference in the results of the simulation.  However, if very precise outputs
are required then this can be changed to ``"float64"``.

keep_history
------------

By default, a `nengo.Probe` stores the probed output from every simulation
timestep.  However, sometimes in NengoDL we want to add a probe to something
for other reasons, and don't necessarily care about all of that data (which can
consume a lot of memory).  For example, we might want to apply a probe to some
connection weights so that we can apply a regularization penalty, but since
the weights aren't changing during a simulation run we don't need to keep
the value from every simulation step.

The ``keep_history`` config option allows Probes to be configured so that they
only store the output of the probed signal from the last simulation timestep.
Calling

.. testcode::

    with nengo.Network() as net:
        nengo_dl.configure_settings(keep_history=False)

will set the default value for all probes in the simulation, which can then
be further configured on a per-probe basis, e.g.

.. testcode::

   with nengo.Network() as net:
      nengo_dl.configure_settings(keep_history=True)

      my_ens = nengo.Ensemble(10, 1)
      my_probe = nengo.Probe(my_ens)
      net.config[my_probe].keep_history = False

.. _config-stateful:

stateful
--------

By default, a NengoDL simulator is built to be stateful (meaning that internal
simulation state can be preserved between runs). However, if you know that you will
not need this functionality (i.e. you want all Simulator executions to begin from
the default initial conditions) it can be disabled by setting
``nengo_dl.configure_settings(stateful=False)``. This may slightly improve the
simulation speed.

Note that in any case the internal state of the Simulation will be
tracked within a given call (e.g. within one call to `.Simulator.run`). This only
affects whether state is preserved between calls.

.. _config-use-loop:

use_loop
--------

By default, NengoDL models run inside a loop within TensorFlow; this is what
allows us to flexibly simulate a model for any number of timesteps. However, in some
cases we may not need this functionality (for example, if we have a simple feedforward
network that will only ever be simulated for a single timestep). In that case we can
set ``nengo_dl.configure_settings(use_loop=False)`` to build the model without the
outer loop, which can improve the simulation speed.

Note that it is still possible to have a model that simulates multiple timesteps by
setting ``nengo_dl.Simulator(..., unroll_simulation=x)``. This will explicitly build
``x`` timesteps into the model (without using a loop).  So if we use
``unroll_simulation=x`` and ``use_loop=False``, then the simulation will always run
for exactly ``x`` timesteps.
