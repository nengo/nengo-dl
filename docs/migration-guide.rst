Migration guide
===============

The goal of this section is to help with upgrading code that was written for an older
version of NengoDL.  Note that this is not a list of all
the new features available in newer versions. Here we will only look at features that
were present in older versions, and how they have changed.

See :doc:`the release history <release-history>` for complete details on what has
changed in each version.

NengoDL 2 to 3
--------------

NengoDL 3 makes some significant changes to the API, mainly motivated by the
release of TensorFlow 2.0.  TensorFlow 2.0 adopts the Keras API as the standard
high-level interface, so in NengoDL 3 we make the same change, modifying the API
to better integrate with Keras.

Simulator changes
^^^^^^^^^^^^^^^^^

1.  **Use** `.Simulator.fit` **instead of** ``Simulator.train``. `.Simulator.fit` is the
    new access point for optimizing a NengoDL model.  It is closely based on the Keras
    `Model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
    function. One important difference between ``fit`` and ``train`` is that the
    optimizer and loss functions are specified in a separate `.Simulator.compile` step
    (analogous to Keras' `Model.compile
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_)
    rather than directly in ``Simulator.train``. Another difference is that input
    and target data are specified in separate ``x`` and ``y`` arguments, rather than
    a single ``data`` argument.

    Also note that you should use the updated TensorFlow 2.0 optimizers/loss functions
    in ``tf.optimizers`` and ``tf.losses``, rather than the deprecated optimizers
    in ``tf.compat.v1.train``.

    .. testcode::

        with nengo.Network() as example_net:
            node = nengo.Node([0])
            ens = nengo.Ensemble(10, 1)
            nengo.Connection(node, ens, synapse=None)
            probe = nengo.Probe(ens)

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net) as sim:
            sim.train(
                data={node: np.zeros((1, 1, 1)), probe: np.zeros((1, 1, 1))},
                optimizer=tf.train.AdamOptimizer(0.01),
                objective=nengo_dl.objectives.mse,
                n_epochs=10,
            )

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            sim.compile(optimizer=tf.optimizers.Adam(0.01), loss=tf.losses.mse)
            sim.fit(
                x={node: np.zeros((1, 1, 1))},
                y={probe: np.zeros((1, 1, 1))},
                epochs=10
            )

    .. testoutput::
        :hide:

        ...

2.  **Use** `.Simulator.evaluate` **instead of** ``Simulator.loss``.  In the same way as
    ``train`` is replaced by ``fit``, ``loss`` is replaced by ``evaluate``, which is
    equivalent to the Keras
    `Model.evaluate <https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate>`_
    function.  It differs from ``loss`` in all the same ways (separate ``compile``
    step and independently specified inputs and targets).

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net) as sim:
            sim.loss(
                data={node: np.zeros((1, 1, 1)), probe: np.zeros((1, 1, 1))},
                objective=nengo_dl.objectives.mse,
            )

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            sim.compile(loss=tf.losses.mse)
            sim.evaluate(
                x={node: np.zeros((1, 1, 1))}, y={probe: np.zeros((1, 1, 1))})

    .. testoutput::
        :hide:

        ...

3.  **Extra simulator steps will no longer be hidden**.  When simulating a number of
    timesteps that is not evenly divisible by ``Simulator.unroll_simulation``,
    extra simulation steps will be executed (this is true in both 2 and 3).
    In NengoDL 2 these extra steps and
    any data associated with them were hidden from the user. In NengoDL 3 the
    number of steps executed is unchanged, but the simulation is now updated to
    reflect the number of steps that were actually executed (rather than the number
    the user requested).

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net, unroll_simulation=5) as sim:
            sim.run_steps(18)
            assert sim.n_steps == 18
            assert len(sim.data[probe]) == 18

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net, unroll_simulation=5) as sim:
            sim.run_steps(18)
            assert sim.n_steps == 20
            assert len(sim.data[probe]) == 20

4.  `.Simulator.save_params` **and** `.Simulator.load_params`
    **arguments** ``include_global`` **and** ``include_local``
    **replaced with** ``include_non_trainable``.  TensorFlow 2.0 removed the division of
    Variables into "global" and "local" collections.  Instead, Keras organizes Variables
    according to whether they are trainable or not.  Generally speaking, in NengoDL 2
    global variables were trainable and local variables were not, so the two
    organization schemes are roughly equivalent. However, it is possible for users to
    manually create non-trainable global variables or trainable local variables, in
    which case these two organization schemes would not be equivalent.

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net) as sim:
            sim.save_params("trainable", include_global=True, include_local=False)
            sim.save_params("non_trainable", include_global=False, include_local=True)
            sim.save_params("both", include_global=True, include_local=True)

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            sim.save_params("trainable", include_non_trainable=False)
            sim.save_params("both", include_non_trainable=True)

    Note that with the simplified single argument it is no longer possible to save
    only the non-trainable parameters. However, it is still possible to save these
    parameters manually if it is critical that trainable parameters not be included.

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            np.savez_compressed(
                "non_trainable",
                *tf.keras.backend.batch_get_value(sim.keras_model.non_trainable_weights)
            )

5.  **Rate/spiking neuron swapping is controlled by Keras** ``learning_phase``.  In
    Nengo DL 2 and 3 spiking neuron models are automatically swapped for rate mode
    equivalents during training.  However, sometimes it is useful to manually enable
    this swapping in other functions (for example, in order to evaluate the loss
    function on test data but with the swapped rate neuron models).  There were a couple
    ways to do this in NengoDL 2; in NengoDL 3 it is all controlled through the Keras
    ``learning_phase``.

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net) as sim:
            sim.loss(
                data={node: np.zeros((1, 1, 1)), probe: np.zeros((1, 1, 1))},
                objective=nengo_dl.objectives.mse,
                training=True,
            )

            sim.run(1.0, extra_feeds={sim.tensor_graph.signals.training: True})

    NengoDL 3:

    .. testcode::

        with tf.keras.backend.learning_phase_scope(1):
            with nengo_dl.Simulator(example_net) as sim:
                sim.compile(loss=tf.losses.mse)
                sim.evaluate(
                    x={node: np.zeros((1, 1, 1))}, y={probe: np.zeros((1, 1, 1))})

                sim.run(1.0)

    .. testoutput::
        :hide:

        ...

6.  **TensorBoard functionality replaced by Keras TensorBoard callback**.  NengoDL
    allows data about training metrics or model parameters to be output and displayed in
    TensorBoard.  In TensorFlow 2.0 the `recommended way of doing this
    <https://www.tensorflow.org/tensorboard/get_started>`_ is through Keras callbacks,
    and NengoDL 3 adopts the same API.

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net, tensorboard="results") as sim:
            sim.train(
                data={node: np.zeros((1, 1, 1)), probe: np.zeros((1, 1, 1))},
                optimizer=tf.train.AdamOptimizer(0.01),
                objective=nengo_dl.objectives.mse,
                n_epochs=10,
                summaries=["loss", ens],
            )

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            sim.compile(optimizer=tf.optimizers.Adam(0.01), loss=tf.losses.mse)
            sim.fit(
                x={node: np.zeros((1, 1, 1))},
                y={probe: np.zeros((1, 1, 1))},
                epochs=10,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir="results"),
                    nengo_dl.callbacks.NengoSummaries("results", sim, [ens]),
                ]
            )

    .. testoutput::
        :hide:

        ...

TensorNode changes
^^^^^^^^^^^^^^^^^^

1.  **Use** `nengo_dl.Layer` **instead of** ``nengo_dl.tensor_layer``.
    ``nengo_dl.tensor_layer`` was designed to mimic the ``tf.layers`` API.
    In TensorFlow 2.0 ``tf.layers`` has been deprecated in favour of
    ``tf.keras.layers``.  `nengo_dl.Layer` has the same functionality as
    ``nengo_dl.tensor_layer``, but mimics the Keras Layer API instead.

    NengoDL 2:

    .. code-block:: python

        with nengo.Network():
            layer = nengo_dl.tensor_layer(node, tf.layers.dense, units=10)

    NengoDL 3:

    .. testcode::

        with nengo.Network():
            layer = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(node)

2.  **Use custom Keras Layers instead of callable classes**.  When making more
    complicated TensorNodes we sometimes need to separate the layer logic into separate
    preparation and execution steps.  In NengoDL 2 this was done by creating a callable
    class that defined ``pre_build``, ``__call__``, and ``post_build`` functions.
    In NengoDL 3 this is done by creating a custom Keras Layer subclass instead, which
    can define ``build`` and ``call`` methods.  There is no longer a ``post_build``
    step, as this was used for TensorNodes that needed access to the TensorFlow
    Session object (which is no longer used in TensorFlow 2.0).

    NengoDL 2:

    .. code-block:: python

        class MyLayer:
            def pre_build(self, shape_in, shape_out):
                self.w = tf.Variable(tf.ones((1,)))

            def __call__(self, t):
                return t * self.weights

        with nengo.Network():
            tensor_node = nengo_dl.TensorNode(MyLayer())

    NengoDL 3:

    .. testcode::

        class MyLayer(tf.keras.layers.Layer):
            def build(self, input_shapes):
                self.w = self.add_weight(
                    shape=(1,), initializer=tf.initializers.ones(),
                )

            def call(self, inputs):
                return inputs * self.weights

        with nengo.Network():
            tensor_node = nengo_dl.TensorNode(MyLayer())

3.  **TensorNodes define multidimensional** ``shape_in``/``shape_out`` **rather than
    scalar** ``size_in``/``size_out``.  In core Nengo all inputs and outputs are
    vectors, and in NengoDL 2 this was also true for ``TensorNodes``.  However, often
    when working with ``TensorNodes`` it is useful to have multidimensional inputs
    and outputs, so in NengoDL 3 TensorNodes are defined with a full shape.  Note that
    these shapes do not include the batch dimension (which is defined when the
    ``Simulator`` is created).

    NengoDL 2:

    .. code-block:: python

        def my_func(t, x):
            assert t.shape == ()
            assert x.shape == (1, 24)
            return tf.reshape(x, (1, 2, 3, 4))

        with nengo.Network():
            tensor_node = nengo_dl.TensorNode(my_func, size_in=24, size_out=24)

    NengoDL 3:

    .. testcode::

        def my_func(t, x):
            assert t.shape == ()
            assert x.shape == (1, 2, 12)
            return tf.reshape(x, (1, 2, 3, 4))

        with nengo.Network():
            tensor_node = nengo_dl.TensorNode(
                my_func, shape_in=(2, 12), shape_out=(2, 3, 4))

4.  **Connections created by** `nengo_dl.Layer` **are non-trainable by default**.  We
    usually don't want these Connections to contain trainable weights (since any weights
    we want would be built into the TensorNode).  In NengoDL 2 they needed to be
    manually marked as non-trainable, but that is the default behaviour in NengoDL 3.

    NengoDL 2:

    .. code-block:: python

        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=None)
            layer, conn = nengo_dl.tensor_layer(
                node, tf.layers.dense, units=10, return_conn=True)
            net.config[conn].trainable = False

    NengoDL 3:

    .. testcode::

        with nengo.Network() as net:
            layer = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(node)

    The connection can still be manually marked as trainable if desired:

    .. testcode::

        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=None)
            layer, conn = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(
                node, return_conn=True)
            net.config[conn].trainable = True

nengo_dl.objectives changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.  ``nengo_dl.objectives`` **renamed to** ``nengo_dl.losses``. This is for consistency
    with ``tf.losses``/``tf.keras.losses``.

2.  **Loss functions take two arguments** ``(y_true, y_pred)`` **instead of**
    ``(outputs, targets)``.  Again this is for consistency with ``tf.losses``. Note that
    this swaps the order of the two arguments (so the ground truth now comes first).

    NengoDL 2:

    .. code-block:: python

        def my_loss(outputs, targets):
            return outputs - targets

    NengoDL 3:

    .. testcode::

        def my_loss(y_true, y_pred):
            return y_pred - y_true

3.  `nengo_dl.losses.Regularize` **accepts two arguments** (``y_true`` **and**
    ``y_pred``) **instead of just** ``outputs``. ``y_true`` is not used, but Keras
    requires all loss functions to accept two arguments regardless.

    NengoDL 2:

    .. code-block:: python

        nengo_dl.objectives.Regularize()(tf.ones((1, 2, 3)))

    NengoDL 3:

    .. testcode::

        nengo_dl.losses.Regularize()(None, tf.ones((1, 2, 3)))

4.  **Use** ``loss_weights`` **parameter in** `.Simulator.compile` **instead of**
    ``weight`` **parameter in** `nengo_dl.losses.Regularize`.

    .. testcode::

        with example_net:
            p0 = nengo.Probe(node)
            p1 = nengo.Probe(ens)

    NengoDL 2:

    .. code-block:: python

        with nengo_dl.Simulator(example_net) as sim:
            sim.train(
                data={node: np.zeros((1, 1, 1)), probe: np.zeros((1, 1, 1))},
                optimizer=tf.train.AdamOptimizer(0.01),
                objective={
                    probe: nengo_dl.objectives.mse,
                    p0: nengo_dl.objectives.Regularize(weight=0.5),
                    p1: nengo_dl.objectives.Regularize(weight=0.5),
                },
                n_epochs=10,
            )

    NengoDL 3:

    .. testcode::

        with nengo_dl.Simulator(example_net) as sim:
            sim.compile(
                optimizer=tf.optimizers.Adam(0.01),
                loss={
                    probe: tf.losses.mse,
                    p0: nengo_dl.losses.Regularize(),
                    p1: nengo_dl.losses.Regularize(),
                },
                loss_weights={probe: 1, p0: 0.5, p1: 0.5},
            )
            sim.fit(
                x={node: np.zeros((1, 1, 1))},
                y={
                    probe: np.zeros((1, 1, 1)),
                    p0: np.zeros((1, 1, 1)),
                    p1: np.zeros((1, 1, 1)),
                },
                epochs=10,
            )

    .. testoutput::
        :hide:

        ...

5.  ``nengo_dl.objectives.mse`` **renamed to** `nengo_dl.losses.nan_mse`.  This is to
    distinguish it from the standard ``tf.losses.mse``, and emphasize the special
    treatment of ``nan`` targets.

    NengoDL 2:

    .. code-block:: python

        assert nengo_dl.objectives.mse(np.zeros((2, 3)), np.ones((2, 3)) * np.nan) == 0

    NengoDL 3:

    .. testcode::

        assert nengo_dl.losses.nan_mse(np.ones((2, 3)) * np.nan, np.zeros((2, 3))) == 0


configure_settings changes
^^^^^^^^^^^^^^^^^^^^^^^^^^

1.  **Specify** ``dtype`` **as string instead of** ``tf.Dtype``.

    NengoDL 2:

    .. code-block:: python

        with nengo.Network():
            nengo_dl.configure_settings(dtype=tf.float32)

    NengoDL 3:

    .. testcode::

        with nengo.Network():
            nengo_dl.configure_settings(dtype="float32")

2.  **Configure trainability separately within subnetworks, rather than marking
    networks as trainable**.

    NengoDL 2:

    .. code-block:: python

        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=None)

            with nengo.Network() as subnet:
                ens = nengo.Ensemble(10, 1)

            net.config[subnet].trainable = False

    NengoDL 3:

    .. testcode::

        with nengo.Network() as net:
            with nengo.Network() as subnet:
                nengo_dl.configure_settings(trainable=False)

                ens = nengo.Ensemble(10, 1)

3.  **Use** ``tf.config`` **instead of** ``session_config``.  TensorFlow 2.0 uses
    functions in the ``tf.config`` namespace to control settings that used to be
    controlled through the SessionConfig object (which no longer exists).  So we no
    longer need the ``session_config`` option, and can instead just directly use those
    ``tf.config`` functions.

    NengoDL 2:

    .. code-block:: python

        with nengo.Network():
            nengo_dl.configure_settings(session_config={"allow_soft_placement": True})

    NengoDL 3:

    .. testcode::

        tf.config.set_soft_device_placement(True)
