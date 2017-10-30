Release History
===============

.. Changelog entries should follow this format:

   version (release date)
   ----------------------

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

0.5.3 (unreleased)
------------------

**Changed**

- Use ``tf.data`` for training data pipeline (improves training speed).

**Fixed**

- Fixed a bug where input nodes that were only read as a view were not
  feedable.
- Updated ``tensorflow-gpu`` installation check

0.5.2 (October 11, 2017)
------------------------

**Added**

- TensorNode outputs can now define a ``post_build`` function that will be
  executed after the simulation is initialized (see the `TensorNode
  documentation for details
  <https://www.nengo.ai/nengo_dl/tensor_node.html>`_).
- Added functionality for outputting summary data during the training process
  that can be viewed in TensorBoard (see the `sim.train documentation
  <https://www.nengo.ai/nengo_dl/training.html#summaries>`__).
- Added some examples demonstrating how to use Nengo DL in a more complicated
  task using semantic pointers to encode/retrieve information
- Added ``sim.training_step`` variable which will track the current training
  iteration (can be used, e.g., for TensorFlow's variable learning rate
  operations).
- Users can manually create ``tf.summary`` ops and pass them to ``sim.train``
  summaries
- The Simulator context will now also set the default TensorFlow graph to the
  one associated with the Simulator (so any TensorFlow ops created within the
  Simulator context will automatically be added to the correct graph)
- Users can now specify a different objective for each output probe during
  training/loss calculation (see the `sim.train documentation
  <https://www.nengo.ai/nengo_dl/training.html#objective>`__).

**Changed**

- Resetting the simulator now only rebuilds the necessary components in the
  graph (as opposed to rebuilding the whole graph)
- The default ``"mse"`` loss implementation will now automatically convert
  ``np.nan`` values in the target to zero error
- If there are multiple target probes given to ``sim.train``/``sim.loss`` the
  total error will now be summed across probes (instead of averaged)

**Fixed**

- ``sim.data`` now implements the full ``collections.Mapping`` interface
- Fixed bug where signal order was non-deterministic for Networks containing
  objects with duplicate names
  (`#9 <https://github.com/nengo/nengo_dl/issues/9>`_)
- Fixed bug where non-slot optimizer variables were not initialized
  (`#11 <https://github.com/nengo/nengo_dl/issues/11>`_)
- Implemented a modified PES builder in order to avoid slicing encoders on
  non-decoded PES connections
- TensorBoard output directory will be automatically created if it doesn't
  exist

0.5.1 (August 28, 2017)
-----------------------

**Changed**

- ``sim.data[obj]`` will now return live parameter values from the simulation,
  rather than initial values from the build process.  That means that it can
  be used to get the values of object parameters after training, e.g.
  ``sim.data[my_conn].weights``.
- Increased minimum Nengo version to 2.5.0.
- Increased minimum TensorFlow version to 1.3.0.

0.5.0 (July 11, 2017)
---------------------

**Added**

- Added ``nengo_dl.tensor_layer`` to help with the construction of
  layer-style TensorNodes (see the `TensorNode documentation
  <https://www.nengo.ai/nengo_dl/tensor_node.html>`_)
- Added an example demonstrating `how to train a neural network
  that can run in spiking neurons
  <https://www.nengo.ai/nengo_dl/examples/spiking_mnist.html>`_
- Added some distributions for weight initialization to ``nengo_dl.dists``
- Added ``sim.train(..., profile=True)`` option to collect profiling
  information during training
- Added new methods to simplify the Nengo operation graph, resulting in faster
  simulation/training speed
- The default graph planner can now be modified by setting the ``planner``
  attribute on the top-level Network config
- Added TensorFlow implementation for general linear synapses
- Added ``backports.tempfile`` and ``backports.print_function`` requirement for
  Python 2.7 systems

**Changed**

- Increased minimum TensorFlow version to 1.2.0
- Improved error checking for input/target data
- Improved efficiency of stateful gradient operations, resulting in faster
  training speed
- The functionality for ``nengo_dl.configure_trainable`` has been subsumed into
  the more general ``nengo_dl.configure_settings(trainable=x)``.  This has
  resulted in some small changes to how trainability is controlled within
  subnetworks; see the `updated documentation
  <https://www.nengo.ai/nengo_dl/training.html#choosing-which-elements-to-optimize>`_
  for details.
- Calling ``Simulator.train``/``Simulator.loss`` no longer resets the internal
  state of the simulation (so they can be safely intermixed with calls to
  ``Simulator.run``)

**Deprecated**

- The old ``step_blocks``/``unroll_simulation`` syntax has been fully
  deprecated, and will result in errors if used

**Fixed**

- Fixed bug related to changing the output of a Node after the model is
  constructed (`#4 <https://github.com/nengo/nengo_dl/issues/4>`_)
- Order of variable creation is now deterministic (helps make saving/loading
  parameters more reliable)
- Configuring whether or not a model element is trainable does not affect
  whether or not that element is minibatched
- Correctly reuse variables created inside a TensorNode when
  ``unroll_simulation`` > 1
- Correctly handle probes that aren't connected to any ops
- Swapped ``fan_in``/``fan_out`` in ``dists.VarianceScaling`` to align with
  the standard definitions
- Temporary patch to fix memory leak in TensorFlow (see
  `#11273 <https://github.com/tensorflow/tensorflow/issues/11273>`_)
- Fixed bug related to nodes that had matching output functions but different
  size_out
- Fixed bug related to probes that do not contain any data yet

0.4.0 (June 8, 2017)
--------------------

**Added**

- Added ability to manually specify which parts of a model are trainable
  (see the `sim.train documentation
  <https://www.nengo.ai/nengo_dl/training.html>`_)
- Added some code examples (see the ``docs/examples`` directory, or the
  `pre-built examples in the documentation
  <https://www.nengo.ai/nengo_dl/examples.html>`_)
- Added the SoftLIFRate neuron type for training LIF networks (based on
  `this paper <https://arxiv.org/abs/1510.08829>`_)

**Changed**

- Updated TensorFuncParam to new Nengo Param syntax
- The interface for Simulator ``step_blocks``/``unroll_simulation`` has been
  changed.  Now ``unroll_simulation`` takes an integer as argument which is
  equivalent to the old ``step_blocks`` value, and ``unroll_simulation=1`` is
  equivalent to the old ``unroll_simulation=False``.  For example,
  ``Simulator(..., unroll_simulation=True, step_blocks=10)`` is now equivalent
  to ``Simulator(..., unroll_simulation=10)``.
- Simulator.train/Simulator.loss no longer require ``step_blocks`` (or the new
  ``unroll_simulation``) to be specified; the number of steps to train across
  will now be inferred from the input data.


0.3.1 (May 12, 2017)
--------------------

**Added**

- Added more documentation on Simulator arguments

**Changed**

- Improved efficiency of tree_planner, made it the new default planner

**Fixed**

- Correctly handle input feeds when n_steps > step_blocks
- Detect cycles in transitive planner
- Fix bug in uneven step_blocks rounding
- Fix bug in Simulator.print_params
- Fix bug related to merging of learning rule with different dimensionality
- Use tf.Session instead of tf.InteractiveSession, to avoid strange side
  effects if the simulator isn't closed properly


0.3.0 (April 25, 2017)
----------------------

**Added**

- Use logger for debug/builder output
- Implemented TensorFlow gradients for sparse Variable update Ops, to allow
  models with those elements to be trained
- Added tutorial/examples on using ``Simulator.train``
- Added support for training models when ``unroll_simulation=False``
- Compatibility changes for Nengo 2.4.0
- Added a new graph planner algorithm, which can improve simulation speed at
  the cost of build time

**Changed**

- Significant improvements to simulation speed

  - Use sparse Variable updates for signals.scatter/gather
  - Improved graph optimizer memory organization
  - Implemented sparse matrix multiplication op, to allow more aggressive
    merging of DotInc operators

- Significant improvements to build speed

  - Added early termination to graph optimization
  - Algorithmic improvements to graph optimization functions

- Reorganized documentation to more clearly direct new users to relevant
  material

**Fixed**

- Fix bug where passing a built model to the Simulator more than once would
  result in an error
- Cache result of calls to ``tensor_graph.build_loss/build_optimizer``, so that
  we don't unnecessarily create duplicate elements in the graph on repeated
  calls
- Fix support for Variables on GPU when ``unroll_simulation=False``
- SimPyFunc operators will always be assigned to CPU, even when
  ``device="/gpu:0"``, since there is no GPU kernel
- Fix bug where ``Simulator.loss`` was not being computed correctly for
  models with internal state
- Data/targets passed to ``Simulator.train`` will be truncated if not evenly
  divisible by the specified minibatch size
- Fixed bug where in some cases Nodes with side effects would not be run if
  their output was not used in the simulation
- Fixed bug where strided reads that cover a full array would be interpreted as
  non-strided reads of the full array


0.2.0 (March 13, 2017)
----------------------

Initial release of TensorFlow-based NengoDL


0.1.0 (June 12, 2016)
---------------------

Initial release of Lasagne-based NengoDL
