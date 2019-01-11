Release History
===============

.. Changelog entries should follow this format:

   version (release date)
   ----------------------

   **section**

   - One-line description of change (link to GitHub issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Fixed
   - Deprecated
   - Removed

2.1.1 (January 11, 2019)
------------------------

**Added**

- Added ``nengo_dl.obj`` as a shortcut alias for ``nengo_dl.objectives``.
- Added tutorial for `Nengo users coming to NengoDL
  <https://www.nengo.ai/nengo-dl/examples/from-nengo.html>`_
- Added tutorial for `TensorFlow users coming to NengoDL
  <https://www.nengo.ai/nengo-dl/examples/from-tensorflow.html>`_

**Changed**

- Increased minimum ``progressbar2`` version to 3.39.0.
- We now only provide ``sdist`` releases, not ``bdist_wheel``. Due to the way
  the TensorFlow packages are organized, ``bdist_wheel``  forces any existing
  TensorFlow installations (e.g. ``tensorflow-gpu`` or ``tf-nightly``)
  to be overwritten by ``tensorflow``, which we don't want to do.

**Removed**

- Removed the ``nef-init`` tutorial (replaced by the new ``from-nengo``
  tutorial).

2.1.0 (December 5, 2018)
------------------------

**Added**

- Added a built-in objective to assist in applying regularization during
  training.
- Added `keep_history config option
  <https://www.nengo.ai/nengo-dl/config.html#keep-history>`_, which can be set
  to ``False`` on Probes if only the data from the most recent simulation step
  is desired (as opposed to the default behaviour of keeping the data from
  all steps).

**Changed**

- Moved ``utils.mse`` to ``objectives.mse``.
- ``sim.loss`` will now apply ``nengo_dl.objectives.mse`` to all probes in
  ``data`` if no explicit ``objective`` is given (mirroring the default
  behaviour in ``sim.train``).
- The Spaun benchmark network will now be installed through pip rather than
  manually cloning and importing the repo.

**Fixed**

- Fixed objective argument parsing if objective is a callable class or method.
- Fixed bug in ``sim.train`` 1-step synapse warning when explicitly specifying
  ``n_steps`` (rather than passing in ``data``).

**Deprecated**

- Passing ``"mse"`` as the objective in ``sim.train``/``sim.loss`` is no longer
  supported.  Use the function ``nengo_dl.objectives.mse`` instead.

2.0.0 (November 23, 2018)
-------------------------

**Breaking API changes**

- ``sim.train`` and ``sim.loss`` now accept a single ``data`` argument, which
  combines the previous ``inputs`` and ``targets`` arguments. For example,

  .. code-block:: python

    sim.train({my_node: x}, {my_probe: y}, ...)

  is now equivalent to

  .. code-block:: python

    sim.train({my_node: x, my_probe: y}, ...)

  The motivation for this change is that not all objective functions require
  target values. Switching to the more generic ``data`` argument simplifies
  the API and makes it more flexible, allowing users to specify whatever
  training/loss data is actually required.
- The ``objective`` argument in ``sim.train``/``sim.loss`` is now always
  specified as a dictionary mapping probes to objective functions.  Note that
  this was available but optional previously; it was also possible to pass
  a single value for the objective function, which would be applied to all
  probes in ``targets``.  The latter is no longer supported.  For example,

  .. code-block:: python

    sim.train(..., objective="mse")

  must now be explicitly specified as

  .. code-block:: python

    sim.train(..., objective={my_probe: "mse"})

  The motivation for this change is that, especially with the other new
  features introduced in the 2.0 update, there were a lot of different ways to
  specify the ``objective`` argument.  This made it somewhat unclear how
  exactly this argument worked, and the automatic "broadcasting" was also
  ambiguous (e.g., should the single objective be applied to each probe
  individually, or to all of them together?).  Making the argument explicit
  helps clarify the mental model.

**Added**

- An integer number of steps can now be passed for the
  ``sim.loss``/``sim.train`` data argument, if no input/target data is
  required.
- The ``objective`` dict in ``sim.train``/``sim.loss`` can now contain
  tuples of probes as the keys, in which case the objective function will be 
  called with a corresponding tuple of probe/target values as each argument.
- Added the ``sim.run_batch`` function.  This exposes all the functionality
  that the ``sim.run``/``sim.train``/``sim.loss`` functions are based on,
  allowing advanced users full control over how to run a NengoDL simulation.
- Added option to disable progress bar in ``sim.train`` and ``sim.loss``.
- Added ``training`` argument to ``sim.loss`` to control whether the loss
  is evaluated in training or inference mode.
- Added support for the new Nengo ``Transform`` API (see
  https://github.com/nengo/nengo/pull/1481).

**Changed**

- Custom objective functions passed to ``sim.train``/``sim.loss`` can now
  accept a single argument (``my_objective(outputs): ...`` instead of
  ``my_objective(outputs, targets): ...``) if no target values are required.
- ``utils.minibatch_generator`` now accepts a single ``data`` argument rather
  than ``inputs`` and ``targets`` (see discussion in "Breaking API changes").
- ``sim.training_step`` is now the same as
  ``tf.train.get_or_create_global_step()``.
- Switched documentation to new
  `nengo-sphinx-theme <https://github.com/nengo/nengo-sphinx-theme>`_.
- Reorganized documentation into "User guide" and "API reference" sections.
- Improve build speed of models with large constants
  (`#69 <https://github.com/nengo/nengo-dl/pull/69>`_)
- Moved op-specific merge logic into the ``OpBuilder`` classes.

**Fixed**

- Ensure that training step is always updated before TensorBoard events are
  added (previously it could update before or after depending on the platform).

**Deprecated**

- The ``sim.run`` ``input_feeds`` argument has been renamed to ``data`` (for
  consistency with other simulator functions).

**Removed**

- NengoDL no longer supports Python 2 (see https://python3statement.org/ for
  more information)

1.2.1 (November 2, 2018)
------------------------

**Added**

- Added a warning if users run one-timestep training with a network containing
  synaptic filters.

**Changed**

- Test Simulator parameters are now controlled through pytest arguments,
  rather than environment variables.
- Disable INFO-level TensorFlow logging (from C side) on import.  Added a
  NengoDL log message indicating the device the simulation will run on, as
  a more concise replacement.
- Boolean signals are now supported
  (`#61 <https://github.com/nengo/nengo-dl/issues/61>`_)

**Fixed**

- Avoid backpropagating NaN gradients from spiking neurons.
- Fixed an error that was thrown when calling ``get_tensor`` on a ``Signal``
  that was first initialized inside the Simulation while loop
  (`#56 <https://github.com/nengo/nengo-dl/issues/56>`_)
- Allow TensorNodes to run in Nengo GUI.
- Avoid bug in TensorFlow 1.11.0 that prevents certain models from
  running (see https://github.com/tensorflow/tensorflow/issues/23383). Note
  that this doesn't prevent this from occurring in user models, as we cannot
  control the model structure there. If your model hangs indefinitely when
  you call ``sim.train``, try downgrading to TensorFlow 1.10.0.
- Ensure that ``sim.training_step`` is always updated after the optimization
  step (in certain race conditions it would sometimes update part-way through
  the optimization step).

1.2.0 (September 5, 2018)
-------------------------

**Added**

- NengoDL will now automatically use a rate-based approximation to compute the
  gradient for spiking neuron types, if one is known (no more need to manually
  swap neuron types for training and inference).
- Added ``nengo_dl.configure_settings(inference_only=True)`` option, which will
  build the network in inference-only mode.  This will slightly improve the
  inference speed of the simulation, but the network will not be trainable.
- Added ``nengo_dl.configure_settings(lif_smoothing=x)`` option, which will
  control how much smoothing is applied to the LIF function during gradient
  calculations (if any).
- Added `documentation <https://www.nengo.ai/nengo-dl/config.html>`__ on the
  various NengoDL config options.
- Added better validation for TensorNode output when ``size_out != None``
  (`#51 <https://github.com/nengo/nengo-dl/issues/51>`_)

**Changed**

- More informative error message if the user tries to pass target values for
  a probe that isn't used in the objective function.
- Switched to ADD_N gradient accumulation (from TREE); this will increase
  the memory usage during training, but improve performance.
- Revert to ``Timeline`` profiling method. ``tf.profiler`` can produce
  incorrect output, and isn't maintained any more
  (https://github.com/tensorflow/tensorflow/issues/15214#issuecomment-382442357)
- Reduce memory usage during training by caching temporary variables used
  when computing ``ScatterUpdate`` gradient.
- Increase minimum TensorFlow version to 1.4.0.
- Increased minimum NumPy version to 1.12.1 (required by TensorFlow)
- Sort write signals as well as reads during graph optimization (encourages
  tighter partitioning, which can improve training/inference speed).
- Moved ``configure_settings`` from ``utils.py`` to ``config.py``.

**Fixed**

- Fixed a bug where
  ``nengo_dl.dists.VarianceScaling(..., distribution="normal")`` did not
  respect the seed if one was given.

**Deprecated**

- The ``Simulator(dtype=...)`` argument has been deprecated; use
  ``nengo_dl.configure_settings(dtype=...)`` instead.  Will be removed in
  1.3.0.

1.1.0 (July 24, 2018)
---------------------

**Added**

- The default TensorFlow Session is now set to the underlying Simulator session
  within the Simulator context.
- Added CLI for benchmarks.py
- Added ``sim.freeze_params`` tool, to more easily extract model parameters for
  reuse in different Simulators.
- Added `documentation on saving and loading model parameters
  <https://www.nengo.ai/nengo-dl/training.html#saving-and-loading-parameters>`_.
- Added `Spaun <http://science.sciencemag.org/content/338/6111/1202.full>`_
  example in ``benchmarks.py``

**Changed**

- Move ``tensorflow-gpu`` installation check to Simulator init, and only apply
  if ``device=None``.
- Switched to ``pylint`` for style checks.
- TensorFlow INFO-level log messages are now disabled by default on import
- All previous releases now tracked in documentation
- Updated spiking MNIST example to simplify and improve performance.
- Passing unknown configuration options to ``nengo_dl.configure_settings``
  will now give a more explicit error message.
- Improved speed of parameter fetching though ``get_nengo_params``
- Raise a warning if user tries to train a network with non-differentiable
  elements (requires ``tensorflow>=1.9.0``)
- Improved accuracy of ``SoftLIFRate`` implementation for small values (`#45
  <https://github.com/nengo/nengo-dl/pull/45>`_)
- Simplified how ``TensorSignals`` are loaded into the TensorFlow graph

**Fixed**

- Better handling of Simulator errors not associated with a specific op (fixes
  `#41 <https://github.com/nengo/nengo-dl/issues/41>`_)
- Fixed node outputs changing after simulator is built (fixes `#4
  <https://github.com/nengo/nengo-dl/issues/4>`__)
- Fixed some broken cross references in the documentation
- Fixed several edge cases for ``get_nengo_params``; don't use trained gains
  for direct neuron connections, error raised if ``get_nengo_params`` applied
  to an Ensemble with Direct neurons
- Compatible with ``tensorflow==1.9.0`` release
- Fixed bug in ``nengo_dl.configure_settings(session_config=...)`` when passing
  a pre-build model to the Simulator instead of a Network
- Fixed TensorFlow version comparisons for 1.10.0

**Deprecated**

- ``Simulator.trange`` argument ``dt`` has been deprecated (replaced with
  ``sample_every``, see https://github.com/nengo/nengo/pull/1384)

**Removed**

- Removed ``nengo_dl.DATA_DIR`` constant
- Removed ``benchmarks.compare_backends`` (use
  ``whitepaper2018_plots.py:compare_backends`` instead)
- Removed ``ghp-import`` dependency


1.0.0 (May 30, 2018)
--------------------

**Added**

- User can now directly specify the output error gradient, rather than using
  targets/objective (useful for when you have some external process for
  computing error that is not easy to implement as an objective function).
  See `the documentation
  <https://www.nengo.ai/nengo-dl/training.html#objective>`__ for details.
- Added `NengoDL white paper <https://arxiv.org/abs/1805.11144>`_

**Changed**

- Extra requirements for documentation/testing are now stored in ``setup.py``'s
  ``extra_requires`` instead of ``requirements-*.txt``.  For example, instead
  of doing ``pip install -r requirements-test.txt``, instead use
  ``pip install nengo-dl[tests]`` (or ``pip install -e .[tests]`` for a
  developer installation).
- Improved efficiency of PES implementation

**Removed**

- Removed ``sphinxcontrib-versioning`` dependency for building documentation

0.6.2 (May 4, 2018)
-------------------

**Added**

- Added ``sim.get_nengo_params`` function to more easily extract
  model parameters for reuse when building different models.
- Added ``Simulator(..., progress_bar=False)`` option to disable the progress
  information printed to console when the network is building.
- TensorFlow session config options can now be set using
  ``nengo_dl.configure_settings`` (e.g.,
  ``nengo_dl.configure_settings(session_config={"gpu_options.allow_growth": True})``)
- The signal sorting/graph simplificaton functions can now be configured
  through ``nengo_dl.configure_settings``
- Added ``extra_feeds`` parameter to ``sim.run/train/loss``, which can be
  used to feed Tensor values directly into the TensorFlow session

**Changed**

- Improved speed of PES implementation by adding a custom operator.
- Renamed project from ``nengo_dl`` to ``nengo-dl`` (to be more consistent with
  standard conventions).  This only affects the display name of the project
  on PyPI/GitHub, and the documentation now resides at
  https://www.nengo.ai/nengo-dl/; there are no functional changes to user code.
- Minor efficiency improvements to graph planner
- Avoid using ``tf.constant``, to get around TensorFlow's 2GB limit on graph
  size when building large models

**Fixed**

- Checking ``nengo_dl`` version without ``nengo`` installed will no longer
  result in an error.
- Updated progress bar to work with ``progressbar2>=3.37.0``
- Updated PES implementation to work with generic synapse types
  (see https://github.com/nengo/nengo/pull/1095)
- Fixed installation to work with ``pip>=10.0``
- Fixed bug when using a TensorNode with a ``pre_build`` function and
  ``size_in==0``

0.6.1 (March 7, 2018)
---------------------

**Added**

- Added TensorFlow implementation for ``nengo.SpikingRectifiedLinear`` neuron
  type.

**Changed**

- Optimizer variables (e.g., momentum values) will only be initialized the
  first time that optimizer is passed to ``sim.train``.  Subsequent calls to
  ``sim.train`` will resume with the values from the previous call.
- Low-level simulation input/output formats have been reworked to make them
  slightly easier to use (for users who want to bypass ``sim.run`` or
  ``sim.train`` and access the TensorFlow session directly).
- Batch dimension will always be first (if present) when checking model
  parameters via ``sim.data``.
- TensorFlow ops created within the Simulator context will now default to
  the same device as the Simulator.
- Update minimum Nengo version to 2.7.0

**Fixed**

- Better error message if training data has incorrect rank
- Avoid reinstalling TensorFlow if one of the nightly build packages is already
  installed
- Lowpass synapse can now be applied to multidimensional inputs
- TensorNodes will no longer be built into the default graph when checking
  their output dimensionality.

**Removed**

- Removed ``utils.cast_dtype`` function

0.6.0 (December 13, 2017)
-------------------------

**Added**

- The ``SoftLIFRate`` neuron type now has an ``amplitude`` parameter, which
  scales the output in the same way as the new ``amplitude`` parameter in
  ``LIF``/``LIFRate`` (see `Nengo PR #1325
  <https://github.com/nengo/nengo/pull/1325>`_).
- Added ``progress_bar=False`` option to ``sim.run``, which will disable the
  information about the simulation status printed to standard output (`#17
  <https://github.com/nengo/nengo-dl/issues/17>`_).
- Added progress bars for the build/simulation process.
- Added truncated backpropagation option to ``sim.train`` (useful for reducing
  memory usage during training).  See `the documentation for details
  <https://www.nengo.ai/nengo-dl/training.html#truncation>`__.

**Changed**

- Changed the default ``tensorboard`` argument in ``Simulator`` from ``False``
  to ``None``
- Use the new `tf.profiler
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md>`_
  tool to collect profiling data in ``sim.run_steps`` and ``sim.train`` when
  ``profile=True``.
- Minor improvements to efficiency of build process.
- Minor improvements to simulation efficiency targeting small ops
  (``tf.reshape/identity/constant``).
- Process inputs are now reseeded for each input when batch processing (if seed
  is not manually set).
- Users can pass a dict of config options for the ``profile`` argument in
  ``run_steps``/``train``, which will be passed on to the TensorFlow
  profiler; see the ``tf.profiler`` documentation for the `available options
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md>`_.

**Removed**

- Removed ``backports.print_function`` dependency

**Fixed**

- Fixed a bug where input nodes that were only read as a view were not
  feedable
- Updated ``tensorflow-gpu`` installation check
- Improved numerical stability of ``LIFRate`` gradients  (`#26 
  <https://github.com/nengo/nengo-dl/issues/26>`_)
- Added more informative error message when data is provided with fewer items
  than ``sim.minibatch_size`` (`#30 <https://github.com/nengo/nengo-dl/issues/30>`_)

0.5.2 (October 11, 2017)
------------------------

**Added**

- TensorNode outputs can now define a ``post_build`` function that will be
  executed after the simulation is initialized (see the `TensorNode
  documentation for details
  <https://www.nengo.ai/nengo-dl/tensor_node.html>`_).
- Added functionality for outputting summary data during the training process
  that can be viewed in TensorBoard (see the `sim.train documentation
  <https://www.nengo.ai/nengo-dl/training.html#summaries>`__).
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
  <https://www.nengo.ai/nengo-dl/training.html#objective>`__).

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
  (`#9 <https://github.com/nengo/nengo-dl/issues/9>`_)
- Fixed bug where non-slot optimizer variables were not initialized
  (`#11 <https://github.com/nengo/nengo-dl/issues/11>`_)
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
  <https://www.nengo.ai/nengo-dl/tensor_node.html>`_)
- Added an example demonstrating `how to train a neural network
  that can run in spiking neurons
  <https://www.nengo.ai/nengo-dl/examples/spiking_mnist.html>`_
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
  <https://www.nengo.ai/nengo-dl/training.html#choosing-which-elements-to-optimize>`_
  for details.
- Calling ``Simulator.train``/``Simulator.loss`` no longer resets the internal
  state of the simulation (so they can be safely intermixed with calls to
  ``Simulator.run``)

**Deprecated**

- The old ``step_blocks``/``unroll_simulation`` syntax has been fully
  deprecated, and will result in errors if used

**Fixed**

- Fixed bug related to changing the output of a Node after the model is
  constructed (`#4 <https://github.com/nengo/nengo-dl/issues/4>`_)
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
  <https://www.nengo.ai/nengo-dl/training.html>`_)
- Added some code examples (see the ``docs/examples`` directory, or the
  `pre-built examples in the documentation
  <https://www.nengo.ai/nengo-dl/examples.html>`_)
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
