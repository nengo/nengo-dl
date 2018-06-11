from __future__ import print_function

from functools import partial
import logging
import re
import sys
import threading
import time
import warnings

from nengo import Connection, Ensemble, Network, ensemble
from nengo.exceptions import SimulationError, ConfigError, NetworkContextError
from nengo.params import BoolParam, Parameter
import numpy as np
import progressbar
import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function

logger = logging.getLogger(__name__)


def sanitize_name(name):
    """Remove illegal TensorFlow name characters from string.

    Valid TensorFlow name characters are ``[A-Za-z0-9_.\\-/]``

    Parameters
    ----------
    name : str
        Name to be sanitized

    Returns
    -------
    str
        Sanitized name
    """

    if not isinstance(name, str):
        name = str(name)

    name = name.replace(" ", "_")
    name = name.replace(":", "_")

    valid_exp = re.compile(r"[A-Za-z0-9_.\-/]")

    return "".join([c for c in name if valid_exp.match(c)])


def function_name(func, sanitize=True):
    """Get the name of the callable object ``func``.

    Parameters
    ----------
    func : callable
        Callable object (e.g., function, callable class)
    sanitize : bool, optional
        If True, remove any illegal TensorFlow name characters from name

    Returns
    -------
    str
        Name of ``func`` (optionally sanitized)
    """

    name = getattr(func, "__name__", func.__class__.__name__)
    if sanitize:
        name = sanitize_name(name)

    return name


def align_func(output_shape, output_dtype):
    """Decorator that ensures the output of ``func`` is an
    :class:`~numpy:numpy.ndarray` with the given shape and dtype.

    Parameters
    ----------
    output_shape : tuple of int
        Desired shape for function output (must have the same size as actual
        function output)
    output_dtype : ``tf.DType`` or :class:`~numpy:numpy.dtype`
        Desired dtype of function output

    Raises
    ------
    :class:`~nengo:nengo.exceptions.SimulationError`
        If the function returns ``None`` or a non-finite value.
    """

    if isinstance(output_dtype, tf.DType):
        output_dtype = output_dtype.as_numpy_dtype

    def apply_align(func):
        def aligned_func(*args):
            output = func(*args)

            if output is None:
                raise SimulationError(
                    "Function %r returned None" %
                    function_name(func, sanitize=False))
            try:
                if not np.all(np.isfinite(output)):
                    raise SimulationError(
                        "Function %r returned invalid value %r" %
                        (function_name(func, sanitize=False), output))
            except (TypeError, ValueError):
                raise SimulationError(
                    "Function %r returned a value %r of invalid type %r" %
                    (function_name(func, sanitize=False), output,
                     type(output)))
            output = np.asarray(output, dtype=output_dtype)
            output = output.reshape(output_shape)
            return output

        return aligned_func

    return apply_align


def print_op(input, message):
    """Inserts a print statement into the TensorFlow graph.

    Parameters
    ----------
    input : ``tf.Tensor``
        The value of this tensor will be printed whenever it is computed
        in the graph
    message : str
        String prepended to the value of ``input``, to help with logging

    Returns
    -------
    ``tf.Tensor``
        New tensor representing the print operation applied to ``input``

    Notes
    -----
    This is what ``tf.Print`` is supposed to do, but it doesn't seem to work
    consistently.
    """

    def print_func(x):  # pragma: no cover
        print(message, str(x))
        return x

    with tf.device("/cpu:0"):
        output = tf.py_func(print_func, [input], input.dtype)
    output.set_shape(input.get_shape())

    return output


def find_non_differentiable(inputs, outputs):
    """Searches through a TensorFlow graph to find non-differentiable elements
    between ``inputs`` and ``outputs`` (elements that would prevent us from
    computing ``d_outputs / d_inputs``.

    Parameters
    ----------
    inputs : list of ``tf.Tensor``
        Input tensors
    outputs : list of ``tf.Tensor``
        Output tensors
    """

    for o in outputs:
        if o in inputs:
            continue
        else:
            try:
                grad = get_gradient_function(o.op)

                if grad is None and len(o.op.inputs) > 0:
                    # note: technically we're not sure that this op is
                    # on the path to inputs. we could wait and propagate this
                    # until we find inputs, but that can take a long time for
                    # large graphs. it seems more useful to fail quickly, and
                    # risk some false positives
                    raise LookupError
                find_non_differentiable(inputs, o.op.inputs)
            except LookupError:
                raise SimulationError(
                    "Graph contains non-differentiable "
                    "elements: %s" % o.op)


class MessageBar(progressbar.BouncingBar):
    """ProgressBar widget for progress bars with possibly unknown duration.

    Parameters
    ----------
    msg : str, optional
        A message to be displayed in the middle of the progress bar
    finish_msg : str, optional
        A message to be displayed when the progress bar is finished
    """

    def __init__(self, msg="", finish_msg="", **kwargs):
        super(MessageBar, self).__init__(**kwargs)
        self.msg = msg
        self.finish_msg = finish_msg

    def __call__(self, progress, data, width):
        if progress.end_time:
            return self.finish_msg

        if progress.max_value is progressbar.UnknownLength:
            bar = progressbar.BouncingBar
        else:
            bar = progressbar.Bar
        line = bar.__call__(self, progress, data, width)

        if data["percentage"] is None:
            msg = self.msg
        else:
            msg = "%s (%d%%)" % (self.msg, data["percentage"])

        offset = width // 2 - len(msg) // 2

        return line[:offset] + msg + line[offset + len(msg):]


class ProgressBar(progressbar.ProgressBar):  # pylint: disable=too-many-ancestors
    """Handles progress bar display for some tracked process.

    Parameters
    ----------
    present : str, optional
        Description of process in present (e.g., "Simulating")
    past : str, optional
        Description of process in past (e.g., "Simulation")
    max_value : int or None, optional
        The maximum number of steps in the tracked process (or ``None`` if
        the maximum number of steps is unknown)
    vars : list of str, optional
        Extra variables that will be displayed at the end of the progress bar

    Notes
    -----
    Launches a separate thread to handle the progress bar display updates.
    """

    def __init__(self, present="", past=None, max_value=1, vars=None,
                 **kwargs):

        self.present = present
        self.sub_bar = None
        self.finished = None

        if past is None:
            past = present

        self.msg_bar = MessageBar(
            msg=present, finish_msg="%s finished in" % past)
        widgets = [self.msg_bar, " "]

        if max_value is None:
            widgets.append(progressbar.Timer(format="%(elapsed)s"))
        else:
            widgets.append(progressbar.ETA(
                format="ETA: %(eta)s",
                format_finished="%(elapsed)s"))

        if vars is not None:
            self.var_vals = progressbar.FormatCustomText(
                " (" + ", ".join("%s: %%(%s)s" % (v, v) for v in vars) + ")",
                {v: "---" for v in vars})
            widgets.append(self.var_vals)
        else:
            self.var_vals = None

        def update_thread():
            while not self.finished:
                if self.sub_bar is None or self.sub_bar.finished:
                    self.update()
                time.sleep(0.001)

        self.thread = threading.Thread(target=update_thread)
        self.thread.daemon = True

        if max_value is None:
            max_value = progressbar.UnknownLength

        super(ProgressBar, self).__init__(
            poll_interval=0.1, widgets=widgets, fd=sys.stdout,
            max_value=max_value, **kwargs)

    def start(self, **kwargs):
        """Start tracking process, initialize display."""

        super(ProgressBar, self).start(**kwargs)

        self.finished = False
        self.thread.start()

        return self

    def finish(self, **kwargs):
        """Stop tracking process, finish display."""

        if self.sub_bar is not None and self.sub_bar.finished is False:
            self.sub_bar.finish()

        self.finished = True
        self.thread.join()

        super(ProgressBar, self).finish(**kwargs)

    def step(self, **vars):
        """Advance the progress bar one step.

        Parameters
        ----------
        vars : dict of {str: str}
            Values for the extra variables displayed at the end of the progress
            bar (defined in ``__init__``)
        """

        if self.var_vals is not None:
            self.var_vals.update_mapping(**vars)
        self.value += 1

    def sub(self, msg=None, **kwargs):
        """Creates a new progress bar for tracking a sub-process.

        Parameters
        ----------
        msg : str, optional
            Description of sub-process
        """

        if self.sub_bar is not None and self.sub_bar.finished is False:
            self.sub_bar.finish()

        self.sub_bar = ProgressBar(
            present="%s: %s" % (self.present, msg) if msg else self.present,
            **kwargs)
        self.sub_bar.finish = partial(self.sub_bar.finish, end="\r")

        return self.sub_bar

    @property
    def max_steps(self):
        return self.max_value

    @max_steps.setter
    def max_steps(self, n):
        self.max_value = n

    def __enter__(self):
        super(ProgressBar, self).__enter__()

        return self.start()

    def __next__(self):
        """Wraps an iterable using this progress bar."""

        try:
            if self.start_time is None:
                self.start()
            else:
                self.step()
            value = next(self._iterable)
            return value
        except StopIteration:
            self.finish()
            raise

    next = __next__  # for python 2.x


class NullProgressBar(progressbar.NullBar):  # pylint: disable=too-many-ancestors
    """A progress bar that does nothing.

    Used to replace ProgressBar when we want to disable output.
    """

    def __init__(self, present="", past=None, max_value=1, vars=None,
                 **kwargs):
        super(NullProgressBar, self).__init__(max_value=max_value, **kwargs)

    def sub(self, *args, **kwargs):
        return self

    def step(self, **kwargs):
        pass


def minibatch_generator(inputs, targets, minibatch_size, shuffle=True,
                        truncation=None, rng=None):
    """Generator to yield ``minibatch_sized`` subsets from ``inputs`` and
    ``targets``.

    Parameters
    ----------
    inputs : dict of {:class:`~nengo:nengo.Node`: \
                      :class:`~numpy:numpy.ndarray`}
        Input values for Nodes in the network
    targets : dict of {:class:`~nengo:nengo.Probe`: \
                       :class:`~numpy:numpy.ndarray`}
        Desired output value at Probes, corresponding to each value in
        ``inputs``
    minibatch_size : int
        The number of items in each minibatch
    shuffle : bool, optional
        If True, the division of items into minibatches will be randomized each
        time the generator is created
    truncation : int, optional
        If not None, divide the data up into sequences of ``truncation``
        timesteps.
    rng : :class:`~numpy:numpy.random.RandomState`, optional
        Seeded random number generator

    Yields
    ------
    offset : int
        The simulation step at which the returned data begins (will only be
        nonzero if ``truncation`` is not ``None``).
    inputs : dict of {:class:`~nengo:nengo.Node`: \
                      :class:`~numpy:numpy.ndarray`}
        The same structure as ``inputs``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    targets : dict of {:class:`~nengo:nengo.Probe`: \
                       :class:`~numpy:numpy.ndarray`}
        The same structure as ``targets``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    """

    n_inputs, n_steps = next(iter(inputs.values())).shape[:2]

    if rng is None:
        rng = np.random

    if shuffle:
        perm = rng.permutation(n_inputs)
    else:
        perm = np.arange(n_inputs)

    if truncation is None:
        truncation = n_steps

    if n_inputs % minibatch_size != 0:
        warnings.warn(UserWarning(
            "Number of inputs (%d) is not an even multiple of "
            "minibatch size (%d); inputs will be truncated" %
            (n_inputs, minibatch_size)))
        perm = perm[:-(n_inputs % minibatch_size)]

    if n_steps % truncation != 0:
        warnings.warn(UserWarning(
            "Length of training data (%d) is not an even multiple of "
            "truncation length (%d); this may result in poor "
            "training results" % (n_steps, truncation)))

    for i in range(0, n_inputs - n_inputs % minibatch_size, minibatch_size):
        batch_inp = {n: inputs[n][perm[i:i + minibatch_size]] for n in inputs}
        batch_tar = {p: targets[p][perm[i:i + minibatch_size]]
                     for p in targets}

        for j in range(0, n_steps, truncation):
            yield (j, {n: batch_inp[n][:, j:j + truncation] for n in inputs},
                   {p: batch_tar[p][:, j:j + truncation] for p in targets})


def configure_settings(**kwargs):
    """
    Pass settings to ``nengo_dl`` by setting them as parameters on the
    top-level Network config.

    The settings are passed as keyword arguments to ``configure_settings``;
    e.g., to set ``trainable`` use ``configure_settings(trainable=True)``.

    Parameters
    ----------
    trainable : bool or None
        Adds a parameter to Nengo Ensembles/Connections/Networks that controls
        whether or not they will be optimized by :meth:`.Simulator.train`.
        Passing ``None`` will use the default ``nengo_dl`` trainable settings,
        or True/False will override the default for all objects.  In either
        case trainability can be further configured on a per-object basis (e.g.
        ``net.config[my_ensemble].trainable = True``.  See `the documentation
        <https://www.nengo.ai/nengo-dl/training.html#choosing-which-elements-to-optimize>`_
        for more details.
    planner : graph planning algorithm
        Pass one of the `graph planners
        <https://www.nengo.ai/nengo-dl/graph_optimizer.html>`_ to change the
        default planner.
    session_config: dict
        Config options passed to ``tf.Session`` initialization (e.g., to change
        the `GPU memory allocation method
        <https://www.tensorflow.org/programmers_guide/using_gpu#allowing_gpu_memory_growth>`_
        pass ``{"gpu_options.allow_growth": True}``).
    """

    # get the toplevel network
    if len(Network.context) > 0:
        config = Network.context[0].config
    else:
        raise NetworkContextError(
            "`configure_settings` must be called within a Network context "
            "(`with nengo.Network(): ...`)")

    try:
        params = config[Network]
    except ConfigError:
        config.configures(Network)
        params = config[Network]

    for attr, val in kwargs.items():
        if attr == "trainable":
            for obj in (Ensemble, Connection, ensemble.Neurons, Network):
                try:
                    obj_params = config[obj]
                except ConfigError:
                    config.configures(obj)
                    obj_params = config[obj]

                obj_params.set_param("trainable", BoolParam("trainable", val,
                                                            optional=True))
        else:
            params.set_param(attr, Parameter(attr, val))
