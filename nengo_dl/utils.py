"""
Utility objects used throughout the code base.
"""

from __future__ import print_function

import logging
import re
import sys
import threading
import time
import warnings

from nengo.exceptions import SimulationError
import numpy as np
import progressbar
import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function

logger = logging.getLogger(__name__)


def sanitize_name(name):
    """
    Remove illegal TensorFlow name characters from string.

    Valid TensorFlow name characters are ``[A-Za-z0-9_.\\-/]``

    Parameters
    ----------
    name : str
        Name to be sanitized

    Returns
    -------
    sanitized : str
        Sanitized name
    """

    if not isinstance(name, str):
        name = str(name)

    name = name.replace(" ", "_")
    name = name.replace(":", "_")

    valid_exp = re.compile(r"[A-Za-z0-9_.\-/]")

    return "".join([c for c in name if valid_exp.match(c)])


def function_name(func, sanitize=True):
    """
    Get the name of the callable object ``func``.

    Parameters
    ----------
    func : callable
        Callable object (e.g., function, callable class)
    sanitize : bool
        If True, remove any illegal TensorFlow name characters from name

    Returns
    -------
    name : str
        Name of ``func`` (optionally sanitized)
    """

    name = getattr(func, "__name__", func.__class__.__name__)
    if sanitize:
        name = sanitize_name(name)

    return name


def align_func(output_shape, output_dtype):
    """
    Decorator that ensures the output of ``func`` is an
    `~numpy.ndarray` with the given shape and dtype.

    Parameters
    ----------
    output_shape : tuple of int
        Desired shape for function output (must have the same size as actual
        function output)
    output_dtype : ``tf.DType`` or `~numpy.dtype`
        Desired dtype of function output

    Raises
    ------
    `~nengo.exceptions.SimulationError`
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
    """
    Inserts a print statement into the TensorFlow graph.

    Parameters
    ----------
    input : ``tf.Tensor``
        The value of this tensor will be printed whenever it is computed
        in the graph
    message : str
        String prepended to the value of ``input``, to help with logging

    Returns
    -------
    op : ``tf.Tensor``
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
    """
    Searches through a TensorFlow graph to find non-differentiable elements
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
    """
    ProgressBar widget for progress bars with possibly unknown duration.

    Parameters
    ----------
    msg : str
        A message to be displayed in the middle of the progress bar
    finish_msg : str
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
    """
    Handles progress bar display for some tracked process.

    Parameters
    ----------
    present : str
        Description of process in present (e.g., "Simulating")
    past : str
        Description of process in past (e.g., "Simulation")
    max_value : int or None
        The maximum number of steps in the tracked process (or ``None`` if
        the maximum number of steps is unknown)
    vars : list of str
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
        """
        Advance the progress bar one step.

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
        """
        Creates a new progress bar for tracking a sub-process.

        Parameters
        ----------
        msg : str
            Description of sub-process
        """

        if self.sub_bar is not None and self.sub_bar.finished is False:
            self.sub_bar.finish()

        self.sub_bar = SubProgressBar(
            present="%s: %s" % (self.present, msg) if msg else self.present,
            **kwargs)

        return self.sub_bar

    @property
    def max_steps(self):
        """
        Alias for max_value to allow this to work with Nengo progress bar
        interface.
        """
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


class SubProgressBar(ProgressBar):  # pylint: disable=too-many-ancestors
    """
    A progress bar representing a sub-task within an overall progress bar.
    """

    def finish(self):
        """Finishing a sub-progress bar doesn't start a new line."""
        super(SubProgressBar, self).finish(end="\r")


class NullProgressBar(progressbar.NullBar):  # pylint: disable=too-many-ancestors
    """
    A progress bar that does nothing.

    Used to replace ProgressBar when we want to disable output.
    """

    def __init__(self, present="", past=None, max_value=1, vars=None,
                 **kwargs):
        super(NullProgressBar, self).__init__(max_value=max_value, **kwargs)

    def sub(self, *args, **kwargs):
        """
        Noop for creating a sub-progress bar.
        """
        return self

    def step(self, **kwargs):
        """
        Noop for incrementing the progress bar.
        """


def minibatch_generator(data, minibatch_size, shuffle=True,
                        truncation=None, rng=None):
    """
    Generator to yield ``minibatch_sized`` subsets from ``inputs`` and
    ``targets``.

    Parameters
    ----------
    data : dict of {``NengoObject``: `~numpy.ndarray`}
        Data arrays to be divided into minibatches.
    minibatch_size : int
        The number of items in each minibatch
    shuffle : bool
        If True, the division of items into minibatches will be randomized each
        time the generator is created
    truncation : int
        If not None, divide the data up into sequences of ``truncation``
        timesteps.
    rng : `~numpy.random.RandomState`
        Seeded random number generator

    Yields
    ------
    offset : int
        The simulation step at which the returned data begins (will only be
        nonzero if ``truncation`` is not ``None``).
    inputs : dict of {`~nengo.Node`: `~numpy.ndarray`}
        The same structure as ``inputs``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    targets : dict of {`~nengo.Probe`: `~numpy.ndarray`}
        The same structure as ``targets``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    """

    if isinstance(data, int):
        n_inputs = None
        n_steps = data
    else:
        n_inputs, n_steps = next(iter(data.values())).shape[:2]

    if rng is None:
        rng = np.random

    if truncation is None:
        truncation = n_steps

    if n_steps % truncation != 0:
        warnings.warn(UserWarning(
            "Length of training data (%d) is not an even multiple of "
            "truncation length (%d); this may result in poor "
            "training results" % (n_steps, truncation)))

    if n_inputs is None:
        # no input to divide up, so we just return the
        # number of steps to be run based on the truncation
        for j in range(0, n_steps, truncation):
            yield (j, min(truncation, n_steps - j))
    else:
        if shuffle:
            perm = rng.permutation(n_inputs)
        else:
            perm = np.arange(n_inputs)

        if n_inputs % minibatch_size != 0:
            warnings.warn(UserWarning(
                "Number of data elements (%d) is not an even multiple of "
                "minibatch size (%d); inputs will be truncated" %
                (n_inputs, minibatch_size)))
            perm = perm[:-(n_inputs % minibatch_size)]

        for i in range(0, n_inputs - n_inputs % minibatch_size,
                       minibatch_size):
            mini_data = {k: v[perm[i:i + minibatch_size]]
                         for k, v in data.items()}

            for j in range(0, n_steps, truncation):
                yield (j, {k: v[:, j:j + truncation]
                           for k, v in mini_data.items()})
