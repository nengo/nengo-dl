from __future__ import print_function

import datetime
import logging
import re
import sys
import time
import warnings

from nengo.exceptions import SimulationError
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function

logger = logging.getLogger(__name__)

if sys.version_info[:2] < (3, 3):

    def print_and_flush(*args, **kwargs):
        print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        file.flush()

else:

    def print_and_flush(*args, **kwargs):
        print(*args, flush=True, **kwargs)


def sanitize_name(name):
    """Remove illegal Tensorflow name characters from string.

    Valid Tensorflow name characters are ``[A-Za-z0-9_.\\-/]``

    Parameters
    ----------
    name : str
        name to be sanitized

    Returns
    -------
    str
        sanitized name
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
        callable object (e.g., function, callable class)
    sanitize : bool, optional
        if True, remove any illegal Tensorflow name characters from name

    Returns
    -------
    str
        (sanitized) name of ``func``
    """

    name = getattr(func, "__name__", func.__class__.__name__)
    if sanitize:
        name = sanitize_name(name)

    return name


def align_func(output_shape, output_dtype):
    """Decorator that ensures the output of ``func`` is an
    :class:`~numpy:numpy.ndarray` with the given shape and dtype.

    Raises a ``SimulationError`` if the
    function returns ``None``.

    Parameters
    ----------
    output_shape : tuple of int
        desired shape for function output (must have the same size as actual
        function output)
    output_dtype : ``tf.DType`` or :class:`~numpy:numpy.dtype`
        desired dtype of function output
    """

    if isinstance(output_dtype, tf.DType):
        output_dtype = output_dtype.as_numpy_dtype

    def apply_align(func):
        def aligned_func(*args):
            output = func(*args)
            if output is None:
                raise SimulationError(
                    "Function %r returned None" % function_name(
                        func, sanitize=False))
            output = np.asarray(output, dtype=output_dtype)
            output = output.reshape(output_shape)
            return output

        return aligned_func

    return apply_align


def print_op(input, message):
    """Inserts a print statement into the tensorflow graph.

    Parameters
    ----------
    input : ``tf.Tensor``
        the value of this tensor will be printed whenever it is computed
        in the graph
    message : str
        string prepended to the value of ``input``, to help with logging

    Returns
    -------
    ``tf.Tensor``
        new tensor representing the print operation applied to ``input``

    Notes
    -----
    This is what ``tf.Print`` is supposed to do, but it doesn't seem to work
    consistently.
    """

    def print_func(x):
        print(message, str(x))
        return x

    with tf.device("/cpu:0"):
        output = tf.py_func(print_func, [input], input.dtype)
    output.set_shape(input.get_shape())

    return output


def cast_dtype(dtype, target):
    """Changes float dtypes to the target dtype, leaves others unchanged.

    Used to map all float values to a target precision.  Also casts numpy
    dtypes to Tensorflow dtypes.

    Parameters
    ----------
    dtype : ``tf.DType`` or :class:`~numpy:numpy.dtype`
        input dtype to be converted
    target : ``tf.DType``
        floating point dtype to which all floating types should be converted

    Returns
    -------
    ``tf.DType``
        input dtype, converted to ``target`` type if necessary
    """

    if not isinstance(dtype, tf.DType):
        dtype = tf.as_dtype(dtype)

    if dtype.is_floating:
        dtype = target

    return dtype


def find_non_differentiable(inputs, outputs):
    """Searches through a Tensorflow graph to find non-differentiable elements
    between ``inputs`` and ``outputs`` (elements that would prevent us from
    computing ``d_outputs / d_inputs``.

    Parameters
    ----------
    inputs : list of ``tf.Tensor``
        input tensors
    outputs : list of ``tf.Tensor``
        output tensors
    """
    for o in outputs:
        if o in inputs:
            continue
        else:
            try:
                get_gradient_function(o.op)
                find_non_differentiable(inputs, o.op.inputs)
            except LookupError as e:
                logger.exception(e)
                raise SimulationError(
                    "Graph contains non-differentiable "
                    "elements: %s" % o.op)


class ProgressBar(object):
    """Displays a progress bar and ETA for tracked steps.

    Parameters
    ----------
    max_steps : int
        number of steps required to complete the tracked process
    label : str, optional
        a description of what is being tracked
    """

    def __init__(self, max_steps, label=None):
        self.max_steps = max_steps
        self.width = 30
        self.label = label

        self.reset()

    def reset(self):
        """Reset the tracker to initial conditions."""

        self.curr_step = 0
        self.start_time = time.time()
        self.progress = -1

        print_and_flush("[%s] ETA: unknown" % (" " * self.width), end="")

    def stop(self):
        """Stop the progress tracker.

        Normally this will be called automatically when ``max_steps`` is
        reached, but it can be called manually to trigger an early finish.
        """

        line = "\n"
        line += ("Completed" if self.label is None else
                 self.label + " completed")
        line += " in %s" % datetime.timedelta(
            seconds=int(time.time() - self.start_time))
        print(line)
        self.curr_step = None

    def step(self, msg=None):
        """Increment the progress tracker one step.

        Parameters
        ----------
        msg : str, optional
            display the given string at the end of the progress bar
        """

        self.curr_step += 1

        tmp = int(self.width * self.curr_step / self.max_steps)
        if tmp > self.progress:
            self.progress = tmp
        else:
            return

        eta = int((time.time() - self.start_time) *
                  (self.max_steps - self.curr_step) / self.curr_step)

        line = "\r[%s%s] ETA: %s" % ("#" * self.progress,
                                     " " * (self.width - self.progress),
                                     datetime.timedelta(seconds=eta))
        if msg is not None or self.label is not None:
            line += " (%s)" % self.label if msg is None else msg

        print_and_flush(line, end="")

        if self.curr_step == self.max_steps:
            self.stop()


def minibatch_generator(inputs, targets, minibatch_size, shuffle=True,
                        rng=None):
    """Generator to yield ``minibatch_sized`` subsets from ``inputs`` and
    ``targets``.

    Parameters
    ----------
    inputs : dict of {:class:`~nengo:nengo.Node`: \
                      :class:`~numpy:numpy.ndarray`}
        input values for Nodes in the network
    targets : dict of {:class:`~nengo:nengo.Probe`: \
                       :class:`~numpy:numpy.ndarray`}
        desired output value at Probes, corresponding to each value in
        ``inputs``
    minibatch_size : int
        the number of items in each minibatch
    shuffle : bool, optional
        if True, the division of items into minibatches will be randomized each
        time the generator is created
    rng : :class:`~numpy:numpy.random.RandomState`, optional
        random number generator

    Yields
    ------
    inputs : dict of {:class:`~nengo:nengo.Node`: \
                      :class:`~numpy:numpy.ndarray`}
        the same structure as ``inputs``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    targets : dict of {:class:`~nengo:nengo.Probe`: \
                       :class:`~numpy:numpy.ndarray`}
        the same structure as ``targets``, but with each array reduced to
        ``minibatch_size`` elements along the first dimension
    """

    n_inputs = next(iter(inputs.values())).shape[0]

    if rng is None:
        rng = np.random

    if shuffle:
        perm = rng.permutation(n_inputs)
    else:
        perm = np.arange(n_inputs)

    if n_inputs % minibatch_size != 0:
        warnings.warn(UserWarning(
            "Number of inputs (%d) is not an even multiple of "
            "minibatch size (%d); inputs will be truncated" %
            (n_inputs, minibatch_size)))
        perm = perm[:-(n_inputs % minibatch_size)]

    for i in range(0, n_inputs - n_inputs % minibatch_size,
                   minibatch_size):
        yield ({n: inputs[n][perm[i:i + minibatch_size]] for n in inputs},
               {p: targets[p][perm[i:i + minibatch_size]] for p in targets})
