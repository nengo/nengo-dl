"""
Utility objects used throughout the code base.
"""

from distutils.version import LooseVersion
import logging
import re
import subprocess
import sys
import threading
import time

from nengo.exceptions import SimulationError
import numpy as np
import progressbar
import tensorflow as tf


logger = logging.getLogger(__name__)

# check if GPU support is available
# note: we run this in a subprocess because list_physical_devices()
# will fix certain process-level TensorFlow configuration
# options the first time it is called
tf_gpu_installed = not subprocess.call(
    [
        sys.executable,
        "-c",
        "import sys; "
        "import tensorflow as tf; "
        "sys.exit(len(tf.config%s.list_physical_devices('GPU')) == 0)"
        % (".experimental" if LooseVersion(tf.__version__) < "2.1.0" else ""),
    ]
)


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
    output_shape : (list of) tuple of int
        Desired shape for function output(s) (must have the same size as actual
        function output)
    output_dtype : (list of) ``tf.DType`` or `~numpy.dtype`
        Desired dtype of function output(s)

    Raises
    ------
    ``nengo.exceptions.SimulationError``
        If the function returns ``None`` or a non-finite value.
    """

    single_output = isinstance(output_shape, tuple)

    if single_output:
        output_shape = [output_shape]
        output_dtype = [output_dtype]

    for i, dtype in enumerate(output_dtype):
        if isinstance(dtype, tf.DType):
            output_dtype[i] = dtype.as_numpy_dtype

    def apply_align(func):
        def aligned_func(*args):
            output = func(*args)

            if output is None:
                raise SimulationError(
                    "Function %r returned None" % function_name(func, sanitize=False)
                )

            if single_output:
                output = [output]

            for i, o in enumerate(output):
                try:
                    if not np.all(np.isfinite(o)):
                        raise SimulationError(
                            "Function %r returned invalid value %r"
                            % (function_name(func, sanitize=False), o)
                        )
                except (TypeError, ValueError):
                    raise SimulationError(
                        "Function %r returned a value %r of invalid type %r"
                        % (function_name(func, sanitize=False), o, type(o))
                    )
                o = np.asarray(o, dtype=output_dtype[i])
                o = o.reshape(output_shape[i])
                output[i] = o

            if single_output:
                output = output[0]

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

    def print_func(x):  # pragma: no cover (runs in TF)
        print(message, str(x))
        return x

    with tf.device("/cpu:0"):
        output = tf.numpy_function(print_func, [input], input.dtype)
    output.set_shape(input.shape)

    return output


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
        super().__init__(**kwargs)
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

        return line[:offset] + msg + line[offset + len(msg) :]


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

    Notes
    -----
    Launches a separate thread to handle the progress bar display updates.
    """

    def __init__(self, present="", past=None, max_value=1, **kwargs):

        self.present = present
        self.sub_bar = None
        self.finished = None

        if past is None:
            past = present

        self.msg_bar = MessageBar(msg=present, finish_msg="%s finished in" % past)
        widgets = [self.msg_bar, " "]

        if max_value is None:
            widgets.append(progressbar.Timer(format="%(elapsed)s"))
        else:
            widgets.append(
                progressbar.ETA(format="ETA: %(eta)s", format_finished="%(elapsed)s")
            )

        def update_thread():
            while not self.finished:
                if self.sub_bar is None or self.sub_bar.finished:
                    self.update()
                time.sleep(0.001)

        self.thread = threading.Thread(target=update_thread)
        self.thread.daemon = True

        if max_value is None:
            max_value = progressbar.UnknownLength

        super().__init__(
            poll_interval=0.1,
            widgets=widgets,
            fd=sys.stdout,
            max_value=max_value,
            **kwargs,
        )

    def start(self, **kwargs):
        """Start tracking process, initialize display."""

        super().start(**kwargs)

        self.finished = False
        self.thread.start()

        return self

    def finish(self, **kwargs):
        """Stop tracking process, finish display."""

        if self.sub_bar is not None and self.sub_bar.finished is False:
            self.sub_bar.finish()

        self.finished = True
        self.thread.join()

        super().finish(**kwargs)

    def step(self):
        """
        Advance the progress bar one step.
        """

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
            present="%s: %s" % (self.present, msg) if msg else self.present, **kwargs
        )

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
        super().__enter__()

        return self.start()

    def __next__(self):
        """Wraps an iterable using this progress bar."""

        try:
            assert self.start_time is not None
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

    def finish(self, **kwargs):
        """Finishing a sub-progress bar doesn't start a new line."""
        super().finish(end="\r", **kwargs)


class NullProgressBar(progressbar.NullBar):  # pylint: disable=too-many-ancestors
    """
    A progress bar that does nothing.

    Used to replace ProgressBar when we want to disable output.
    """

    def __init__(self, present="", past=None, max_value=1, **kwargs):
        super().__init__(max_value=max_value, **kwargs)

    def sub(self, *args, **kwargs):
        """
        Noop for creating a sub-progress bar.
        """
        return self

    def step(self, **kwargs):
        """
        Noop for incrementing the progress bar.
        """
