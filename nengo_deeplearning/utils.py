import re

from nengo.exceptions import SimulationError
import numpy as np
import tensorflow as tf


def sanitize_name(name):
    """Remove illegal tensorflow name characters from string."""

    if not isinstance(name, str):
        name = str(name)

    name = name.replace(" ", "_")
    name = name.replace(":", "_")

    valid_exp = re.compile("[A-Za-z0-9_.\\-/]")

    return "".join([c for c in name if valid_exp.match(c)])


def function_name(func, sanitize=True):
    """Get the name of the callable object `func`.

    Parameters
    ----------
    func : callable
        callable object (e.g., function, callable class)
    sanitize : bool, optional
        if True, remove any illegal tensorflow name characters from name

    Returns
    -------
    str
        (sanitized) name of `func`
    """

    name = getattr(func, "__name__", func.__class__.__name__)
    if sanitize:
        name = sanitize_name(name)

    return name


def align_func(output_shape, output_dtype):
    """Decorator that ensures the output of `func` has the given shape and
    dtype."""

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
    input : `tf.Tensor`
        the value of this tensor will be printed whenever it is computed
        in the graph
    message : str
        string prepended to the value of `input`, to help with identification

    Returns
    -------
    `tf.Tensor`
        new tensor representing the print operation applied to `input`

    Notes
    -----
    This is what `tf.Print` is supposed to do, but it didn't work for me.
    """

    def print_func(x):
        print(message, str(x))
        return x

    output = tf.py_func(print_func, [input], input.dtype)
    output.set_shape(input.get_shape())

    return output


def cast_dtype(dtype, target):
    """Changes float dtypes to the target dtype, leaves others unchanged.

    Used to map all float values to a target precision.  Also casts numpy
    dtypes to tensorflow dtypes.

    Parameters
    ----------
    dtype : `tf.DType` or numpy dtype
        input dtype to be converted
    target: `tf.DType`
        floating point dtype to which all floating types should be converted

    Returns
    -------
    `tf.DType`
        input dtype, converted to `target` type if necessary
    """

    if not isinstance(dtype, tf.DType):
        dtype = tf.as_dtype(dtype)

    if dtype.is_floating:
        dtype = target

    return dtype
