from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, TimeUpdate, SlicedCopy, SimPyFunc)
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


@Builder.register(TimeUpdate)
def time_update(op, signals, dt):
    signals[op.step] = tf.assign_add(signals[op.step], 1)
    # TODO: is there really no way to multiply an int by a float?
    signals[op.time] = tf.assign(signals[op.time],
                                 tf.to_float(signals[op.step]) * dt)

    return signals[op.step], signals[op.time]


@Builder.register(Reset)
def reset(op, signals):
    if DEBUG:
        print("reset")
        print(op)
        print("dst", signals[op.dst])
        print("val", op.value)

    # convert value to appropriate dtype
    value = np.asarray(op.value, dtype=op.dst.dtype)

    # convert value to appropriate shape (note: this can change the
    # number of elements, which we use to broadcast scalars up to full
    # matrices)
    value = np.resize(value, op.dst.shape)

    if DEBUG:
        print("val", value)

    return signals.assign_view(op.dst, tf.constant(value))


@Builder.register(Copy)
def copy(op, signals):
    return signals.assign_view(op.dst, op.src)


@Builder.register(SlicedCopy)
def sliced_copy(op, signals):
    if DEBUG:
        print("sliced_copy")
        print(op)
        print("dst", signals[op.dst])
        print("dst.base", signals[op.dst.base])
        print(op.dst_slice)
        print("src", signals[op.src])
        print(op.src_slice)

    return signals.assign_view(op.dst, op.src, dst_slice=op.dst_slice,
                               src_slice=op.src_slice, inc=op.inc)


@Builder.register(ElementwiseInc)
def elementwise_inc(op, signals):
    if DEBUG:
        print("elementwise_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    return signals.assign_view(op.Y, signals[op.A] * signals[op.X], inc=True)


@Builder.register(DotInc)
def dot_inc(op, signals):
    # note: this is matrix/vector (A) by vector (X) multiplication,
    # not matrix-matrix

    if DEBUG:
        print("dot_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    dot = tf.mul(signals[op.A], signals[op.X])
    if op.A.ndim == 2:
        dot = tf.reduce_sum(dot, axis=1)

    return signals.assign_view(op.Y, dot, inc=True)


@Builder.register(SimPyFunc)
def sim_py_func(op, signals):
    if DEBUG:
        print("sim_py_func")
        print(op)
        print("t", op.t)
        print("x", op.x)
        print("fn", op.fn)

    inputs = []
    if op.t is not None:
        inputs += [signals[op.t]]
    if op.x is not None:
        inputs += [signals[op.x]]

    if op.output is None:
        def noop_func(*args):
            op.fn(*args)
            return args

        node_outputs = tf.py_func(
            noop_func, inputs, [x.dtype for x in inputs],
            name=utils.function_name(op.fn))
    else:
        node_outputs = tf.py_func(
            utils.align_func(op.fn, op.output),
            inputs, tf.as_dtype(op.output.dtype),
            name=utils.function_name(op.fn))

        signals.assign_view(op.output, node_outputs)

    node_outputs = ([node_outputs] if isinstance(node_outputs, tf.Tensor) else
                    node_outputs)
    return node_outputs
