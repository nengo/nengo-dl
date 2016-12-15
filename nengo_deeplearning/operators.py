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
        print("val", op.value)

    # convert value to appropriate dtype
    value = np.asarray(op.value, dtype=op.dst.dtype)

    # convert value to appropriate shape (note: this can change the
    # number of elements, which we use to broadcast scalars up to full
    # matrices)
    value = np.resize(value, op.dst.shape)

    signals[op.dst] = tf.constant(value)
    return signals[op.dst]


@Builder.register(Copy)
def copy(op, signals):
    signals[op.dst] = signals[op.src]
    return signals[op.dst]


@Builder.register(SlicedCopy)
def sliced_copy(op, signals):
    if DEBUG:
        print("sliced_copy")
        print(op)
        print("dst", signals.get(op.dst, None))
        print("dst.base", signals.get(op.dst.base, None))
        print(op.dst_slice)
        print("src", signals[op.src])
        print(op.src_slice)

    # TODO: merge this and assign_view somehow?

    src = signals[op.src]
    if isinstance(op.src_slice, slice):
        src = src[op.src_slice]
    elif op.src_slice is not Ellipsis:
        # advanced indexing
        src = tf.gather(src, tf.constant(op.src_slice))

    if not op.dst.is_view and op.dst_slice is Ellipsis:
        if op.inc:
            signals[op.dst] = signals[op.dst] + src
        else:
            signals[op.dst] = src
    else:
        # TODO: make this work for multidimensional arrays?
        assert op.dst.ndim == 1

        # TODO: all this nested if/else logic could be simplified a bit
        if op.dst_slice is Ellipsis:
            # update the whole dst tensor
            if op.inc:
                signals[op.dst] = signals[op.dst] + src
            else:
                signals[op.dst] = src

            # update the appropriate slice of dst.base
            start = op.dst.elemoffset
            stride = op.dst.elemstrides[0]
            stop = op.dst.elemoffset + op.dst.size * op.dst.elemstrides[0]

            indices = tf.range(start, stop, stride)

            signals[op.dst.base] = scatter(signals[op.dst.base], indices, src,
                                           inc=op.inc)
        elif isinstance(op.dst_slice, slice):
            # update slice of dst
            indices = tf.range(op.dst_slice.start, op.dst_slice.stop,
                               op.dst_slice.step)
            signals[op.dst] = scatter(signals[op.dst], indices, src,
                                      inc=op.inc)

            if op.dst.is_view:
                # update dst.base
                start = (op.dst.elemoffset +
                         op.dst_slice.start * op.dst.elemstrides[0])
                stride = op.dst.elemstrides[0] * op.dst_slice.step
                stop = (op.dst.elemoffset +
                        op.dst_slice.stop * op.dst.elemstrides[0])

                indices = tf.range(start, stop, stride)
                signals[op.dst.base] = scatter(signals[op.dst.base], indices,
                                               src, inc=op.inc)
        else:
            # advanced indexing
            indices = np.asarray(op.dst_slice)
            signals[op.dst] = scatter(signals[op.dst], tf.constant(indices),
                                      src, inc=op.inc)

            if op.dst.is_view:
                # update dst.base
                indices *= op.dst.elemstrides[0]
                indices += op.dst.elemoffset

                signals[op.dst.base] = scatter(
                    signals[op.dst.base], tf.constant(indices), src,
                    inc=op.inc)

    return signals[op.dst]


def scatter(dst, indices, src, inc=False):
    """Mimics the interface of tf.scatter_add/update, but for Tensors
    instead of Variables."""

    # indices are expected to be shape (number of items in slice, dst.ndims)
    indices = tf.expand_dims(indices, 1)

    # expand source to target shape
    scatter_src = tf.scatter_nd(indices, src, tf.shape(dst))

    if inc:
        return dst + scatter_src
    else:
        mask = tf.scatter_nd(indices, tf.ones_like(src), tf.shape(dst))
        return tf.where(mask, scatter_src, dst)


@Builder.register(ElementwiseInc)
def elementwise_inc(op, signals):
    if DEBUG:
        print("elementwise_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    signals[op.Y] = signals[op.Y] + signals[op.A] * signals[op.X]

    return signals[op.Y]


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

    signals[op.Y] = signals[op.Y] + dot

    return signals[op.Y]


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

        signals[op.output] = node_outputs

    node_outputs = ([node_outputs] if isinstance(node_outputs, tf.Tensor) else
                    node_outputs)
    return node_outputs
