from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, TimeUpdate, SlicedCopy, SimPyFunc)
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


@Builder.register(TimeUpdate)
def time_update(op, signals, dt):
    # note: the step signal is handled as part of the state in the
    # simulation loop
    signals[op.time] = tf.cast(signals[op.step], signals.dtype) * dt


@Builder.register(Reset)
def reset(op, signals):
    if DEBUG:
        print("reset")
        print(op)
        print("val", op.value)

    # convert value to appropriate dtype
    value = np.asarray(op.value)
    value = value.astype(utils.cast_dtype(value.dtype,
                                          signals.dtype).as_numpy_dtype)

    # convert value to appropriate shape (note: this can change the
    # number of elements, which we use to broadcast scalars up to full
    # matrices)
    value = np.resize(value, op.dst.shape)

    const = tf.constant(value)
    const.zero_constant = np.all(value == 0)

    signals[op.dst] = const


@Builder.register(Copy)
def copy(op, signals):
    signals[op.dst] = signals[op.src]


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
            signals.inc(op.dst, src)
        else:
            signals[op.dst] = src
    else:
        # TODO: make this work for multidimensional arrays?
        assert op.dst.ndim == 1

        # TODO: all this nested if/else logic could be simplified a bit
        if op.dst_slice is Ellipsis:
            # update the whole dst tensor
            if op.inc:
                signals.inc(op.dst, src)
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
            indices = np.asarray(op.dst_slice, dtype=np.int32)
            signals[op.dst] = scatter(signals[op.dst], tf.constant(indices),
                                      src, inc=op.inc)

            if op.dst.is_view:
                # update dst.base
                indices *= op.dst.elemstrides[0]
                indices += op.dst.elemoffset

                signals[op.dst.base] = scatter(
                    signals[op.dst.base], tf.constant(indices), src,
                    inc=op.inc)


def scatter(dst, indices, src, inc=False):
    """Mimics the interface of tf.scatter_add/update, but for Tensors
    instead of Variables."""

    # indices are expected to be shape (number of items in slice, dst.ndims)
    indices = tf.expand_dims(indices, 1)

    # expand source to target shape
    scatter_src = tf.scatter_nd(indices, src, tf.shape(dst))

    if getattr(dst, "zero_constant", False):
        return scatter_src
    elif inc:
        return dst + scatter_src
    else:
        mask = tf.scatter_nd(indices, tf.ones_like(src), tf.shape(dst))
        return tf.where(mask, scatter_src, dst)


@Builder.register(DotInc)
@Builder.register(ElementwiseInc)
def dot_inc(op, signals):
    if DEBUG:
        print("dot_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    # elementwise product
    dot = tf.mul(signals[op.A], signals[op.X])

    # for the case of matrix-vector multiplication we do a sum along the
    # vector axis (so we're doing a matrix-vector dot product, not an
    # elementwise multiplication)
    if op.A.ndim == 2 and op.X.ndim == 1:
        dot = tf.reduce_sum(dot, axis=1)

    signals.inc(op.Y, dot)


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
            return args[0]

        node_output = tf.py_func(
            noop_func, inputs, inputs[0].dtype,
            name=utils.function_name(op.fn))
        node_output.set_shape(())
    else:
        output_dtype = utils.cast_dtype(op.output.dtype, signals.dtype)
        node_output = tf.py_func(
            utils.align_func(op.fn, op.output.shape, output_dtype),
            inputs, output_dtype, name=utils.function_name(op.fn))
        node_output.set_shape(op.output.shape)

        signals[op.output] = node_output

    return node_output
