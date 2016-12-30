from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, TimeUpdate, SlicedCopy, SimPyFunc)
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


# @Builder.register(TimeUpdate)
# def time_update(ops, signals, dt):
#     # there should only ever be one TimeUpdate
#     assert len(ops) == 1
#
#     # note: the step signal is handled as part of the state in the
#     # simulation loop
#     signals[ops[0].time] = tf.cast(signals[ops[0].step], signals.dtype) * dt


@Builder.register(Reset)
def reset(ops, signals):
    if DEBUG:
        print("reset")
        print([str(x) for x in ops])
        print("val", [op.value for op in ops])
        print("dst", [op.dst for op in ops])

    # # convert value to appropriate dtype
    # value = np.asarray(op.value)
    # value = value.astype(utils.cast_dtype(value.dtype,
    #                                       signals.dtype).as_numpy_dtype)
    #
    # # convert value to appropriate shape (note: this can change the
    # # number of elements, which we use to broadcast scalars up to full
    # # matrices)
    # value = np.resize(value, op.dst.shape)
    #
    # const = tf.constant(value)
    # const.zero_constant = np.all(value == 0)
    #
    # signals[op.dst] = const

    dtype = utils.cast_dtype(np.asarray(ops[0].value).dtype,
                             signals.dtype).as_numpy_dtype
    value = np.concatenate(
        [np.resize(np.asarray(op.value).astype(dtype), op.dst.shape)
         for op in ops], axis=0)

    const = tf.constant(value)
    signals[[x.dst for x in ops]] = const


# @Builder.register(Copy)
# def copy(op, signals):
#     signals[op.dst] = signals[op.src]


@Builder.register(SlicedCopy)
@Builder.register(Copy)
def sliced_copy(ops, signals):
    if DEBUG:
        print("sliced_copy")
        print([str(op) for op in ops])
        print("src", [op.src for op in ops])
        print("src_slice", [getattr(op, "src_slice", None) for op in ops])
        print("dst", [op.dst for op in ops])
        print("dst_slice", [getattr(op, "dst_slice", None) for op in ops])

    # # TODO: merge this and assign_view somehow?
    #
    # src = signals[op.src]
    # if isinstance(op.src_slice, slice):
    #     src = src[op.src_slice]
    # elif op.src_slice is not Ellipsis:
    #     # advanced indexing
    #     src = tf.gather(src, tf.constant(op.src_slice))
    #
    # if not op.dst.is_view and op.dst_slice is Ellipsis:
    #     if op.inc:
    #         signals.inc(op.dst, src)
    #     else:
    #         signals[op.dst] = src
    # else:
    #     # TODO: make this work for multidimensional arrays?
    #     assert op.dst.ndim == 1
    #
    #     # TODO: all this nested if/else logic could be simplified a bit
    #     if op.dst_slice is Ellipsis:
    #         # update the whole dst tensor
    #         if op.inc:
    #             signals.inc(op.dst, src)
    #         else:
    #             signals[op.dst] = src
    #
    #         # update the appropriate slice of dst.base
    #         start = op.dst.elemoffset
    #         stride = op.dst.elemstrides[0]
    #         stop = op.dst.elemoffset + op.dst.size * op.dst.elemstrides[0]
    #
    #         indices = tf.range(start, stop, stride)
    #
    #         signals[op.dst.base] = scatter(signals[op.dst.base], indices, src,
    #                                        inc=op.inc)
    #     elif isinstance(op.dst_slice, slice):
    #         # update slice of dst
    #         indices = tf.range(op.dst_slice.start, op.dst_slice.stop,
    #                            op.dst_slice.step)
    #         signals[op.dst] = scatter(signals[op.dst], indices, src,
    #                                   inc=op.inc)
    #
    #         if op.dst.is_view:
    #             # update dst.base
    #             start = (op.dst.elemoffset +
    #                      op.dst_slice.start * op.dst.elemstrides[0])
    #             stride = op.dst.elemstrides[0] * op.dst_slice.step
    #             stop = (op.dst.elemoffset +
    #                     op.dst_slice.stop * op.dst.elemstrides[0])
    #
    #             indices = tf.range(start, stop, stride)
    #             signals[op.dst.base] = scatter(signals[op.dst.base], indices,
    #                                            src, inc=op.inc)
    #     else:
    #         # advanced indexing
    #         indices = np.asarray(op.dst_slice, dtype=np.int32)
    #         signals[op.dst] = scatter(signals[op.dst], tf.constant(indices),
    #                                   src, inc=op.inc)
    #
    #         if op.dst.is_view:
    #             # update dst.base
    #             indices *= op.dst.elemstrides[0]
    #             indices += op.dst.elemoffset
    #
    #             signals[op.dst.base] = scatter(
    #                 signals[op.dst.base], tf.constant(indices), src,
    #                 inc=op.inc)

    srcs = []
    dsts = []
    for op in ops:
        src_slice = getattr(op, "src_slice", Ellipsis)
        dst_slice = getattr(op, "dst_slice", Ellipsis)
        srcs += [signals.sig_map[op.src][src_slice]]
        dsts += [signals.sig_map[op.dst][dst_slice]]

    inc = getattr(ops[0], "inc", False)

    signals.scatter(dsts, signals[srcs], mode="inc" if inc else "update")


# def scatter(dst, indices, src, inc=False):
#     """Mimics the interface of tf.scatter_add/update, but for Tensors
#     instead of Variables."""
#
#     # indices are expected to be shape (number of items in slice, dst.ndims)
#     indices = tf.expand_dims(indices, 1)
#
#     # expand source to target shape
#     scatter_src = tf.scatter_nd(indices, src, tf.shape(dst))
#
#     if getattr(dst, "zero_constant", False):
#         return scatter_src
#     elif inc:
#         return dst + scatter_src
#     else:
#         mask = tf.scatter_nd(indices, tf.ones_like(src), tf.shape(dst))
#         return tf.where(mask, scatter_src, dst)


@Builder.register(DotInc)
@Builder.register(ElementwiseInc)
def dot_inc(ops, signals):
    # if DEBUG:
    #     print("dot_inc")
    #     print(ops)
    #     print("dst", signals[ops.Y])
    #     print("A", signals[ops.A])
    #     print("X", signals[ops.X])
    #
    # # elementwise product
    # dot = tf.mul(signals[ops.A], signals[ops.X])
    #
    # # for the case of matrix-vector multiplication we do a sum along the
    # # vector axis (so we're doing a matrix-vector dot product, not an
    # # elementwise multiplication)
    # if ops.A.ndim == 2 and ops.X.ndim == 1:
    #     dot = tf.reduce_sum(dot, axis=1)
    #
    # signals.inc(ops.Y, dot)

    if DEBUG:
        print("dot_inc"), len(ops)
        print("\n".join([str(x) for x in ops]))
        print("dst", [op.Y for op in ops])
        print("A", [op.A for op in ops])
        print("X", [op.X for op in ops])

    # group all the A's and X's
    # A = signals.combine([op.A for op in ops])
    # X = signals.combine([op.X for op in ops])
    A = [signals.sig_map[op.A] for op in ops]
    X = [signals.sig_map[op.X] for op in ops]

    if DEBUG:
        print("A tensors", [str(a) for a in A])
        print("X tensors", [str(x) for x in X])

    # A = signals[[op.A for op in ops]]
    # X = signals[[op.X for op in ops]]
    # A = tf.reshape(A, (len(ops), -1) + ops[0].A.shape[1:])
    # X = tf.reshape(X, (len(ops), -1) + ops[0].X.shape[1:])
    Y = [op.Y for op in ops]

    if A[0].ndim == 2 and X[0].ndim == 2:
        # vector-vector outer product
        assert all([a.shape[1] == 1 for a in A])
        assert all([x.shape[0] == 1 for x in X])

        for i in range(len(ops)):
            A[i] = A[i].broadcast(1, X[i].shape[1])
            X[i] = X[i].broadcast(0, A[i].shape[0])

        A = signals[A]
        X = signals[X]

        dot = tf.mul(A, X)

    elif A[0].ndim == 2 and X[0].ndim == 1:
        # matrix-vector inner product

        # A = tf.reshape(signals[A], (len(ops), -1) + A[0].shape[1:])
        # X = tf.reshape(signals[X], (len(ops), -1))
        # X = tf.expand_dims(X, axis=1)
        for i in range(len(ops)):
            X[i] = X[i].broadcast(0, A[i].shape[0])

        A = signals[A]
        X = tf.reshape(signals[X], A.get_shape())

        dot = tf.mul(A, X)
        dot = tf.reduce_sum(dot, axis=-1)

        dot = tf.reshape(dot, (-1,))
    else:
        # in all other cases we're just doing elementwise multiplication
        # add empty dimensions for broadcasting
        # while A.get_shape().ndims < X.get_shape().ndims:
        #     A = tf.expand_dims(A, axis=-1)
        # while X.get_shape().ndims < A.get_shape().ndims:
        #     X = tf.expand_dims(X, axis=-1)
        #
        # dot = tf.mul(A, X)
        # dot = tf.reshape(dot, (-1,) + ops[0].A.shape[1:])

        # TODO: check if this indexed-based broadcasting is faster than
        # the reshape approach
        for i in range(len(ops)):
            # if the first axes don't match it's because we're
            # multiplying a vector by a scalar (interpreted as a length 1
            # vector), so we repeat the scalar
            if A[i].shape[0] < X[i].shape[0]:
                # A[i] = A[i].broadcast(1, X[i].shape[0] // A[i].shape[0])
                assert A[i].shape[0] == 1
                A[i] = A[i].tile(X[i].shape[0] // A[i].shape[0])
            elif X[i].shape[0] < A[i].shape[0]:
                # X[i] = X[i].broadcast(1, A[i].shape[0] // X[i].shape[0])
                assert X[i].shape[0] == 1
                X[i] = X[i].tile(A[i].shape[0] // X[i].shape[0])

            # add empty broadcasting dimensions for any axes > 0
            while A[i].ndim < X[i].ndim:
                A[i] = A[i].broadcast(1, X[i].shape[A[i].ndim])
            while X[i].ndim < A[i].ndim:
                X[i] = X[i].broadcast(1, A[i].shape[X[i].ndim])

            assert A[i].shape == X[i].shape

        A = signals[A]
        X = signals[X]
        dot = tf.mul(A, X)

    signals.scatter(Y, dot, mode="inc")


# def gather_inputs(tensors):
#     # we want to avoid slicing and then repacking inputs. so this function
#     # tries to find inputs that are slices of a common base, and then gets
#     # a single larger slice on that base
#
#     tensors = set(tensors)
#     slices = set([x for x in tensors if x.op.type == "StridedSlice" or
#                   x.op.type == "Gather"])
#
#     # remove slices from tensor list
#     tensors.difference_update(slices)
#
#     groups = []
#     while len(slices) > 0:
#         group = set(slices[:1])
#
#         # find all the slices with the same base
#         group = set(
#             [x for x in slices if x.op.inputs[0] is slices[0].op.inputs[0]])
#
#         # generate a combined set of indices
#         for x in group:
#             if x.op.type == "StridedSlice":
#                 indices = tf.range(*x.op.inputs[1:])


@Builder.register(SimPyFunc)
def sim_py_func(ops, signals):
    if DEBUG:
        print("sim_py_func")
        print([str(op) for op in ops])
        print("t", [op.t for op in ops])
        print("x", [op.x for op in ops])
        print("fn", [op.fn for op in ops])

    # inputs = []
    # if op.t is not None:
    #     inputs += [signals.time]
    # if op.x is not None:
    #     inputs += [signals[op.x]]

    # if ops[0].t is None:
    #     time = []
    # else:
    #     time = [signals.time]
    time = signals.time if ops[0].t is not None else []

    # if ops[0].x is None:
    #     inputs = []
    # else:
    #     # input = signals[[op.x for op in ops]]
    #     # input = [tf.reshape(input, (len(ops), -1) + ops[0].x.shape[1:])]
    #     # inputs = [signals[op.x] for op in ops]
    #     inputs = [signals[[op.x for op in ops]]]
    inputs = signals[[op.x for op in ops]] if ops[0].x is not None else []

    if ops[0].output is not None:
        comb_output = signals.combine([op.output for op in ops])
        output_dtype = comb_output.dtype
    else:
        comb_output = None
        output_dtype = signals.dtype

    def merged_func(time, inputs):
        outputs = []
        offset = 0
        for i, op in enumerate(ops):
            if op.output is None:
                func = op.fn
            else:
                func = utils.align_func(op.fn, op.output.shape, output_dtype)

            if op.x is None:
                output = func(time)
            else:
                func_input = inputs[offset:offset + op.x.shape[0]]
                offset += op.x.shape[0]
                if op.t is None:
                    output = func(func_input)
                else:
                    output = func(time, func_input)

            if op.output is None:
                # just return time as a noop (since we need to return
                # something)
                output = [time]

            outputs += [output]
        return np.concatenate(outputs, axis=0)

    # if op.output is None:
    #     node_output = tf.py_func(
    #         noop_func, inputs, inputs[0].dtype,
    #         name=utils.function_name(op.fn))
    #     node_output.set_shape(())
    # else:
    #     output_dtype = utils.cast_dtype(op.output.dtype, signals.dtype)
    #     node_output = tf.py_func(
    #         utils.align_func(op.fn, op.output.shape, output_dtype),
    #         inputs, output_dtype, name=utils.function_name(op.fn))
    #     node_output.set_shape(op.output.shape)
    #
    #     signals[op.output] = node_output

    node_outputs = tf.py_func(
        merged_func, [time, inputs], output_dtype,
        name="_".join([utils.function_name(op.fn) for op in ops]))

    if comb_output is not None:
        node_outputs.set_shape(comb_output.shape)
        signals[comb_output] = node_outputs
    else:
        node_outputs.set_shape(len(ops))

    # note: we only need to run the node for side effects, not the assignment
    # operator. if the result of the assignment is actually used anywhere,
    # then it will be run as part of the normal graph.
    return node_outputs
