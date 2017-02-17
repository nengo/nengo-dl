from collections import defaultdict

from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, SlicedCopy, SimPyFunc)
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG
from nengo_deeplearning.builder import Builder, OpBuilder


@Builder.register(Reset)
class ResetBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.operator.Reset`
    operators."""
    def __init__(self, ops, signals):
        if DEBUG:
            print("reset")
            print([str(x) for x in ops])
            print("val", [op.value for op in ops])
            print("dst", [op.dst for op in ops])

        dtype = utils.cast_dtype(np.asarray(ops[0].value).dtype,
                                 signals.dtype).as_numpy_dtype

        # unlike other ops, Reset signals might be spread across multiple
        # bases, which we need to handle
        scatters = defaultdict(list)
        for op in ops:
            scatters[signals.sig_map[op.dst].key] += [op]
        self.scatters = []
        for group in scatters.values():
            value = np.concatenate(
                [np.resize(np.asarray(x.value).astype(dtype), x.dst.shape)
                 for x in group], axis=0)
            value = np.tile(
                value[..., None],
                tuple(1 for _ in value.shape) + (signals.minibatch_size,))
            self.scatters += [(signals.combine([x.dst for x in group]),
                               tf.constant(value))]

        if DEBUG:
            print("scatters")
            print("\n".join([str(x) for x in self.scatters]))

    def build_step(self, signals):
        for data, val in self.scatters:
            signals.scatter(data, val)


@Builder.register(SlicedCopy)
@Builder.register(Copy)
class SlicedCopyBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.operator.Copy` and
    :class:`~nengo:nengo.builder.operator.SlicedCopy` operators."""
    def __init__(self, ops, signals):
        if DEBUG:
            print("sliced_copy")
            print([str(op) for op in ops])
            print("src", [op.src for op in ops])
            print("src_slice", [getattr(op, "src_slice", None) for op in ops])
            print("dst", [op.dst for op in ops])
            print("dst_slice", [getattr(op, "dst_slice", None) for op in ops])

        srcs = []
        dsts = []
        for op in ops:
            src_slice = getattr(op, "src_slice", Ellipsis)
            dst_slice = getattr(op, "dst_slice", Ellipsis)
            srcs += [signals.sig_map[op.src][src_slice]]
            dsts += [signals.sig_map[op.dst][dst_slice]]

        self.mode = "inc" if getattr(ops[0], "inc", False) else "update"

        self.src_data = signals.combine(srcs, load_indices=False)
        self.dst_data = signals.combine(dsts)

        if not self.src_data.minibatched and self.dst_data.minibatched:
            # broadcast indices so that the un-minibatched src data gets
            # copied to each minibatch dimension in dst
            self.src_data = self.src_data.broadcast(-1, signals.minibatch_size)

        self.src_data.load_indices()

    def build_step(self, signals):
        signals.scatter(self.dst_data, signals.gather(self.src_data),
                        mode=self.mode)


@Builder.register(ElementwiseInc)
# @Builder.register(DotInc)
class ElementwiseIncBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.operator.ElementwiseInc`
    operators."""
    def __init__(self, ops, signals):
        if DEBUG:
            print("elementwise_inc"), len(ops)
            print("\n".join([str(x) for x in ops]))
            print("dst", [op.Y for op in ops])
            print("A", [op.A for op in ops])
            print("X", [op.X for op in ops])

        self.dot_inc = isinstance(ops[0], DotInc)

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        A_data = signals.combine([op.A for op in ops], load_indices=False)
        X_data = signals.combine([op.X for op in ops], load_indices=False)

        # separate data from each op along the first dimension
        self.A_data = A_data.reshape((len(ops), -1) + A_data.shape[1:])
        self.X_data = X_data.reshape((len(ops), -1) + X_data.shape[1:])

        if self.dot_inc:
            # add empty dimension to X for broadcasting across rows
            self.X_data = self.X_data.reshape((self.X_data.shape[0], 1) +
                                              self.X_data.shape[1:])
        else:
            # add empty trailing dimensions for elementwise broadcasting
            while self.A_data.ndim < self.X_data.ndim:
                self.A_data = self.A_data.reshape(self.A_data.shape + (1,))
            while self.X_data.ndim < self.A_data.ndim:
                self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

        # add broadcast dimension for minibatch, if needed
        if not self.A_data.minibatched and self.X_data.minibatched:
            self.A_data = self.A_data.reshape(self.A_data.shape + (1,))
        elif self.A_data.minibatched and not self.X_data.minibatched:
            self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

        self.A_data.load_indices()
        self.X_data.load_indices()

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        result = tf.multiply(A, X)

        if self.dot_inc:
            reduce_axis = -1 - (self.A_data.minibatched or
                                self.X_data.minibatched)
            result = tf.reduce_sum(result, axis=reduce_axis)

        signals.scatter(self.Y_data, result, mode="inc")


@Builder.register(DotInc)
class DotIncBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.operator.DotInc`
    operators."""
    def __init__(self, ops, signals):
        if DEBUG:
            print("dot_inc"), len(ops)
            print("\n".join([str(x) for x in ops]))
            print("dst", [op.Y for op in ops])
            print("A", [op.A for op in ops])
            print("X", [op.X for op in ops])

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        A_data = signals.combine([op.A for op in ops], load_indices=False)
        X_data = signals.combine([op.X for op in ops], load_indices=False)

        # separate data from each op along the first dimension
        self.A_data = A_data.reshape((len(ops), -1, A_data.shape[1]))
        self.X_data = X_data.reshape((len(ops), -1))

        # approach #2
        # if self.A_data.minibatched or not self.X_data.minibatched:
        #     self.X_data = self.X_data.reshape(self.X_data.shape[:2] + (1,) +
        #                                       self.X_data.shape[2:])
        #     if self.A_data.minibatched and not self.X_data.minibatched:
        #         self.X_data = self.X_data.broadcast(
        #             -1, signals.minibatch_size)

        # approach #3
        # TODO: what should the minibatch_size cutoff be?
        self.using_matmul = (
            signals.minibatch_size >= 32 and
            not self.A_data.minibatched and self.X_data.minibatched)
        if not self.using_matmul:
            # add empty dimension to X for broadcasting (since we'll be doing
            # it with the mul->reduce method)
            self.X_data = self.X_data.reshape((self.X_data.shape[0], 1) +
                                              self.X_data.shape[1:])

            # add empty minibatch dimension if needed
            if not self.A_data.minibatched and self.X_data.minibatched:
                self.A_data = self.A_data.reshape(self.A_data.shape + (1,))
            if self.A_data.minibatched and not self.X_data.minibatched:
                self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

        self.A_data.load_indices()
        self.X_data.load_indices()

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        # approach #1: using einsum (einsum seems to be super slow)
        # if self.A_data.minibatched and self.X_data.minibatched:
        #     dot = tf.einsum("ijkl,ikl->ijl", A, X)
        # elif self.A_data.minibatched and not self.X_data.minibatched:
        #     dot = tf.einsum("ijkl,ik->ijl", A, X)
        # elif not self.A_data.minibatched and self.X_data.minibatched:
        #     dot = tf.batch_matmul(A, X)
        # else:
        #     dot = tf.einsum("ijk,ik->ij", A, X)

        # approach #2: transpose/tile and use batch_matmul for everything
        # if not self.A_data.minibatched and self.X_data.minibatched:
        #     dot = tf.batch_matmul(A, X)
        # else:
        #     minibatched = self.A_data.minibatched or self.X_data.minibatched
        #
        #     A = tf.transpose(A, (0, 3, 1, 2)) if minibatched else A
        #     X = tf.transpose(X, (0, 3, 1, 2)) if minibatched else X
        #     dot = tf.batch_matmul(A, X)
        #
        #     if minibatched:
        #         dot = tf.transpose(dot, (0, 2, 3, 1))

        # approach #3: mix of batch_matmul and manual multiply/reduce
        if self.using_matmul:
            dot = tf.matmul(A, X)
        else:
            dot = tf.multiply(A, X)
            reduce_axis = -1 - (self.A_data.minibatched or
                                self.X_data.minibatched)
            dot = tf.reduce_sum(dot, axis=reduce_axis)

        signals.scatter(self.Y_data, dot, mode="inc")


@Builder.register(SimPyFunc)
class SimPyFuncBuilder(OpBuilder):
    """Build a group of :class:`~nengo:nengo.builder.operator.SimPyFunc`
    operators."""
    def __init__(self, ops, signals):
        if DEBUG:
            print("sim_py_func")
            print([str(op) for op in ops])
            print("t", [op.t for op in ops])
            print("x", [op.x for op in ops])
            print("fn", [op.fn for op in ops])

        self.time_input = ops[0].t is not None
        self.input_data = signals.combine([op.x for op in ops])

        if ops[0].output is not None:
            self.output_data = signals.combine([op.output for op in ops])
            self.output_dtype = self.output_data.dtype
        else:
            self.output_data = None
            self.output_dtype = signals.dtype

        def merged_func(time, inputs):
            outputs = []
            offset = 0
            for i, op in enumerate(ops):
                if op.output is None:
                    func = op.fn
                else:
                    func = utils.align_func(
                        op.output.shape, self.output_dtype)(op.fn)

                func_input = inputs[offset:offset + op.x.shape[0]]
                offset += op.x.shape[0]

                mini_out = []
                for j in range(signals.minibatch_size):
                    if op.t is None:
                        func_out = func(func_input[..., j])
                    else:
                        func_out = func(time, func_input[..., j])

                    if op.output is None:
                        # just return time as a noop (since we need to
                        # return something)
                        func_out = time
                    mini_out += [func_out]
                outputs += [np.stack(mini_out, axis=-1)]

            return np.concatenate(outputs, axis=0)

        self.merged_func = merged_func
        self.merged_func.__name__ == "_".join(
            [utils.function_name(op.fn) for op in ops])
        self.output_shape = ((len(ops),) if self.output_data is None else
                             self.output_data.shape)
        self.output_shape += (signals.minibatch_size,)

    def build_step(self, signals):
        time = signals.time if self.time_input else []
        inputs = ([] if self.input_data is None
                  else signals.gather(self.input_data))

        node_outputs = tf.py_func(
            self.merged_func, [time, inputs], self.output_dtype,
            name=self.merged_func.__name__)
        node_outputs.set_shape(self.output_shape)

        if self.output_data is not None:
            signals.scatter(self.output_data, node_outputs)

        # note: we only need to run the node for side effects, not the
        # assignment operator. if the result of the assignment is actually
        # used anywhere, then it will be run as part of the normal graph.
        return node_outputs
