from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, TimeUpdate, SlicedCopy, SimPyFunc)
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG
from nengo_deeplearning.builder import Builder, OpBuilder


# @Builder.register(TimeUpdate)
# def time_update(ops, signals, dt):
#     # there should only ever be one TimeUpdate
#     assert len(ops) == 1
#
#     # note: the step signal is handled as part of the state in the
#     # simulation loop
#     signals[ops[0].time] = tf.cast(signals[ops[0].step], signals.dtype) * dt


@Builder.register(Reset)
class ResetBuilder(OpBuilder):
    def __init__(self, ops, signals):
        if DEBUG:
            print("reset")
            print([str(x) for x in ops])
            print("val", [op.value for op in ops])
            print("dst", [op.dst for op in ops])

        dtype = utils.cast_dtype(np.asarray(ops[0].value).dtype,
                                 signals.dtype).as_numpy_dtype
        value = np.concatenate(
            [np.resize(np.asarray(op.value).astype(dtype), op.dst.shape)
             for op in ops], axis=0)

        self.dst_data = signals.combine([x.dst for x in ops])
        self.reset_val = tf.constant(value)

        if DEBUG:
            print("dst_data", self.dst_data)
            print("reset_val", self.reset_val)

    def build_step(self, signals):
        signals.scatter(self.dst_data, self.reset_val)


@Builder.register(SlicedCopy)
@Builder.register(Copy)
class SlicedCopyBuilder(OpBuilder):
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

        self.src_data = signals.combine(srcs)
        self.dst_data = signals.combine(dsts)

    def build_step(self, signals):
        signals.scatter(self.dst_data, signals.gather(self.src_data),
                        mode=self.mode)


@Builder.register(DotInc)
@Builder.register(ElementwiseInc)
class DotIncBuilder(OpBuilder):
    def __init__(self, ops, signals):
        if DEBUG:
            print("dot_inc"), len(ops)
            print("\n".join([str(x) for x in ops]))
            print("dst", [op.Y for op in ops])
            print("A", [op.A for op in ops])
            print("X", [op.X for op in ops])

        n_ops = len(ops)

        # group all the A's and X's
        # A = [signals.sig_map[op.A] for op in ops]
        # X = [signals.sig_map[op.X] for op in ops]
        A_data = signals.combine([op.A for op in ops], load_indices=False)
        X_data = signals.combine([op.X for op in ops], load_indices=False)

        # if DEBUG:
        #     print("A tensors", [str(a) for a in A])
        #     print("X tensors", [str(x) for x in X])

        self.Y_data = signals.combine([op.Y for op in ops])

        if A_data.ndim == 2 and X_data.ndim == 2:
            # vector-vector outer product
            assert all([op.A.shape[1] == 1 for op in ops])
            assert all([op.X.shape[0] == 1 for op in ops])

            # for i in range(len(ops)):
            #     A[i] = A[i].broadcast(1, X[i].shape[1])
            #     X[i] = X[i].broadcast(0, A[i].shape[0])
            #
            # self.A_data = signals.combine(A)
            # self.X_data = signals.combine(X)

            self.A_data = A_data.reshape((n_ops, -1, 1))
            self.X_data = X_data.reshape((n_ops, 1, -1))

            self.reduce = False
        elif A_data.ndim == 2 and X_data.ndim == 1:
            # matrix-vector inner product

            # for i in range(len(ops)):
            #     X[i] = X[i].broadcast(0, A[i].shape[0])
            #
            # self.A_data = signals.combine(A)
            # self.X_data = signals.combine(X)

            self.A_data = A_data.reshape((n_ops, -1, A_data.shape[1]))
            self.X_data = X_data.reshape((n_ops, 1, -1))

            self.reduce = True
        else:
            # in all other cases we're just doing elementwise multiplication
            # add empty dimensions for broadcasting

            # TODO: change it so that only evenly sized arrays can be merged
            # for dotinc, then we can use normal tensorflow broadcasting
            # instead of creating huge index arrays

            # for i in range(len(ops)):
            #     # if the first axes don't match it's because we're
            #     # multiplying a vector by a scalar (interpreted as a length 1
            #     # vector), so we repeat the scalar
            #     if A[i].shape[0] < X[i].shape[0]:
            #         # A[i] = A[i].broadcast(1, X[i].shape[0] // A[i].shape[0])
            #         assert A[i].shape[0] == 1
            #         A[i] = A[i].tile(X[i].shape[0] // A[i].shape[0])
            #     elif X[i].shape[0] < A[i].shape[0]:
            #         # X[i] = X[i].broadcast(1, A[i].shape[0] // X[i].shape[0])
            #         assert X[i].shape[0] == 1
            #         X[i] = X[i].tile(A[i].shape[0] // X[i].shape[0])
            #
            #     # add empty broadcasting dimensions for any axes > 0
            #     while A[i].ndim < X[i].ndim:
            #         A[i] = A[i].broadcast(1, X[i].shape[A[i].ndim])
            #     while X[i].ndim < A[i].ndim:
            #         X[i] = X[i].broadcast(1, A[i].shape[X[i].ndim])
            #
            #     assert A[i].shape == X[i].shape
            #
            # self.A_data = signals.combine(A)
            # self.X_data = signals.combine(X)

            self.A_data = A_data.reshape((n_ops, -1) + A_data.shape[1:])
            self.X_data = X_data.reshape((n_ops, -1) + X_data.shape[1:])

            # add empty dimensions for broadcasting
            while self.A_data.ndim < self.X_data.ndim:
                self.A_data = self.A_data.reshape(self.A_data.shape + (1,))
            while self.X_data.ndim < self.A_data.ndim:
                self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

            self.reduce = False

        self.A_data.load_indices()
        self.X_data.load_indices()

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        # if self.reduce:
        #     X = tf.reshape(X, A.get_shape())

        dot = tf.mul(A, X)

        if self.reduce:
            dot = tf.reduce_sum(dot, axis=-1)

        dot = tf.reshape(dot, self.Y_data.shape)

        signals.scatter(self.Y_data, dot, mode="inc")


@Builder.register(SimPyFunc)
class SimPyFuncBuilder(OpBuilder):
    def __init__(self, ops, signals):
        if DEBUG:
            print("sim_py_func")
            print([str(op) for op in ops])
            print("t", [op.t for op in ops])
            print("x", [op.x for op in ops])
            print("fn", [op.fn for op in ops])

        self.time_input = ops[0].t is not None
        self.input_data = (None if ops[0].x is None else
                           signals.combine([op.x for op in ops]))

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
                    func = utils.align_func(op.fn, op.output.shape,
                                            self.output_dtype)

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

        self.merged_func = merged_func
        self.merged_func.__name__ == "_".join(
            [utils.function_name(op.fn) for op in ops])
        self.output_shape = (len(ops) if self.output_data is None else
                             self.output_data.shape)

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
