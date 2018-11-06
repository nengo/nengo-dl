"""
Build classes for basic Nengo operators.
"""

from collections import defaultdict
from distutils.version import LooseVersion
import logging

from nengo.builder.operator import (
    Reset, Copy, ElementwiseInc, DotInc, SimPyFunc)
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_sparse_ops

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder

logger = logging.getLogger(__name__)


class ResetInc(Reset):
    """
    A version of Reset that increments the target value rather than setting it.
    """
    @property
    def dst(self):
        return self.incs[0]


@Builder.register(Reset)
@Builder.register(ResetInc)
class ResetBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.Reset` operators.
    """

    def __init__(self, ops, signals, config):
        super(ResetBuilder, self).__init__(ops, signals, config)

        logger.debug("val %s", [op.value for op in ops])
        logger.debug("dst %s", [op.dst for op in ops])

        self.mode = "inc" if type(ops[0]) == ResetInc else "update"

        dtype = np.asarray(ops[0].value).dtype
        if np.issubdtype(dtype, np.floating):
            dtype = signals.dtype.as_numpy_dtype

        # unlike other ops, Reset signals might be spread across multiple
        # bases, which we need to handle
        scatters = defaultdict(list)
        for op in ops:
            scatters[signals[op.dst].key] += [op]
        self.scatters = []
        for group in scatters.values():
            value = np.concatenate(
                [np.resize(np.asarray(x.value).astype(dtype), x.dst.shape)
                 for x in group], axis=0)
            value = np.tile(
                value[..., None],
                tuple(1 for _ in value.shape) + (signals.minibatch_size,))
            self.scatters += [(signals.combine([x.dst for x in group]),
                               signals.constant(value))]

        logger.debug("scatters")
        logger.debug("\n".join([str(x) for x in self.scatters]))

    def build_step(self, signals):
        for data, val in self.scatters:
            signals.scatter(data, val, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        return True


@Builder.register(Copy)
class CopyBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.Copy` operators.
    """

    def __init__(self, ops, signals, config):
        super(CopyBuilder, self).__init__(ops, signals, config)

        logger.debug("src %s", [op.src for op in ops])
        logger.debug("src_slice %s", [getattr(op, "src_slice", None)
                                      for op in ops])
        logger.debug("dst %s", [op.dst for op in ops])
        logger.debug("dst_slice %s", [getattr(op, "dst_slice", None)
                                      for op in ops])

        srcs = []
        dsts = []
        for op in ops:
            srcs += [signals[op.src][op.src_slice]]
            dsts += [signals[op.dst][op.dst_slice]]

        self.mode = "inc" if ops[0].inc else "update"

        self.src_data = signals.combine(srcs)
        self.dst_data = signals.combine(dsts)

        if not self.src_data.minibatched and self.dst_data.minibatched:
            # broadcast indices so that the un-minibatched src data gets
            # copied to each minibatch dimension in dst
            self.src_data = self.src_data.broadcast(-1, signals.minibatch_size)

    def build_step(self, signals):
        signals.scatter(self.dst_data, signals.gather(self.src_data),
                        mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        return True


# class ElementwiseSet(ElementwiseInc):
#     @property
#     def Y(self):
#         return self.sets[0]


@Builder.register(ElementwiseInc)
# @Builder.register(ElementwiseSet)
class ElementwiseIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.ElementwiseInc` operators.
    """

    def __init__(self, ops, signals, config):
        super(ElementwiseIncBuilder, self).__init__(ops, signals, config)

        logger.debug("dst %s", [op.Y for op in ops])
        logger.debug("A %s", [op.A for op in ops])
        logger.debug("X %s", [op.X for op in ops])

        self.mode = "inc" if type(ops[0]) == ElementwiseInc else "update"

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        self.A_data = signals.combine([op.A for op in ops])
        self.X_data = signals.combine([op.X for op in ops])

        # separate data from each op along the first dimension
        if self.A_data.shape[0] != self.X_data.shape[0]:
            self.A_data = self.A_data.reshape(
                (len(ops), -1) + self.A_data.shape[1:])
            self.X_data = self.X_data.reshape(
                (len(ops), -1) + self.X_data.shape[1:])

        # add empty trailing dimensions for elementwise broadcasting
        while self.A_data.ndim < self.X_data.ndim:
            self.A_data = self.A_data.reshape(self.A_data.shape + (1,))

        # add broadcast dimension for minibatch, if needed
        if not self.A_data.minibatched and self.X_data.minibatched:
            self.A_data = self.A_data.reshape(self.A_data.shape + (1,))

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        result = tf.multiply(A, X)

        signals.scatter(self.Y_data, result, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        # for these operations we enforce that the first dimensions
        # match (we know all the other dimensions match due to the generic
        # checks).
        # this allows us to stack all the arguments into continuous array
        # blocks, allowing for more efficient multiplication (mainly
        # because it allows us to take advantage of broadcasting)
        for s0, s1 in zip(x.all_signals, y.all_signals):
            shape0 = s0.shape[0] if s0.shape != () else 1
            shape1 = s1.shape[0] if s1.shape != () else 1
            if shape0 != shape1:
                return False

        return True


# class DotSet(DotInc):
#     @property
#     def Y(self):
#         return self.sets[0]


# @Builder.register(DotInc)
class DotIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.DotInc` operators.
    """

    def __init__(self, ops, signals, config):
        super(DotIncBuilder, self).__init__(ops, signals, config)

        logger.debug("dst %s", [op.Y for op in ops])
        logger.debug("A %s", [op.A for op in ops])
        logger.debug("X %s", [op.X for op in ops])

        self.mode = "inc" if type(ops[0]) == DotInc else "update"

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        A_data = signals.combine([op.A for op in ops])
        X_data = signals.combine([op.X for op in ops])

        # separate data from each op along the first dimension
        self.A_data = A_data.reshape((len(ops), -1, A_data.shape[1]))
        self.X_data = X_data.reshape((len(ops), -1))

        # approach #1: transpose (if necessary) and use matmul for everything
        if self.A_data.minibatched:
            # add broadcast dimension to X
            self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

            # precompute transposition indices
            self.perm = tf.constant((0, 3, 1, 2))
            self.perm_inv = tf.constant((0, 2, 3, 1))

        # approach #2: mix of matmul and manual multiply/reduce
        # self.using_matmul = (
        #     signals.minibatch_size >= 32 and
        #     not self.A_data.minibatched and self.X_data.minibatched)
        # if not self.using_matmul:
        #     # add empty dimension to X for broadcasting (since we'll be doing
        #     # it with the mul->reduce method)
        #     self.X_data = self.X_data.reshape((self.X_data.shape[0], 1) +
        #                                       self.X_data.shape[1:])
        #
        #     # add empty minibatch dimension if needed
        #     if not self.A_data.minibatched and self.X_data.minibatched:
        #         self.A_data = self.A_data.reshape(self.A_data.shape + (1,))

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        # approach #1
        if self.A_data.minibatched and self.X_data.minibatched:
            # dot = tf.einsum("ijkl,ikl->ijl", A, X)

            # note: this is just a duplicate of what einsum does internally;
            # we do it manually so that we can move the perm/perm_inv constants
            # into the pre-build step
            A = tf.transpose(A, self.perm)
            X = tf.transpose(X, self.perm)
            dot = tf.matmul(A, X)
            dot = tf.transpose(dot, self.perm_inv)
            dot.set_shape(self.A_data.shape[:2] + (1, signals.minibatch_size))
        elif not self.A_data.minibatched and self.X_data.minibatched:
            dot = tf.matmul(A, X)
        else:
            # note: these cases never come up (so far) in nengo, since X
            # is always minibatched. but preserving them here for posterity,
            # in case they are ever used

            # A minibatched, X not minibatched
            # dot = tf.einsum("ijkl,ik->ijl", A, X)
            # A not minibatched, X not minibatched
            # dot = tf.einsum("ijk,ik->ij", A, X)
            raise NotImplementedError

        # approach #2
        # if self.using_matmul:
        #     dot = tf.matmul(A, X)
        # else:
        #     dot = tf.multiply(A, X)
        #     reduce_axis = -1 - (self.A_data.minibatched or
        #                         self.X_data.minibatched)
        #     dot = tf.reduce_sum(dot, axis=reduce_axis)

        signals.scatter(self.Y_data, dot, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        # if the matrix (A) is minibatched, then the first dimensions need
        # to match up (to allow us to transpose the dimensions)
        if x.A.minibatched:
            for s0, s1 in zip(x.all_signals, y.all_signals):
                shape0 = s0.shape[0] if s0.shape != () else 1
                shape1 = s1.shape[0] if s1.shape != () else 1
                if shape0 != shape1:
                    return False

        return True


@Builder.register(DotInc)
# @Builder.register(DotSet)
class SparseDotIncBuilder(DotIncBuilder):
    """
    Build a group of `~nengo.builder.operator.DotInc` operators.
    """

    def __init__(self, ops, signals, config):
        # note: bypassing the DotIncBuilder init
        # pylint: disable=bad-super-call
        super(DotIncBuilder, self).__init__(ops, signals, config)

        logger.debug("dst %s", [op.Y for op in ops])
        logger.debug("A %s", [op.A for op in ops])
        logger.debug("X %s", [op.X for op in ops])

        self.mode = "inc" if type(ops[0]) == DotInc else "update"

        self.len_match = True
        for i, s0 in enumerate(ops[0].all_signals):
            shape0 = s0.shape[0] if s0.shape != () else 1

            for op in ops:
                s1 = op.all_signals[i]
                shape1 = s1.shape[0] if s1.shape != () else 1
                if shape0 != shape1:
                    self.len_match = False
                    break

            if not self.len_match:
                break

        if self.len_match:
            super(SparseDotIncBuilder, self).__init__(ops, signals, config)
        else:
            self.Y_data = signals.combine([op.Y for op in ops])

            # group all the A's and X's
            self.A_data = signals.combine([op.A for op in ops])
            self.A_data = self.A_data.reshape((-1,))
            self.X_data = signals.combine([op.X for op in ops])

            assert not self.A_data.minibatched
            assert self.X_data.minibatched and self.Y_data.minibatched

            sparse_indices = []
            corner = np.zeros(2, dtype=np.int64)
            for op in ops:
                block_shape = (op.A.shape[0], op.A.shape[1])
                idxs = np.reshape(np.dstack(np.meshgrid(
                    np.arange(block_shape[0]), np.arange(block_shape[1]),
                    indexing="ij")), (-1, 2))
                idxs += corner
                corner += block_shape
                sparse_indices += [idxs]

            sparse_indices = np.concatenate(sparse_indices, axis=0)
            self.sparse_indices = signals.constant(sparse_indices, dtype=(
                tf.int32 if np.all(sparse_indices < np.iinfo(np.int32).max)
                else tf.int64))
            self.A_shape = tf.constant(corner, dtype=tf.int64)

    def build_step(self, signals):
        if self.len_match:
            super(SparseDotIncBuilder, self).build_step(signals)
            return

        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        assert A.get_shape()[0] == self.sparse_indices.get_shape()[0]

        # approach 1: using sparse_tensor_dense_matmul
        if LooseVersion(tf.__version__) < LooseVersion("1.7.0"):
            mat_mul = gen_sparse_ops._sparse_tensor_dense_mat_mul
        else:
            mat_mul = gen_sparse_ops.sparse_tensor_dense_mat_mul
        dot = mat_mul(self.sparse_indices, A, self.A_shape, X)

        # approach 2: matmul(a_is_sparse)
        # sparse_A = tf.scatter_nd(self.sparse_indices, A, self.A_shape)
        # dot = tf.matmul(sparse_A, X, a_is_sparse=self.is_sparse)

        dot.set_shape(self.Y_data.shape + (signals.minibatch_size,))

        signals.scatter(self.Y_data, dot, mode=self.mode)


@Builder.register(SimPyFunc)
class SimPyFuncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.SimPyFunc` operators.
    """

    def __init__(self, ops, signals, config):
        super(SimPyFuncBuilder, self).__init__(ops, signals, config)

        logger.debug("t %s", [op.t for op in ops])
        logger.debug("x %s", [op.x for op in ops])
        logger.debug("fn %s", [op.fn for op in ops])

        self.time_input = ops[0].t is not None
        self.input_data = signals.combine([op.x for op in ops])

        if ops[0].output is not None:
            self.output_data = signals.combine([op.output for op in ops])
            self.output_dtype = self.output_data.dtype
        else:
            self.output_data = None
            self.output_dtype = signals.dtype

        def merged_func(time, inputs):  # pragma: no cover
            outputs = []
            offset = 0
            for op in ops:
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
        self.merged_func.__name__ = "_".join(
            [utils.function_name(op.fn) for op in ops])
        self.output_shape = ((len(ops),) if self.output_data is None else
                             self.output_data.shape)
        self.output_shape += (signals.minibatch_size,)

    def build_step(self, signals):
        time = signals.time if self.time_input else []
        inputs = ([] if self.input_data is None
                  else signals.gather(self.input_data))

        with tf.device("/cpu:0"):
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

    @staticmethod
    def mergeable(x, y):
        # for these we need to make a special check that the functions
        # all do/do not get time as input, otherwise we could end
        # up confusing a node that only gets a scalar float input with
        # a node that only gets time as input
        return x.t == y.t
