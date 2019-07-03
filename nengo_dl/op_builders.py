"""
Build classes for basic Nengo operators.
"""

from collections import defaultdict
import logging
import warnings

from nengo.builder.operator import Reset, Copy, ElementwiseInc, DotInc, SimPyFunc
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_sparse_ops

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.compat import tf_compat, SparseDotInc, SparseMatrix

logger = logging.getLogger(__name__)


class ResetInc(Reset):
    """
    A version of Reset that increments the target value rather than setting it.
    """

    @property
    def dst(self):
        """Overridden to return from incs rather than sets."""
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
                [
                    np.resize(np.asarray(x.value).astype(dtype), x.dst.shape)
                    for x in group
                ],
                axis=0,
            )
            value = np.tile(
                value[..., None],
                tuple(1 for _ in value.shape) + (signals.minibatch_size,),
            )
            self.scatters += [
                (signals.combine([x.dst for x in group]), signals.constant(value))
            ]

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
        logger.debug("src_slice %s", [getattr(op, "src_slice", None) for op in ops])
        logger.debug("dst %s", [op.dst for op in ops])
        logger.debug("dst_slice %s", [getattr(op, "dst_slice", None) for op in ops])

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
        signals.scatter(self.dst_data, signals.gather(self.src_data), mode=self.mode)

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
            self.A_data = self.A_data.reshape((len(ops), -1) + self.A_data.shape[1:])
            self.X_data = self.X_data.reshape((len(ops), -1) + self.X_data.shape[1:])

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


def sparse_matmul(A_indices, A_data, A_shape, X):
    """
    Matrix multiplication between sparse matrix A and dense matrix X

    Parameters
    ----------
    A_indices : ``tf.Tensor``
        N, 2) rray of [row,col] non-zero entries
    A_data : ``tf.Tensor``
        (N,) array of data in the nonzero entries specified in ``A_indices``
    A_shape : tuple of int
        Shape of full A matrix
    X : ``tf.Tensor``
        Dense matrix being multiplied by A

    Returns
    -------
    dot : ``tf.Tensor``
        Result of matrix multiplication between A and X
    """

    must_downcast = A_data.dtype.base_dtype != tf.float32 and (
        "gpu" in A_data.device.lower()
        or (A_data.device == "" and utils.tf_gpu_installed)
    )
    if must_downcast:
        assert A_data.dtype.base_dtype == X.dtype.base_dtype
        warnings.warn(
            "Downcasting data to float32 in sparse_matmul, since "
            "only float32 is supported on the GPU."
        )
        A = tf.cast(A_data, tf.float32)
        X = tf.cast(X, tf.float32)
    else:
        A = A_data

    dot = gen_sparse_ops.sparse_tensor_dense_mat_mul(A_indices, A, A_shape, X)

    if must_downcast:
        dot = tf.cast(dot, A_data.dtype.base_dtype)

    return dot


# class DotSet(DotInc):
#     @property
#     def Y(self):
#         return self.sets[0]


@Builder.register(DotInc)
# @Builder.register(DotSet)
class DotIncBuilder(OpBuilder):
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

        # check if all the signals have the same size for the first dimension
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

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        A_data = signals.combine([op.A for op in ops])
        X_data = signals.combine([op.X for op in ops])

        if self.len_match:
            # if the first dimensions all match, then we can used the
            # (batched) matrix multiplication op

            # separate data from each op along the first dimension
            self.A_data = A_data.reshape((len(ops), -1, A_data.shape[1]))
            self.X_data = X_data.reshape((len(ops), -1))

            if self.A_data.minibatched:
                # add broadcast dimension to X
                self.X_data = self.X_data.reshape(self.X_data.shape + (1,))

                # precompute transposition indices
                self.perm = tf.constant((0, 3, 1, 2))
                self.perm_inv = tf.constant((0, 2, 3, 1))
        else:
            # if the first dimensions don't match, then we create a block
            # diagonal matrix out of all the op matrices, and then multiply
            # them using a sparse matrix multiplication

            self.A_data = A_data.reshape((-1,))
            self.X_data = X_data

            assert not self.A_data.minibatched
            assert self.X_data.minibatched and self.Y_data.minibatched

            sparse_indices = []
            corner = np.zeros(2, dtype=np.int64)
            for op in ops:
                block_shape = (op.A.shape[0], op.A.shape[1])
                idxs = np.reshape(
                    np.dstack(
                        np.meshgrid(
                            np.arange(block_shape[0]),
                            np.arange(block_shape[1]),
                            indexing="ij",
                        )
                    ),
                    (-1, 2),
                )
                idxs += corner
                corner += block_shape
                sparse_indices += [idxs]

            sparse_indices = np.concatenate(sparse_indices, axis=0)
            self.sparse_indices = signals.constant(
                sparse_indices,
                dtype=(
                    tf.int32
                    if np.all(sparse_indices < np.iinfo(np.int32).max)
                    else tf.int64
                ),
            )
            self.A_shape = tf.constant(corner, dtype=tf.int64)

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        if self.len_match:
            if self.A_data.minibatched and self.X_data.minibatched:
                # dot = tf.einsum("ijkl,ikl->ijl", A, X)

                # note: this is just a duplicate of what einsum does
                # internally; we do it manually so that we can move the
                # perm/perm_inv constants into the pre-build step
                A = tf.transpose(a=A, perm=self.perm)
                X = tf.transpose(a=X, perm=self.perm)
                dot = tf.matmul(A, X)
                dot = tf.transpose(a=dot, perm=self.perm_inv)
                dot.set_shape(self.A_data.shape[:2] + (1, signals.minibatch_size))
            elif not self.A_data.minibatched and self.X_data.minibatched:
                dot = tf.matmul(A, X)
            else:
                # note: these cases never come up (so far) in nengo, since X
                # is always minibatched. but preserving them here for
                # posterity, in case they are ever used

                # A minibatched, X not minibatched
                # dot = tf.einsum("ijkl,ik->ijl", A, X)
                # A not minibatched, X not minibatched
                # dot = tf.einsum("ijk,ik->ij", A, X)
                raise NotImplementedError
        else:
            dot = sparse_matmul(self.sparse_indices, A, self.A_shape, X)

            dot.set_shape(self.Y_data.shape + (signals.minibatch_size,))

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

        def merged_func(time, inputs):  # pragma: no cover (runs in TF)
            outputs = []
            offset = 0
            for op in ops:
                if op.output is None:
                    func = op.fn
                else:
                    func = utils.align_func(op.output.shape, self.output_dtype)(op.fn)

                func_input = inputs[offset : offset + op.x.shape[0]]
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
        self.merged_func.__name__ = "_".join([utils.function_name(op.fn) for op in ops])
        self.output_shape = (
            (len(ops),) if self.output_data is None else self.output_data.shape
        )
        self.output_shape += (signals.minibatch_size,)

    def build_step(self, signals):
        time = signals.time if self.time_input else []
        inputs = [] if self.input_data is None else signals.gather(self.input_data)

        with tf.device("/cpu:0"):
            node_outputs = tf_compat.py_func(
                self.merged_func,
                [time, inputs],
                self.output_dtype,
                name=self.merged_func.__name__,
            )
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


@Builder.register(SparseDotInc)
class SparseDotIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.SparseDotInc` operators.
    """

    def __init__(self, ops, signals, config):
        super().__init__(ops, signals, config)

        self.Y_data = signals.combine([op.Y for op in ops])

        # group all the A's and X's
        self.A_data = signals.combine([op.A for op in ops])
        self.X_data = signals.combine([op.X for op in ops])

        # the only way A would be minibatched is if it is targeted by an
        # online learning rule, which isn't supported for sparse transforms
        assert not self.A_data.minibatched
        assert self.X_data.minibatched and self.Y_data.minibatched

        # arrange the sparse matrices into a (sparse) block diagonal matrix
        # by adding an offset to each sparse matrix's indices
        sparse_indices = []
        corner = np.zeros(2, dtype=np.int64)
        for op in ops:
            if isinstance(op.A.initial_value, SparseMatrix):
                idxs = np.array(op.A.initial_value.indices)
            else:
                initial_value = op.A.initial_value.tocoo()
                idxs = np.stack((initial_value.row, initial_value.col), axis=1)

            block_shape = (op.A.shape[0], op.A.shape[1])
            idxs += corner
            corner += block_shape
            sparse_indices += [idxs]

        sparse_indices = np.concatenate(sparse_indices, axis=0)
        self.sparse_indices = signals.constant(
            sparse_indices,
            dtype=(
                tf.int32
                if np.all(sparse_indices < np.iinfo(np.int32).max)
                else tf.int64
            ),
        )
        self.A_shape = tf.constant(corner, dtype=tf.int64)

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        dot = sparse_matmul(self.sparse_indices, A, self.A_shape, X)

        dot.set_shape(self.Y_data.shape + (signals.minibatch_size,))

        signals.scatter(self.Y_data, dot, mode="inc")

    @staticmethod
    def mergeable(x, y):
        return True
