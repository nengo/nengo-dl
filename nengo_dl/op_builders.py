"""
Build classes for basic Nengo operators.
"""

import logging
import warnings
from collections import defaultdict

import numpy as np
import tensorflow as tf
from nengo.builder.operator import (
    Copy,
    DotInc,
    ElementwiseInc,
    Reset,
    SimPyFunc,
    TimeUpdate,
)
from nengo.builder.transforms import SparseDotInc
from nengo.transforms import SparseMatrix
from tensorflow.python.ops import gen_sparse_ops

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.compat import SimProbe

logger = logging.getLogger(__name__)


class ResetInc(Reset):
    """
    A version of `~nengo.builder.operator.Reset` that increments the target value
    rather than overwriting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def dst(self):
        """dst is stored in ``incs`` rather than ``sets``."""
        return self.incs[0]


class ElementwiseSet(ElementwiseInc):
    """
    A version of `~nengo.builder.operator.ElementwiseInc` that overwrites the target
    rather than incrementing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def Y(self):
        """Y is stored in ``sets`` rather than ``incs``."""
        return self.sets[0]


class DotSet(DotInc):
    """
    A version of `~nengo.builder.operator.DotInc` that overwrites the target rather
    than incrementing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def Y(self):
        """Y is stored in ``sets`` rather than ``incs``."""
        return self.sets[0]


class SparseDotSet(SparseDotInc):
    """
    A version of `~nengo.builder.operator.SparseDotInc` that overwrites the target
    rather than incrementing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def Y(self):
        """Y is stored in ``sets`` rather than ``incs``."""
        return self.sets[0]


@Builder.register(Reset)
@Builder.register(ResetInc)
class ResetBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.Reset` operators.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        logger.debug("val %s", [op.value for op in self.ops])
        logger.debug("dst %s", [op.dst for op in self.ops])

        self.mode = "inc" if type(self.ops[0]) == ResetInc else "update"

        dtype = np.asarray(self.ops[0].value).dtype
        if np.issubdtype(dtype, np.floating):
            dtype = signals.dtype.as_numpy_dtype

        # Reset signals might be spread across multiple bases, so group them
        # by the ones that do share a base
        scatters = defaultdict(list)
        for op in self.ops:
            scatters[signals[op.dst].key].append(op)
        self.scatters = []
        for group in scatters.values():
            value = np.concatenate(
                [
                    np.broadcast_to(
                        np.asarray(x.value, dtype=dtype),
                        (signals.minibatch_size,) + x.dst.shape,
                    )
                    for x in group
                ],
                axis=1,
            )
            self.scatters.append(
                (
                    signals.combine([x.dst for x in group]),
                    tf.constant(value, dtype=dtype),
                )
            )

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

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        logger.debug("src %s", [op.src for op in self.ops])
        logger.debug(
            "src_slice %s", [getattr(op, "src_slice", None) for op in self.ops]
        )
        logger.debug("dst %s", [op.dst for op in self.ops])
        logger.debug(
            "dst_slice %s", [getattr(op, "dst_slice", None) for op in self.ops]
        )

        self.src_data = signals.combine(
            [signals[op.src][op.src_slice] for op in self.ops]
        )
        self.dst_data = signals.combine(
            [signals[op.dst][op.dst_slice] for op in self.ops]
        )

        self.mode = "inc" if self.ops[0].inc else "update"

    def build_step(self, signals):
        src = signals.gather(self.src_data)
        if not self.src_data.minibatched and self.dst_data.minibatched:
            src = tf.broadcast_to(src, self.dst_data.full_shape)
        signals.scatter(self.dst_data, src, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        return True


@Builder.register(ElementwiseInc)
@Builder.register(ElementwiseSet)
class ElementwiseIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.ElementwiseInc` operators.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        logger.debug("dst %s", [op.Y for op in self.ops])
        logger.debug("A %s", [op.A for op in self.ops])
        logger.debug("X %s", [op.X for op in self.ops])

        self.mode = "inc" if type(self.ops[0]) == ElementwiseInc else "update"

        self.Y_data = signals.combine([op.Y for op in self.ops])

        # group all the A's and X's
        self.A_data = signals.combine([op.A for op in self.ops])
        self.X_data = signals.combine([op.X for op in self.ops])

        # separate data from each op along the first dimension
        # (we only need to do this if they don't have the same length already)
        if self.A_data.shape[0] != self.X_data.shape[0]:
            self.A_data = self.A_data.reshape(
                (len(self.ops), -1) + self.A_data.shape[1:]
            )
            self.X_data = self.X_data.reshape(
                (len(self.ops), -1) + self.X_data.shape[1:]
            )

        # add empty trailing dimensions for elementwise broadcasting
        while self.A_data.ndim < self.X_data.ndim:
            self.A_data = self.A_data.reshape(self.A_data.shape + (1,))

        # add broadcast dimension for minibatch, if needed
        if not self.A_data.minibatched and self.X_data.minibatched:
            self.A_data = self.A_data.reshape((1,) + self.A_data.shape)

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


def sparse_matmul(A_indices, A_data, A_shape, X, transpose_x=False):
    """
    Matrix multiplication between sparse matrix A and dense matrix X

    Parameters
    ----------
    A_indices : ``tf.Tensor``
        (N, 2) array of [row,col] non-zero entries
    A_data : ``tf.Tensor``
        (N,) array of data in the nonzero entries specified in ``A_indices``
    A_shape : tuple of int
        Shape of full A matrix
    X : ``tf.Tensor``
        Dense matrix being multiplied by A
    transpose_x : bool
        Transpose X before multiply

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

    dot = gen_sparse_ops.sparse_tensor_dense_mat_mul(
        A_indices, A, A_shape, X, adjoint_b=transpose_x
    )

    if must_downcast:
        dot = tf.cast(dot, A_data.dtype.base_dtype)

    return dot


@Builder.register(DotInc)
@Builder.register(DotSet)
class DotIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.DotInc` operators.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        logger.debug("dst %s", [op.Y for op in self.ops])
        logger.debug("A %s", [op.A for op in self.ops])
        logger.debug("X %s", [op.X for op in self.ops])

        self.mode = "inc" if type(self.ops[0]) == DotInc else "update"

        self.Y_data = signals.combine([op.Y for op in self.ops])

        # group all the A's and X's
        A_data = signals.combine([op.A for op in self.ops])
        X_data = signals.combine([op.X for op in self.ops])

        # separate data from each op along the first dimension
        self.A_data = A_data.reshape((len(self.ops), -1, A_data.shape[1]))
        self.X_data = X_data.reshape((len(self.ops), -1))

        if self.A_data.minibatched:
            # change X to matrix
            self.X_data = self.X_data.reshape(self.X_data.shape + (1,))
        else:
            # precompute transposition permutation
            self.perm = tf.constant((1, 2, 0))
            self.perm_inv = tf.constant((2, 0, 1))

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        if self.A_data.minibatched and self.X_data.minibatched:
            # (batch, n_ops, a0, a1) x (batch, n_ops, a1, 1)
            dot = tf.matmul(A, X)
        elif not self.A_data.minibatched and self.X_data.minibatched:
            # (n_ops, a0, a1) x (batch, n_ops, a1)
            # -> (n_ops, a0, a1) x (n_ops, a1, batch)
            dot = tf.matmul(A, tf.transpose(X, perm=self.perm))

            # transpose back to (batch, n_ops, a0)
            dot = tf.transpose(dot, perm=self.perm_inv)

            # for some reason the transposing causes TensorFlow to lose track of
            # the shape (only when the `perm` constants are outside the loop)
            dot.set_shape((signals.minibatch_size,) + self.A_data.shape[:2])
        else:
            raise NotImplementedError

        signals.scatter(self.Y_data, dot, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        # the first dimensions need to match up (to allow us to separate them by op)
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

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        logger.debug("t %s", [op.t for op in self.ops])
        logger.debug("x %s", [op.x for op in self.ops])
        logger.debug("fn %s", [op.fn for op in self.ops])

        self.time_data = (
            None if self.ops[0].t is None else signals[self.ops[0].t].reshape(())
        )
        self.input_data = signals.combine([op.x for op in self.ops])

        if self.ops[0].output is not None:
            self.output_data = signals.combine([op.output for op in self.ops])
            self.output_dtype = self.output_data.dtype
        else:
            self.output_data = None
            self.output_dtype = signals.dtype

        def merged_func(time, inputs):  # pragma: no cover (runs in TF)
            outputs = []
            offset = 0
            for op in self.ops:
                if op.output is None:
                    func = op.fn
                else:
                    func = utils.align_func(self.output_dtype)(op.fn)

                func_input = inputs[:, offset : offset + op.x.shape[0]]
                offset += op.x.shape[0]

                mini_out = []
                for j in range(signals.minibatch_size):
                    if op.t is None:
                        func_out = func(func_input[j])
                    else:
                        func_out = func(time, func_input[j])

                    func_out = np.atleast_1d(func_out)

                    if op.output is None:
                        # just return time as a noop (since we need to
                        # return something)
                        func_out = [time]
                    mini_out += [func_out]
                outputs += [np.stack(mini_out, axis=0)]

            return np.concatenate(outputs, axis=1)

        self.merged_func = merged_func
        self.merged_func.__name__ = "_".join(
            [utils.function_name(op.fn) for op in self.ops]
        )
        self.output_shape = (signals.minibatch_size,)
        self.output_shape += (
            (len(self.ops),) if self.output_data is None else self.output_data.shape
        )

    def build_step(self, signals):
        time = [] if self.time_data is None else signals.gather(self.time_data)
        inputs = [] if self.input_data is None else signals.gather(self.input_data)

        node_outputs = tf.numpy_function(
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
@Builder.register(SparseDotSet)
class SparseDotIncBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.SparseDotInc` operators.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.mode = "inc" if type(self.ops[0]) == SparseDotInc else "update"

        self.Y_data = signals.combine([op.Y for op in self.ops])

        # group all the A's and X's
        self.A_data = signals.combine([op.A for op in self.ops])
        self.X_data = signals.combine([op.X for op in self.ops])

        # the only way A would be minibatched is if it is targeted by an
        # online learning rule, which isn't supported for sparse transforms
        assert not self.A_data.minibatched
        assert self.X_data.minibatched and self.Y_data.minibatched

        # arrange the sparse matrices into a (sparse) block diagonal matrix
        # by adding an offset to each sparse matrix's indices
        sparse_indices = []
        corner = np.zeros(2, dtype=np.int64)
        for op in self.ops:
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
        self.sparse_indices = tf.constant(
            sparse_indices,
            dtype=(
                tf.int32
                if np.all(sparse_indices < np.iinfo(np.int32).max)
                else tf.int64
            ),
        )
        self.A_shape = tf.constant(corner, dtype=tf.int64)

        self.perm = tf.constant((1, 0))

    def build_step(self, signals):
        A = signals.gather(self.A_data)
        X = signals.gather(self.X_data)

        # (sum(a0s), sum(a1s)) x (batch, sum(a1s))
        # -> (sum(a0s), sum(a1s)) x (sum(a1s), batch)
        dot = sparse_matmul(self.sparse_indices, A, self.A_shape, X, transpose_x=True)

        # transpose result back to (batch, sum(a0s))
        dot = tf.transpose(dot, perm=self.perm)

        dot.set_shape((signals.minibatch_size,) + self.Y_data.shape)

        signals.scatter(self.Y_data, dot, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        return True


@Builder.register(TimeUpdate)
class TimeUpdateBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.operator.TimeUpdate` operators.
    """

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        assert len(self.ops) == 1
        op = self.ops[0]

        self.step_data = signals[op.step]
        self.time_data = signals[op.time]
        self.one = tf.constant(1, dtype=tf.int32)

    def build_step(self, signals):
        step = signals.gather(self.step_data)

        step += self.one

        signals.scatter(self.step_data, step)
        signals.scatter(self.time_data, tf.cast(step, signals.dtype) * signals.dt)

    @staticmethod
    def mergeable(x, y):
        # there should only ever be one TimeUpdate so this should never be called
        raise NotImplementedError


@Builder.register(SimProbe)
class SimProbeBuilder(OpBuilder):
    """
    Build a group of `~nengo.builder.probe.SimProbe` operators.
    """

    def build_step(self, signals):
        # doesn't do anything (probe reading is handled directly in
        # TensorGraph._build_inner_loop)
        pass

    @staticmethod
    def mergeable(x, y):
        # set mergeable=False because we don't want to force the read signals into
        # a big block
        return False
