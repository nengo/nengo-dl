"""
Build classes for Nengo process operators.
"""

from collections import OrderedDict
from distutils.version import LooseVersion
import logging
import warnings

from nengo.builder.processes import SimProcess
from nengo.exceptions import SimulationError
from nengo.synapses import Lowpass, LinearFilter
from nengo.utils.filter_design import (cont2discrete, tf2ss, ss2tf,
                                       BadCoefficients)
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_sparse_ops

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder

logger = logging.getLogger(__name__)


class GenericProcessBuilder(OpBuilder):
    """
    Builds all process types for which there is no custom TensorFlow
    implementation.

    Notes
    -----
    These will be executed as native Python functions, requiring execution to
    move in and out of TensorFlow.  This can significantly slow down the
    simulation, so any performance-critical processes should consider
    adding a custom TensorFlow implementation for their type instead.
    """

    def __init__(self, ops, signals, config):
        super(GenericProcessBuilder, self).__init__(ops, signals, config)

        self.input_data = (None if ops[0].input is None else
                           signals.combine([op.input for op in ops]))
        self.output_data = signals.combine([op.output for op in ops])
        self.output_shape = self.output_data.shape + (signals.minibatch_size,)
        self.mode = "inc" if ops[0].mode == "inc" else "update"
        self.prev_result = []

        # build the step function for each process
        self.step_fs = [[None for _ in range(signals.minibatch_size)]
                        for _ in ops]

        # `merged_func` calls the step function for each process and
        # combines the result
        @utils.align_func(self.output_shape, self.output_data.dtype)
        def merged_func(time, input):  # pragma: no cover
            if any(x is None for a in self.step_fs for x in a):
                raise SimulationError(
                    "build_post has not been called for %s" % self)

            input_offset = 0
            func_output = []
            for i, op in enumerate(ops):
                if op.input is not None:
                    input_shape = op.input.shape[0]
                    func_input = input[input_offset:input_offset + input_shape]
                    input_offset += input_shape

                mini_out = []
                for j in range(signals.minibatch_size):
                    x = [] if op.input is None else [func_input[..., j]]
                    mini_out += [self.step_fs[i][j](*([time] + x))]
                func_output += [np.stack(mini_out, axis=-1)]

            return np.concatenate(func_output, axis=0)

        self.merged_func = merged_func
        self.merged_func.__name__ = utils.sanitize_name(
            "_".join([type(op.process).__name__ for op in ops]))

    def build_step(self, signals):
        input = ([] if self.input_data is None
                 else signals.gather(self.input_data))

        # note: we need to make sure that the previous call to this function
        # has completed before the next starts, since we don't know that the
        # functions are thread safe
        with tf.control_dependencies(self.prev_result), tf.device("/cpu:0"):
            result = tf.py_func(
                self.merged_func, [signals.time, input],
                self.output_data.dtype, name=self.merged_func.__name__)
        result.set_shape(self.output_shape)
        self.prev_result = [result]

        signals.scatter(self.output_data, result, mode=self.mode)

    def build_post(self, ops, signals, sess, rng):
        for i, op in enumerate(ops):
            for j in range(signals.minibatch_size):
                self.step_fs[i][j] = op.process.make_step(
                    op.input.shape if op.input is not None else (0,),
                    op.output.shape, signals.dt_val, op.process.get_rng(rng))


class LowpassBuilder(OpBuilder):
    """Build a group of `~nengo.Lowpass` synapse operators."""

    def __init__(self, ops, signals, config):
        super(LowpassBuilder, self).__init__(ops, signals, config)

        self.input_data = signals.combine([op.input for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        nums = []
        dens = []
        for op in ops:
            if op.process.tau <= 0.03 * signals.dt_val:
                num = 1
                den = 0
            else:
                num, den, _ = cont2discrete((op.process.num, op.process.den),
                                            signals.dt_val, method="zoh")
                num = num.flatten()

                num = num[1:] if num[0] == 0 else num
                assert len(num) == 1
                num = num[0]

                assert len(den) == 2
                den = den[1]

            nums += [num] * op.input.shape[0]
            dens += [den] * op.input.shape[0]

        nums = np.asarray(nums)
        while nums.ndim < len(self.input_data.full_shape):
            nums = np.expand_dims(nums, -1)

        # note: applying the negative here
        dens = -np.asarray(dens)
        while dens.ndim < len(self.input_data.full_shape):
            dens = np.expand_dims(dens, -1)

        # need to manually broadcast for scatter_mul
        # dens = np.tile(dens, (1, signals.minibatch_size))

        self.nums = signals.constant(nums, dtype=self.output_data.dtype)
        self.dens = signals.constant(dens, dtype=self.output_data.dtype)

        # create a variable to represent the internal state of the filter
        # self.state_sig = signals.make_internal(
        #     "state", self.output_data.shape)

    def build_step(self, signals):
        # signals.scatter(self.output_data, self.dens, mode="mul")
        # input = signals.gather(self.input_data)
        # signals.scatter(self.output_data, self.nums * input, mode="inc")

        input = signals.gather(self.input_data)
        output = signals.gather(self.output_data)

        signals.scatter(self.output_data,
                        self.dens * output + self.nums * input)

        # method using internal state signal
        # note: this isn't used for efficiency reasons (we can avoid an extra
        # scatter by reusing the output signal as the state signal)
        # input = signals.gather(self.input_data)
        # prev_state = signals.gather(self.state_sig)
        # new_state = self.dens * prev_state + self.nums * input
        # signals.scatter(self.state_sig, new_state)
        # signals.scatter(self.output_data, new_state)


class LinearFilterBuilder(OpBuilder):
    """Build a group of `~nengo.LinearFilter` synapse operators."""

    def __init__(self, ops, signals, config):
        super(LinearFilterBuilder, self).__init__(ops, signals, config)

        self.input_data = signals.combine([op.input for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        self.n_ops = len(ops)
        self.signal_d = ops[0].input.shape[0]
        As = []
        Cs = []
        Ds = []
        # compute the A/C/D matrices for each operator
        for op in ops:
            A, B, C, D = tf2ss(op.process.num, op.process.den)

            if op.process.analog:
                # convert to discrete system
                A, B, C, D, _ = cont2discrete((A, B, C, D), signals.dt_val,
                                              method="zoh")

            # convert to controllable form
            num, den = ss2tf(A, B, C, D)

            if op.process.analog:
                # add shift
                num = np.concatenate((num, [[0]]), axis=1)

            with warnings.catch_warnings():
                # ignore the warning about B, since we aren't using it anyway
                warnings.simplefilter("ignore", BadCoefficients)
                A, _, C, D = tf2ss(num, den)

            As.append(A)
            Cs.append(C[0])
            Ds.append(D.item())

        self.state_d = sum(x.shape[0] for x in Cs)

        # build a sparse matrix containing the A matrices as blocks
        # along the diagonal
        sparse_indices = []
        corner = np.zeros(2, dtype=np.int64)
        for A in As:
            idxs = np.reshape(np.dstack(np.meshgrid(
                np.arange(A.shape[0]), np.arange(A.shape[1]),
                indexing="ij")), (-1, 2))
            idxs += corner
            corner += A.shape
            sparse_indices += [idxs]
        sparse_indices = np.concatenate(sparse_indices, axis=0)
        self.A = signals.constant(np.concatenate(As, axis=0).flatten(),
                                  dtype=signals.dtype)
        self.A_indices = signals.constant(sparse_indices, dtype=(
            tf.int32 if np.all(sparse_indices < np.iinfo(np.int32).max)
            else tf.int64))
        self.A_shape = tf.constant(corner, dtype=tf.int64)

        if np.allclose(Cs, 0):
            self.C = None
        else:
            # add empty dimension for broadcasting
            self.C = signals.constant(np.concatenate(Cs)[:, None],
                                      dtype=signals.dtype)

        if np.allclose(Ds, 0):
            self.D = None
        else:
            # add empty dimension for broadcasting
            self.D = signals.constant(np.asarray(Ds)[:, None],
                                      dtype=signals.dtype)

        self.offsets = tf.expand_dims(
            tf.range(0, len(ops) * As[0].shape[0], As[0].shape[0]),
            axis=1)

        # create a variable to represent the internal state of the filter
        self.state_sig = signals.make_internal(
            "state", (self.state_d, signals.minibatch_size * self.signal_d),
            minibatched=False)

    def build_step(self, signals):
        input = signals.gather(self.input_data)
        input = tf.reshape(input, (self.n_ops, -1))

        state = signals.gather(self.state_sig)

        # compute output
        if self.C is None:
            output = tf.zeros_like(input)
        else:
            output = state * self.C
            output = tf.reshape(
                output,
                (self.n_ops, -1, signals.minibatch_size * self.signal_d))
            output = tf.reduce_sum(output, axis=1)

        if self.D is not None:
            output += self.D * input

        signals.scatter(self.output_data, output)

        # update state
        if LooseVersion(tf.__version__) < LooseVersion("1.7.0"):
            mat_mul = gen_sparse_ops._sparse_tensor_dense_mat_mul
        else:
            mat_mul = gen_sparse_ops.sparse_tensor_dense_mat_mul
        r = mat_mul(self.A_indices, self.A, self.A_shape, state)

        with tf.control_dependencies([output]):
            state = r + tf.scatter_nd(self.offsets, input,
                                      self.state_sig.shape)
            # TODO: tensorflow does not yet support sparse_tensor_dense_add
            # on the GPU
            # state = gen_sparse_ops._sparse_tensor_dense_add(
            #     self.offsets, input, self.state_sig.shape, r)
        state.set_shape(self.state_sig.shape)

        signals.mark_gather(self.input_data)
        signals.mark_gather(self.state_sig)
        signals.scatter(self.state_sig, state)


@Builder.register(SimProcess)
class SimProcessBuilder(OpBuilder):
    """
    Builds a group of `~nengo.builder.processes.SimProcess` operators.

    Calls the appropriate sub-build class for the different process types.

    Attributes
    ----------
    TF_PROCESS_IMPL : dict of {`~nengo.Process`: `.builder.OpBuilder`}
        Mapping from process types to custom build classes (processes without
        a custom builder will use the generic builder).
    """

    # we use OrderedDict because it is important that Lowpass come before
    # LinearFilter (since we'll be using isinstance to find the right builder,
    # and Lowpass is a subclass of LinearFilter)
    TF_PROCESS_IMPL = OrderedDict([
        (Lowpass, LowpassBuilder),
        (LinearFilter, LinearFilterBuilder),
    ])

    def __init__(self, ops, signals, config):
        super(SimProcessBuilder, self).__init__(ops, signals, config)

        logger.debug("process %s", [op.process for op in ops])
        logger.debug("input %s", [op.input for op in ops])
        logger.debug("output %s", [op.output for op in ops])
        logger.debug("t %s", [op.t for op in ops])

        # if we have a custom tensorflow implementation for this process type,
        # then we build that. otherwise we'll execute the process step
        # function externally (using `tf.py_func`).
        for process_type, process_builder in self.TF_PROCESS_IMPL.items():
            if isinstance(ops[0].process, process_type):
                self.built_process = process_builder(ops, signals, config)
                break
        else:
            self.built_process = GenericProcessBuilder(ops, signals, config)

    def build_step(self, signals):
        self.built_process.build_step(signals)

    def build_post(self, ops, signals, sess, rng):
        if isinstance(self.built_process, GenericProcessBuilder):
            self.built_process.build_post(ops, signals, sess, rng)

    @staticmethod
    def mergeable(x, y):
        # we can merge ops if they have a custom implementation, or merge
        # generic processes, but can't mix the two
        custom_impl = tuple(SimProcessBuilder.TF_PROCESS_IMPL.keys())
        if isinstance(x.process, custom_impl):
            if type(x.process) == Lowpass:
                # lowpass ops can only be merged with other lowpass ops, since
                # they have a custom implementation
                if type(y.process) != Lowpass:
                    return False
            elif isinstance(x.process, LinearFilter):
                # we can only merge linearfilters that have the same state
                # dimensionality and the same signal dimensionality
                if (not isinstance(y.process, LinearFilter) or
                        len(y.process.den) != len(x.process.den) or
                        x.input.shape[0] != y.input.shape[0]):
                    return False
            else:
                raise NotImplementedError()
        elif isinstance(y.process, custom_impl):
            return False

        return True
