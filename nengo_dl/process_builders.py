import logging
import warnings

from nengo.builder.processes import SimProcess
from nengo.synapses import Lowpass, LinearFilter
from nengo.utils.filter_design import (cont2discrete, tf2ss, ss2tf,
                                       BadCoefficients)
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_sparse_ops

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder

logger = logging.getLogger(__name__)


@Builder.register(SimProcess)
class SimProcessBuilder(OpBuilder):
    """Builds a group of :class:`~nengo:nengo.builder.processes.SimProcess`
    operators.

    Calls the appropriate sub-build class for the different process types.

    Attributes
    ----------
    TF_PROCESS_IMPL : list of :class:`~nengo:nengo.Process`
        the process types that have a custom implementation
    """

    TF_PROCESS_IMPL = (Lowpass, LinearFilter)
    pass_rng = True

    def __init__(self, ops, signals, rng):
        logger.debug("sim_process")
        logger.debug([op for op in ops])
        logger.debug("process %s", [op.process for op in ops])
        logger.debug("input %s", [op.input for op in ops])
        logger.debug("output %s", [op.output for op in ops])
        logger.debug("t %s", [op.t for op in ops])

        # if we have a custom tensorflow implementation for this process type,
        # then we build that. otherwise we'll just execute the process step
        # function externally (using `tf.py_func`), so we just need to set up
        # the inputs/outputs for that.

        if isinstance(ops[0].process, self.TF_PROCESS_IMPL):
            # note: we do this two-step check (even though it's redundant) to
            # make sure that TF_PROCESS_IMPL is kept up to date

            if type(ops[0].process) == Lowpass:
                self.built_process = LowpassBuilder(ops, signals)
            elif isinstance(ops[0].process, LinearFilter):
                self.built_process = LinearFilterBuilder(ops, signals)
                # self.built_process = GenericProcessBuilder(ops, signals, rng)
        else:
            self.built_process = GenericProcessBuilder(ops, signals, rng)

    def build_step(self, signals):
        self.built_process.build_step(signals)


class GenericProcessBuilder(object):
    """Builds all process types for which there is no custom Tensorflow
    implementation.

    Notes
    -----
    These will be executed as native Python functions, requiring execution to
    move in and out of Tensorflow.  This can significantly slow down the
    simulation, so any performance-critical processes should consider
    adding a custom Tensorflow implementation for their type instead.
    """

    def __init__(self, ops, signals, rng):
        self.input_data = (None if ops[0].input is None else
                           signals.combine([op.input for op in ops]))
        self.output_data = signals.combine([op.output for op in ops])
        self.output_shape = self.output_data.shape + (signals.minibatch_size,)
        self.mode = "inc" if ops[0].mode == "inc" else "update"
        self.prev_result = []

        # build the step function for each process
        step_fs = [
            [op.process.make_step(
                op.input.shape if op.input is not None else (0,),
                op.output.shape, signals.dt_val,
                op.process.get_rng(rng))
             for _ in range(signals.minibatch_size)] for op in ops]

        # `merged_func` calls the step function for each process and
        # combines the result
        @utils.align_func(self.output_shape, self.output_data.dtype)
        def merged_func(time, input):  # pragma: no cover
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
                    mini_out += [step_fs[i][j](*([time] + x))]
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


class LowpassBuilder(object):
    """Build a group of :class:`~nengo:nengo.Lowpass` synapse operators."""

    def __init__(self, ops, signals):
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

        nums = np.asarray(nums)[:, None]

        # note: applying the negative here
        dens = -np.asarray(dens)[:, None]

        # need to manually broadcast for scatter_mul
        # dens = np.tile(dens, (1, signals.minibatch_size))

        self.nums = tf.constant(nums, dtype=self.output_data.dtype)
        self.dens = tf.constant(dens, dtype=self.output_data.dtype)

    def build_step(self, signals):
        # signals.scatter(self.output_data, self.dens, mode="mul")
        # input = signals.gather(self.input_data)
        # signals.scatter(self.output_data, self.nums * input, mode="inc")

        input = signals.gather(self.input_data)
        output = signals.gather(self.output_data)
        signals.scatter(self.output_data,
                        self.dens * output + self.nums * input)


class LinearFilterBuilder(object):
    """Build a group of :class:`~nengo:nengo.LinearFilter` synapse
    operators."""

    def __init__(self, ops, signals):
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
        self.A = tf.constant(np.concatenate(As, axis=0).flatten(),
                             dtype=signals.dtype)
        self.A_indices = tf.constant(sparse_indices, dtype=(
            tf.int32 if np.all(sparse_indices < np.iinfo(np.int32).max)
            else tf.int64))
        self.A_shape = tf.constant(corner, dtype=tf.int64)

        if np.allclose(Cs, 0):
            self.C = None
        else:
            # add empty dimension for broadcasting
            self.C = tf.constant(np.concatenate(Cs)[:, None],
                                 dtype=signals.dtype)

        if np.allclose(Ds, 0):
            self.D = None
        else:
            # add empty dimension for broadcasting
            self.D = tf.constant(np.asarray(Ds)[:, None], dtype=signals.dtype)

        self.offsets = tf.range(0, len(ops) * As[0].shape[0], As[0].shape[0])

        # create a variable to represent the internal state of the filter
        with tf.variable_scope(utils.sanitize_name(str(op.process)),
                               reuse=False):
            self.state = tf.get_local_variable(
                "state", shape=(self.state_d,
                                signals.minibatch_size * self.signal_d),
                dtype=signals.dtype, trainable=False,
                initializer=tf.zeros_initializer())

    def build_step(self, signals):
        input = signals.gather(self.input_data)
        input = tf.reshape(input, (self.n_ops, -1))

        if self.C is None:
            output = tf.zeros_like(input)
        else:
            output = self.state * self.C
            output = tf.reshape(
                output, (self.n_ops, -1,
                         signals.minibatch_size * self.signal_d))
            output = tf.reduce_sum(output, axis=1)

        if self.D is not None:
            output += self.D * input

        r = gen_sparse_ops._sparse_tensor_dense_mat_mul(
            self.A_indices, self.A, self.A_shape, self.state)

        # make sure that the values based on state have been computed before
        # we update the state
        with tf.control_dependencies([output, r]):
            self.state = tf.assign(self.state, r)
            self.state = tf.scatter_add(self.state, self.offsets, input)

        # make sure that the state update ops run before the next time we
        # apply the filter
        with tf.control_dependencies([self.state]):
            signals.scatter(self.output_data, output)
