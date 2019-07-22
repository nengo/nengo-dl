"""
Build classes for Nengo process operators.
"""

from collections import OrderedDict
from distutils.version import LooseVersion
import logging

from nengo.builder.processes import SimProcess
from nengo.exceptions import SimulationError
from nengo.synapses import Lowpass, LinearFilter
from nengo.utils.filter_design import cont2discrete
from nengo.version import version as nengo_version
import numpy as np
import tensorflow as tf

from nengo_dl import utils
from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.compat import (
    tf_compat, make_linear_step, NoX, OneX, NoD, General, make_process_state,
    make_process_step)

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
        self.state_data = [
            signals.combine([list(op.state.values())[i] for op in ops])
            for i in range(len(ops[0].state))]
        self.mode = "inc" if ops[0].mode == "inc" else "update"
        self.prev_result = []

        # build the step function for each process
        self.step_fs = [[None for _ in range(signals.minibatch_size)]
                        for _ in ops]

        # `merged_func` calls the step function for each process and
        # combines the result
        @utils.align_func(
            [self.output_data.full_shape]
            + [s.full_shape for s in self.state_data],
            [self.output_data.dtype]
            + [s.dtype for s in self.state_data])
        def merged_func(time, *input_state):  # pragma: no cover (runs in TF)
            if any(x is None for a in self.step_fs for x in a):
                raise SimulationError(
                    "build_post has not been called for %s" % self)

            if self.input_data is None:
                input = None
                state = input_state
            else:
                input = input_state[0]
                state = input_state[1:]

            # update state in-place (this will update the state values
            # inside step_fs)
            for i, s in enumerate(state):
                self.step_states[i][...] = s

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

            return [np.concatenate(func_output, axis=0)] + self.step_states

        self.merged_func = merged_func
        self.merged_func.__name__ = utils.sanitize_name(
            "_".join([type(op.process).__name__ for op in ops]))

    def build_step(self, signals):
        input = ([] if self.input_data is None
                 else [signals.gather(self.input_data)])
        state = [signals.gather(s) for s in self.state_data]

        # note: we need to make sure that the previous call to this function
        # has completed before the next starts, since we don't know that the
        # functions are thread safe
        with tf.control_dependencies(self.prev_result), tf.device("/cpu:0"):
            result = tf_compat.py_func(
                self.merged_func, [signals.time] + input + state,
                [self.output_data.dtype] + [s.dtype for s in self.state_data],
                name=self.merged_func.__name__)
            output = result[0]
            state = result[1:]
        self.prev_result = [output]

        output.set_shape(self.output_data.full_shape)
        signals.scatter(self.output_data, output, mode=self.mode)
        for i, s in enumerate(state):
            s.set_shape(self.state_data[i].full_shape)
            signals.scatter(self.state_data[i], s, mode="update")

    def build_post(self, ops, signals, sess, rng):
        # generate state for each op
        step_states = [
            make_process_state(
                op.process,
                op.input.shape if op.input is not None else (0,),
                op.output.shape, signals.dt_val)
            for op in ops]

        # build all the states into combined array with shape
        # (n_states, n_ops, state_d, minibatch_size)
        combined_states = [[None for _ in ops]
                           for _ in range(len(ops[0].state))]
        for i, op in enumerate(ops):
            for j, s in enumerate(step_states[i].values()):
                # duplicate state for each minibatch
                if signals[list(op.state.values())[j]].minibatched:
                    s = np.tile(s[..., None],
                                (1,) * s.ndim + (signals.minibatch_size,))
                combined_states[j][i] = s

        # combine op states along first dimension, giving shape
        # (n_states, n_ops * state_d, minibatch_size)
        # (keeping track of the offset of where each op's state lies in the
        # combined array)
        offsets = [[s.shape[0] for s in state]
                   for state in combined_states]
        offsets = np.cumsum(offsets, axis=-1)
        self.step_states = [np.concatenate(state, axis=0)
                            for state in combined_states]

        # build the step functions
        for i, op in enumerate(ops):
            for j in range(signals.minibatch_size):
                # pass each make_step function a view into the combined state
                state = {}
                for k, name in enumerate(step_states[i]):
                    start = 0 if i == 0 else offsets[k][i - 1]
                    stop = offsets[k][i]

                    state[name] = self.step_states[k][start:stop, ..., j]

                    assert np.allclose(state[name], step_states[i][name])

                self.step_fs[i][j] = make_process_step(
                    op.process,
                    op.input.shape if op.input is not None else (0,),
                    op.output.shape, signals.dt_val, op.process.get_rng(rng),
                    state)


class LowpassBuilder(OpBuilder):
    """Build a group of `~nengo.Lowpass` synapse operators."""

    def __init__(self, ops, signals, config):
        super(LowpassBuilder, self).__init__(ops, signals, config)

        # the main difference between this and the general linearfilter
        # OneX implementation is that this version allows us to merge
        # synapses with different input dimensionality (by duplicating
        # the synapse parameters for every input, rather than using
        # broadcasting)

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

        # note: linear filters are linear systems with n_inputs/n_outputs == 1.
        # we apply them to multidimensional inputs, but we do so by
        # broadcasting that SISO linear system (so it's effectively
        # d 1-dimensional linear systems). this means that we can make
        # some simplifying assumptions, namely that B has shape (state_d, 1),
        # C has shape (1, state_d), and D has shape (1, 1), and then we can
        # implement those operations as (broadcasted) multiplies rather than
        # full matrix multiplications.
        # this also means that the minibatch dimension is identical to the
        # signal dimension (i.e. n m-dimensional signals is the same as
        # 1 n*m-dimensional signal); in either case we're just doing that
        # B/C/D broadcasting along all the non-state dimensions. so in these
        # computations we collapse minibatch and signal dimensions into one.

        self.input_data = signals.combine([op.input for op in ops])
        self.output_data = signals.combine([op.output for op in ops])

        steps = [
            make_linear_step(
                op.process, op.input.shape, op.output.shape, signals.dt_val)
            for op in ops]
        self.step_type = type(steps[0])
        assert all(type(step) == self.step_type for step in steps)

        self.n_ops = len(ops)
        self.signal_d = ops[0].input.shape[0]
        self.state_d = steps[0].A.shape[0]

        if self.step_type == NoX:
            self.A = None
            self.B = None
            self.C = None
            # combine D scalars for each op, and broadcast along signal_d
            self.D = signals.constant(
                np.concatenate([step.D[:, None] for step in steps]),
                dtype=signals.dtype)

            assert self.D.get_shape() == (self.n_ops, 1)
        elif self.step_type == OneX:
            # combine A scalars for each op, and broadcast along state_d
            self.A = signals.constant(
                np.concatenate([step.A for step in steps]),
                dtype=signals.dtype)
            # combine B and C scalars for each op, and broadcast along signal_d
            self.B = signals.constant(
                np.concatenate([step.B * step.C for step in steps]),
                dtype=signals.dtype)
            self.C = None
            self.D = None

            assert self.A.get_shape() == (self.n_ops, 1)
            assert self.B.get_shape() == (self.n_ops, 1)
        else:
            self.A = signals.constant(
                np.stack([step.A for step in steps], axis=0),
                dtype=signals.dtype)
            self.B = signals.constant(
                np.stack([step.B for step in steps], axis=0),
                dtype=signals.dtype)
            self.C = signals.constant(
                np.stack([step.C for step in steps], axis=0),
                dtype=signals.dtype)

            if self.step_type == NoD:
                self.D = None
            else:
                self.D = signals.constant(
                    np.concatenate([step.D[:, None, None] for step in steps]),
                    dtype=signals.dtype)
                assert self.D.get_shape() == (self.n_ops, 1, 1)

            # create a variable to represent the internal state of the filter
            if LooseVersion(nengo_version) < "3.0.0":
                self.state_data = signals.make_internal(
                    "state",
                    (self.n_ops * self.state_d, self.signal_d))
            else:
                self.state_data = signals.combine(
                    [op.state["X"] for op in ops])

            assert self.A.get_shape() == (
                self.n_ops, self.state_d, self.state_d)
            assert self.B.get_shape() == (self.n_ops, self.state_d, 1)
            assert self.C.get_shape() == (self.n_ops, 1, self.state_d)

    def build_step(self, signals):
        input = signals.gather(self.input_data)

        if self.step_type == NoX:
            signals.scatter(self.output_data, self.D * input)
        elif self.step_type == OneX:
            input = tf.reshape(input, (self.n_ops, -1))

            # note: we use the output signal in place of a separate state
            output = signals.gather(self.output_data)
            output = tf.reshape(output, (self.n_ops, -1))

            signals.scatter(self.output_data, self.A * output + self.B * input)
        else:
            input = tf.reshape(input, (self.n_ops, 1, -1))

            state = signals.gather(self.state_data)
            state = tf.reshape(state, (self.n_ops, self.state_d,
                                       self.signal_d * signals.minibatch_size))

            if self.step_type == NoD:
                # for NoD, we update the state before computing the output
                new_state = tf.matmul(self.A, state) + self.B * input
                signals.scatter(self.state_data, new_state)

                output = tf.matmul(self.C, new_state)

                signals.scatter(self.output_data, output)
            else:
                # in the general case, we compute the output before updating
                # the state
                output = tf.matmul(self.C, state)
                if self.step_type == General:
                    output += self.D * input
                signals.scatter(self.output_data, output)

                new_state = tf.matmul(self.A, state) + self.B * input

                signals.mark_gather(self.state_data)
                signals.mark_gather(self.input_data)
                signals.scatter(self.state_data, new_state)


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
            if type(x.process) == Lowpass or type(y.process) == Lowpass:
                # lowpass ops can only be merged with other lowpass ops, since
                # they have a custom implementation
                if type(x.process) != type(y.process):  # noqa: E721
                    return False
            elif isinstance(x.process, LinearFilter):
                # we can only merge linearfilters that have the same state
                # dimensionality (den), the same step type (num), and the same
                # input signal dimensionality
                if (not isinstance(y.process, LinearFilter)
                        or len(y.process.num) != len(x.process.num)
                        or len(y.process.den) != len(x.process.den)
                        or x.input.shape[0] != y.input.shape[0]):
                    return False
            else:
                raise NotImplementedError()
        elif isinstance(y.process, custom_impl):
            return False

        return True
