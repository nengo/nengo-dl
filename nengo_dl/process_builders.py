"""
Build classes for Nengo process operators.
"""

import contextlib
import logging

import numpy as np
import tensorflow as tf
from nengo.builder.processes import SimProcess
from nengo.exceptions import SimulationError
from nengo.synapses import LinearFilter, Lowpass
from nengo.utils.filter_design import cont2discrete

from nengo_dl import compat, utils
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

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.time_data = signals[self.ops[0].t].reshape(())
        self.input_data = (
            None
            if self.ops[0].input is None
            else signals.combine([op.input for op in self.ops])
        )
        self.output_data = signals.combine([op.output for op in self.ops])
        self.state_data = [
            signals.combine([list(op.state.values())[i] for op in self.ops])
            for i in range(len(self.ops[0].state))
        ]
        self.mode = "inc" if self.ops[0].mode == "inc" else "update"

        self.prev_result = []

        # `merged_func` calls the step function for each process and
        # combines the result
        def merged_func(time, *input_state):  # pragma: no cover (runs in TF)
            if not hasattr(self, "step_fs"):
                raise SimulationError(f"build_post has not been called for {self}")

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
            for i, op in enumerate(self.ops):
                if op.input is not None:
                    input_shape = op.input.shape[0]
                    func_input = input[:, input_offset : input_offset + input_shape]
                    input_offset += input_shape

                mini_out = []
                for j in range(signals.minibatch_size):
                    x = [] if op.input is None else [func_input[j]]
                    mini_out += [self.step_fs[i][j](*([time] + x))]
                func_output += [np.stack(mini_out, axis=0)]

            return [np.concatenate(func_output, axis=1)] + self.step_states

        self.merged_func = merged_func
        self.merged_func.__name__ = utils.sanitize_name(
            "_".join([type(op.process).__name__ for op in self.ops])
        )

    def build_step(self, signals):
        time = [signals.gather(self.time_data)]
        input = [] if self.input_data is None else [signals.gather(self.input_data)]
        state = [signals.gather(s) for s in self.state_data]

        if compat.eager_enabled():
            # noop
            control_deps = contextlib.suppress()
        else:
            # we need to make sure that the previous call to this function
            # has completed before the next starts, since we don't know that the
            # functions are thread safe
            control_deps = tf.control_dependencies(self.prev_result)

        with control_deps:
            result = tf.numpy_function(
                self.merged_func,
                time + input + state,
                [self.output_data.dtype] + [s.dtype for s in self.state_data],
                name=self.merged_func.__name__,
            )

        # TensorFlow will automatically squeeze length-1 outputs (if there is
        # no state), which we don't want
        result = tf.nest.flatten(result)

        output = result[0]
        state = result[1:]

        self.prev_result = [output]

        output.set_shape(self.output_data.full_shape)
        signals.scatter(self.output_data, output, mode=self.mode)
        for i, s in enumerate(state):
            s.set_shape(self.state_data[i].full_shape)
            signals.scatter(self.state_data[i], s, mode="update")

    def build_post(self, signals):
        # generate state for each op
        step_states = [
            op.process.make_state(
                op.input.shape if op.input is not None else (0,),
                op.output.shape,
                signals.dt_val,
            )
            for op in self.ops
        ]

        # build all the states into combined array with shape
        # (n_states, n_ops, *state_d)
        combined_states = [
            [None for _ in self.ops] for _ in range(len(self.ops[0].state))
        ]
        for i, op in enumerate(self.ops):
            # note: we iterate over op.state so that the order is always based on that
            # dict's order (which is what we used to set up self.state_data)
            for j, name in enumerate(op.state):
                combined_states[j][i] = step_states[i][name]

        # combine op states, giving shape
        # (n_states, n_ops * state_d[0], *state_d[1:])
        # (keeping track of the offset of where each op's state lies in the
        # combined array)
        offsets = [[s.shape[0] for s in state] for state in combined_states]
        offsets = np.cumsum(offsets, axis=-1)
        self.step_states = [np.concatenate(state, axis=0) for state in combined_states]

        # cast to appropriate dtype
        for i, state in enumerate(self.state_data):
            self.step_states[i] = self.step_states[i].astype(state.dtype)

        # duplicate state for each minibatch, giving shape
        # (n_states, minibatch_size, n_ops * state_d[0], *state_d[1:])
        assert all(s.minibatched for op in self.ops for s in op.state.values())
        for i, state in enumerate(self.step_states):
            self.step_states[i] = np.tile(
                state[None, ...], (signals.minibatch_size,) + (1,) * state.ndim
            )

        # build the step functions
        self.step_fs = [[None for _ in range(signals.minibatch_size)] for _ in self.ops]
        for i, op in enumerate(self.ops):
            for j in range(signals.minibatch_size):
                # pass each make_step function a view into the combined state
                state = {}
                for k, name in enumerate(op.state):
                    start = 0 if i == 0 else offsets[k][i - 1]
                    stop = offsets[k][i]

                    state[name] = self.step_states[k][j, start:stop]

                    assert np.allclose(state[name], step_states[i][name])

                self.step_fs[i][j] = op.process.make_step(
                    op.input.shape if op.input is not None else (0,),
                    op.output.shape,
                    signals.dt_val,
                    op.process.get_rng(self.config.rng),
                    state,
                )

                self.step_fs[i][j] = utils.align_func(self.output_data.dtype)(
                    self.step_fs[i][j]
                )


class LowpassBuilder(OpBuilder):
    """Build a group of `~nengo.Lowpass` synapse operators."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        # the main difference between this and the general linearfilter
        # OneX implementation is that this version allows us to merge
        # synapses with different input dimensionality (by duplicating
        # the synapse parameters for every input, rather than using
        # broadcasting)

        self.input_data = signals.combine([op.input for op in self.ops])
        self.output_data = signals.combine([op.output for op in self.ops])

        nums = []
        dens = []
        for op in self.ops:
            if op.process.tau <= 0.03 * signals.dt_val:
                num = 1
                den = 0
            else:
                num, den, _ = cont2discrete(
                    (op.process.num, op.process.den), signals.dt_val, method="zoh"
                )
                num = num.flatten()

                num = num[1:] if num[0] == 0 else num
                assert len(num) == 1
                num = num[0]

                assert len(den) == 2
                den = den[1]

            nums += [num] * op.input.shape[0]
            dens += [den] * op.input.shape[0]

        if self.input_data.minibatched:
            # add batch dimension for broadcasting
            nums = np.expand_dims(nums, 0)
            dens = np.expand_dims(dens, 0)

        # apply the negative here
        dens = -np.asarray(dens)

        self.nums = tf.constant(nums, dtype=self.output_data.dtype)
        self.dens = tf.constant(dens, dtype=self.output_data.dtype)

        # create a variable to represent the internal state of the filter
        # self.state_sig = signals.make_internal(
        #     "state", self.output_data.shape)

    def build_step(self, signals):
        input = signals.gather(self.input_data)
        output = signals.gather(self.output_data)

        signals.scatter(self.output_data, self.dens * output + self.nums * input)

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

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

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

        self.input_data = signals.combine([op.input for op in self.ops])
        self.output_data = signals.combine([op.output for op in self.ops])

        if self.input_data.ndim != 1:
            raise NotImplementedError(
                "LinearFilter of non-vector signals is not implemented"
            )

        steps = [
            op.process.make_step(
                op.input.shape,
                op.output.shape,
                signals.dt_val,
                state=op.process.make_state(
                    op.input.shape, op.output.shape, signals.dt_val
                ),
                rng=None,
            )
            for op in self.ops
        ]
        self.filter_step = steps[0]
        assert all(isinstance(step, type(self.filter_step)) for step in steps)

        self.n_ops = len(self.ops)
        self.signal_d = self.ops[0].input.shape[0]
        self.state_d = steps[0].A.shape[0]

        if isinstance(self.filter_step, LinearFilter.NoX):
            self.A = None
            self.B = None
            self.C = None
            # combine D scalars for each op, and broadcast along minibatch and
            # signal dimensions
            self.D = tf.constant(
                np.concatenate([step.D[None, :, None] for step in steps], axis=1),
                dtype=signals.dtype,
            )

            assert self.D.shape == (1, self.n_ops, 1)
        elif isinstance(self.filter_step, LinearFilter.OneX):
            # combine A scalars for each op, and broadcast along batch/state
            self.A = tf.constant(
                np.concatenate([step.A for step in steps])[None, :], dtype=signals.dtype
            )
            # combine B and C scalars for each op, and broadcast along batch/state
            self.B = tf.constant(
                np.concatenate([step.B * step.C for step in steps])[None, :],
                dtype=signals.dtype,
            )
            self.C = None
            self.D = None

            assert self.A.shape == (1, self.n_ops, 1)
            assert self.B.shape == (1, self.n_ops, 1)
        else:
            self.A = tf.constant(
                np.stack([step.A for step in steps], axis=0), dtype=signals.dtype
            )
            self.B = tf.constant(
                np.stack([step.B for step in steps], axis=0), dtype=signals.dtype
            )
            self.C = tf.constant(
                np.stack([step.C for step in steps], axis=0), dtype=signals.dtype
            )

            if isinstance(self.filter_step, LinearFilter.NoD):
                self.D = None
            else:
                self.D = tf.constant(
                    np.concatenate([step.D[:, None, None] for step in steps]),
                    dtype=signals.dtype,
                )
                assert self.D.shape == (self.n_ops, 1, 1)

            # create a variable to represent the internal state of the filter
            self.state_data = signals.combine([op.state["X"] for op in self.ops])

            assert self.A.shape == (self.n_ops, self.state_d, self.state_d)
            assert self.B.shape == (self.n_ops, self.state_d, 1)
            assert self.C.shape == (self.n_ops, 1, self.state_d)

    def build_step(self, signals):
        input = signals.gather(self.input_data)

        if isinstance(self.filter_step, LinearFilter.NoX):
            input = tf.reshape(input, (signals.minibatch_size, self.n_ops, -1))

            signals.scatter(self.output_data, self.D * input)
        elif isinstance(self.filter_step, LinearFilter.OneX):
            input = tf.reshape(input, (signals.minibatch_size, self.n_ops, -1))

            # note: we use the output signal in place of a separate state
            output = signals.gather(self.output_data)
            output = tf.reshape(output, (signals.minibatch_size, self.n_ops, -1))

            signals.scatter(self.output_data, self.A * output + self.B * input)
        else:
            # TODO: possible to rework things to not require all the
            #  transposing/reshaping required for moving batch to end?
            def undo_batch(x):
                x = tf.reshape(x, x.shape[:-1].as_list() + [-1, signals.minibatch_size])
                x = tf.transpose(x, np.roll(np.arange(x.shape.ndims), 1))
                return x

            # separate by op and collapse batch/state dimensions
            assert input.shape.ndims == 2
            input = tf.transpose(input)
            input = tf.reshape(
                input, (self.n_ops, 1, self.signal_d * signals.minibatch_size)
            )

            state = signals.gather(self.state_data)
            assert input.shape.ndims == 3
            state = tf.transpose(state, perm=(1, 2, 0))
            state = tf.reshape(
                state,
                (self.n_ops, self.state_d, self.signal_d * signals.minibatch_size),
            )

            if isinstance(self.filter_step, LinearFilter.NoD):
                # for NoD, we update the state before computing the output
                new_state = tf.matmul(self.A, state) + self.B * input

                signals.scatter(self.state_data, undo_batch(new_state))

                output = tf.matmul(self.C, new_state)

                signals.scatter(self.output_data, undo_batch(output))
            else:
                # in the general case, we compute the output before updating
                # the state
                output = tf.matmul(self.C, state)
                if isinstance(self.filter_step, LinearFilter.General):
                    output += self.D * input
                signals.scatter(self.output_data, undo_batch(output))

                new_state = tf.matmul(self.A, state) + self.B * input

                signals.scatter(self.state_data, undo_batch(new_state))


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

    # it is important that Lowpass come before LinearFilter because we'll be using
    # isinstance to find the right builder, and Lowpass is a subclass of LinearFilter
    TF_PROCESS_IMPL = {Lowpass: LowpassBuilder, LinearFilter: LinearFilterBuilder}

    def __init__(self, ops):
        super().__init__(ops)

        logger.debug("process %s", [op.process for op in ops])
        logger.debug("input %s", [op.input for op in ops])
        logger.debug("output %s", [op.output for op in ops])
        logger.debug("t %s", [op.t for op in ops])

        # if we have a custom tensorflow implementation for this process type,
        # then we build that. otherwise we'll execute the process step
        # function externally (using `tf.py_func`).
        for process_type, process_builder in self.TF_PROCESS_IMPL.items():
            if isinstance(ops[0].process, process_type):
                self.built_process = process_builder(ops)
                break
        else:
            self.built_process = GenericProcessBuilder(ops)

    def build_pre(self, signals, config):
        self.built_process.build_pre(signals, config)

    def build_step(self, signals):
        self.built_process.build_step(signals)

    def build_post(self, signals):
        if isinstance(self.built_process, GenericProcessBuilder):
            self.built_process.build_post(signals)

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
                if (
                    not isinstance(y.process, LinearFilter)
                    or len(y.process.num) != len(x.process.num)
                    or len(y.process.den) != len(x.process.den)
                    or x.input.shape[0] != y.input.shape[0]
                ):
                    return False
            else:
                raise NotImplementedError()
        elif isinstance(y.process, custom_impl):
            return False

        return True
