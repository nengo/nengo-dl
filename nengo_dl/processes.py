import logging

from nengo.builder.processes import SimProcess
from nengo.synapses import Lowpass
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf

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

    TF_PROCESS_IMPL = (Lowpass,)
    pass_rng = True

    def __init__(self, ops, signals, rng):
        logger.debug("sim_process")
        logger.debug([op for op in ops])
        logger.debug("process %s", [op.process for op in ops])
        logger.debug("input %s", [op.input for op in ops])
        logger.debug("output %s", [op.output for op in ops])
        logger.debug("t %s", [op.t for op in ops])

        process_type = type(ops[0].process)

        # if we have a custom tensorflow implementation for this process type,
        # then we build that. otherwise we'll just execute the process step
        # function externally (using `tf.py_func`), so we just need to set up
        # the inputs/outputs for that.
        if process_type in self.TF_PROCESS_IMPL:
            # note: we do this two-step check (even though it's redundant) to
            # make sure that TF_PROCESS_IMPL is kept up to date

            if process_type == Lowpass:
                self.built_process = LowpassBuilder(ops, signals)
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
        def merged_func(time, input):
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
    """Build a group of :class:`~nengo:nengo.LinearFilter`
    neuron operators."""
    def __init__(self, ops, signals):
        # TODO: implement general linear filter (using queues?)

        self.input_data = (None if ops[0].input is None else
                           signals.combine([op.input for op in ops]))
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

                den = den[1:]  # drop first element (equal to 1)
                if len(den) == 0:
                    den = 0
                else:
                    assert len(den) == 1
                    den = den[0]

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
