from nengo.builder.processes import SimProcess
from nengo.synapses import Lowpass
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG
from nengo_deeplearning.builder import Builder, OpBuilder

TF_PROCESS_IMPL = (Lowpass,)


@Builder.register(SimProcess)
class SimProcessBuilder(OpBuilder):
    pass_rng = True

    def __init__(self, ops, signals, rng):
        if DEBUG:
            print("sim_process")
            print([op for op in ops])
            print("process", [op.process for op in ops])
            print("input", [op.input for op in ops])
            print("output", [op.output for op in ops])
            print("t", [op.t for op in ops])

        self.input_data = (None if ops[0].input is None else
                           signals.combine([op.input for op in ops]))

        self.output_data = signals.combine([op.output for op in ops])
        self.mode = "inc" if ops[0].mode == "inc" else "update"

        self.process_type = type(ops[0].process)

        if self.process_type in TF_PROCESS_IMPL:
            # note: we do this two-step check (even though it's redundant) to
            # make sure that TF_PROCESS_IMPL is kept up to date

            if self.process_type == Lowpass:
                self.process = LinearFilter(ops, signals, self.input_data,
                                            self.output_data)
        else:
            step_fs = [
                op.process.make_step(
                    op.input.shape if op.input is not None else (0,),
                    op.output.shape, signals.dt.dt_val,
                    op.process.get_rng(rng)) for op in ops]

            def merged_func(time, input):
                input_offset = 0
                func_output = []
                for i, op in enumerate(ops):
                    func_input = [time]
                    if op.input is not None:
                        input_shape = op.input.shape[0]
                        func_input += [
                            input[input_offset:input_offset + input_shape]]
                        input_offset += input_shape
                    func_output += [step_fs[i](*func_input)]

                return np.concatenate(func_output, axis=0)

            self.merged_func = merged_func
            self.merged_func.__name__ = utils.sanitize_name(
                "_".join([type(op.process).__name__ for op in ops]))

    def build_step(self, signals):
        if self.process_type in TF_PROCESS_IMPL:
            # note: we do this two-step check (even though it's redundant) to make
            # sure that TF_PROCESS_IMPL is kept up to date

            if self.process_type == Lowpass:
                # output = signals.gather(self.output_data)
                self.process.build_step(signals)
        else:
            input = ([] if self.input_data is None
                     else signals.gather(self.input_data))

            result = tf.py_func(
                utils.align_func(self.merged_func, self.output_data.shape,
                                 self.output_data.dtype),
                [signals.time, input], self.output_data.dtype,
                name=self.merged_func.__name__)
            result.set_shape(self.output_data.shape)

            signals.scatter(self.output_data, result, mode=self.mode)


class LinearFilter(object):
    def __init__(self, ops, signals, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

        nums = []
        dens = []
        for op in ops:
            if op.process.tau <= 0.03 * signals.dt.dt_val:
                num = 1
                den = 0
            else:
                num, den, _ = cont2discrete((op.process.num, op.process.den),
                                            signals.dt.dt_val, method="zoh")
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

        self.nums = tf.constant(nums, dtype=output_data.dtype)
        # note: applying the negative here
        self.dens = tf.constant(-np.asarray(dens), dtype=output_data.dtype)

    def build_step(self, signals):
        input = signals.gather(self.input_data)
        signals.scatter(self.output_data, self.dens, mode="mul")
        signals.scatter(self.output_data, self.nums * input, mode="inc")
