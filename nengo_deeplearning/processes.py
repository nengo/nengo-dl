from nengo.builder.processes import SimProcess
from nengo.synapses import Lowpass
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG

TF_PROCESS_IMPL = (Lowpass,)


@Builder.register(SimProcess)
def sim_process(ops, signals, dt, rng):
    # TODO: create specialized operators for synapses

    if DEBUG:
        print("sim_process")
        print([op for op in ops])
        print("process", [op.process for op in ops])
        print("input", [op.input for op in ops])
        print("output", [op.output for op in ops])
        print("t", [op.t for op in ops])

    # TODO: what to do if it is None? does this ever happen?
    assert not any([op.output is None for op in ops])

    if ops[0].input is None:
        input = []
    else:
        input = signals[[op.input for op in ops]]

    output = signals.combine([op.output for op in ops])

    process_type = type(ops[0].process)

    if process_type in TF_PROCESS_IMPL:
        # note: we do this two-step check (even though it's redundant) to make
        # sure that TF_PROCESS_IMPL is kept up to date

        if process_type == Lowpass:
            result = linear_filter(input, signals[output], ops, dt)

            # result = utils.print_op(result, "executing filter")
    else:
        # shape_in = op.input.shape if op.input is not None else (0,)
        # shape_out = op.output.shape
        # rng = op.process.get_rng(rng)

        step_fs = [op.process.make_step(
            op.input.shape if op.input is not None else (0,), op.output.shape,
            dt, op.process.get_rng(rng)) for op in ops]

        def merged_func(time, input):
            input_offset = 0
            func_output = []
            for i, op in enumerate(ops):
                func_input = [time]
                if op.input is not None:
                    func_input += [
                        input[input_offset:input_offset + op.input.shape[0]]]
                    input_offset += op.input.shape[0]
                func_output += [step_fs[i](*func_input)]

            return np.concatenate(func_output, axis=0)

        result = tf.py_func(
            utils.align_func(merged_func, output.shape, output.dtype),
            [signals.time, input], output.dtype,
            name=utils.sanitize_name("_".join([type(op.process).__name__
                                               for op in ops])))
        result.set_shape(output.shape)

    # note: if op.mode=="update" we return the new variable (which will be used
    # in the next iteration of the while loop), but we don't update `signals`
    # (so it doesn't affect calculations on this time step)
    # if ops[0].mode == "inc":
    #     signals.scatter(output, result, mode="inc")
    # elif ops[0].mode == "set":
    #     signals.scatter(output, result, mode="update")
    signals.scatter(
        [output], result, mode="inc" if ops[0].mode == "inc" else "update")


    # return result


def linear_filter(input, output, ops, dt):
    nums = []
    dens = []
    for op in ops:
        if op.process.tau <= 0.03 * dt:
            num = 1
            den = 0
        else:
            num, den, _ = cont2discrete((op.process.num, op.process.den), dt,
                                        method="zoh")
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

    nums = tf.constant(nums, dtype=output.dtype)
    dens = tf.constant(dens, dtype=output.dtype)

    # if len(num) == 1 and len(den) == 0:
    #     return num[0] * input
    # elif len(num) == 1 and len(den) == 1:
    #     return -den[0] * output + num[0] * input

    tmp = -dens * output + nums * input
    return tmp
