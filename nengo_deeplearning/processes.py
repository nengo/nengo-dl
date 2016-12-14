from nengo.builder.processes import SimProcess
from nengo.synapses import Lowpass
from nengo.utils.filter_design import cont2discrete
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


@Builder.register(SimProcess)
def sim_process(op, signals, dt, rng):
    # TODO: create specialized operators for synapses

    if DEBUG:
        print("sim_process")
        print(op)
        print("process", op.process)
        print("input", None if op.input is None else signals[op.input])
        print("output", None if op.output is None else signals[op.output])
        print("t", signals[op.t])
        print("input", op.input)
        print("reads", op.reads)
        print("sets", op.sets)
        print("incs", op.incs)
        print("updates", op.updates)

    input = signals[op.input] if op.input is not None else None
    output = signals[op.output]

    # TODO: what to do if it is None? does this ever happen?
    assert op.output is not None

    if isinstance(op.process, Lowpass):
        if op.process.tau <= 0.03 * dt:
            result = input
        else:
            result = linear_filter(input, output, op.process.num,
                                   op.process.den, dt)
    else:
        shape_in = op.input.shape if op.input is not None else (0,)
        shape_out = op.output.shape
        rng = op.process.get_rng(rng)

        step_f = op.process.make_step(shape_in, shape_out, dt, rng)

        result = tf.py_func(
            utils.align_func(step_f, op.output),
            [signals[op.t]] + ([] if input is None else [input]),
            signals[op.output].dtype,
            name=utils.sanitize_name(type(op.process).__name__))

    return signals.assign_view(op.output, result, inc=op.mode == "inc")


def linear_filter(input, output, num, den, dt):
    num, den, _ = cont2discrete((num, den), dt, method="zoh")
    num = num.flatten()

    num = num[1:] if num[0] == 0 else num
    den = den[1:]  # drop first element (equal to 1)

    if len(num) == 1 and len(den) == 0:
        return num[0] * input
    elif len(num) == 1 and len(den) == 1:
        return -den[0] * output + num[0] * input

    raise NotImplementedError
