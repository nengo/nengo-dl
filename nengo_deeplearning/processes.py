from nengo.builder.processes import SimProcess
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

    # TODO: what to do if it is None? does this ever happen?
    assert op.output is not None

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
