from nengo.builder.neurons import SimNeurons
import tensorflow as tf

from nengo_deeplearning import utils, Builder, DEBUG


@Builder.register(SimNeurons)
def sim_neurons(op, signals, dt):
    # TODO: create specialized operators for different neuron types
    if DEBUG:
        print("sim_neurons")
        print(op)
        print("J", signals[op.J])
        print("output", signals[op.output])
        print("states", [str(signals[s]) for s in op.states])

    def return_step_math(dt, J, output, *states):
        op.neurons.step_math(dt, J, output, *states)

        return (output,) + states

    output = signals[op.output]
    states = [signals[s] for s in op.states]

    result = tf.py_func(return_step_math,
                        [tf.constant(dt), signals[op.J], output] + states,
                        [output.dtype] + [s.dtype for s in states],
                        name=utils.sanitize_name(repr(op.neurons)))

    for i in range(len(states)):
        signals.assign_view(op.states[i], result[i + 1])

    # we need the control_dependencies to force the state update operators
    # to run (otherwise they look like unused nodes and get optimized out)
    with tf.control_dependencies([signals[s] for s in op.states]):
        output = signals.assign_view(op.output, result[0])

    return output
