import nengo
from nengo.builder.signal import Signal, SignalError
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, learning_rules

DEBUG = False


def convert(op, signals, dt, rng):
    if DEBUG:
        print("===================")
        print("CONVERTING", op)

    # add operator signals to `signals` dict
    for sig in op.all_signals:
        if sig not in signals:
            if DEBUG:
                print("creating signal", sig)
            signals.init(sig)

    # TODO: some fancier registry system
    if isinstance(op, nengo.builder.operator.Reset):
        output = reset(op, signals)
    elif isinstance(op, nengo.builder.operator.Copy):
        output = copy(op, signals)
    elif isinstance(op, nengo.builder.operator.ElementwiseInc):
        output = elementwise_inc(op, signals)
    elif isinstance(op, nengo.builder.operator.DotInc):
        output = dot_inc(op, signals)
    elif isinstance(op, nengo.builder.operator.TimeUpdate):
        output = time_update(op, signals, dt)
    elif isinstance(op, nengo.builder.operator.SlicedCopy):
        output = sliced_copy(op, signals)
    elif isinstance(op, nengo.builder.operator.SimPyFunc):
        output = sim_py_func(op, signals)
    elif isinstance(op, nengo.builder.neurons.SimNeurons):
        output = sim_neurons(op, signals, dt)
    elif isinstance(op, nengo.builder.processes.SimProcess):
        output = sim_process(op, signals, dt, rng)
    elif isinstance(op, nengo.builder.learning_rules.SimBCM):
        output = learning_rules.sim_bcm(op, signals, dt)
    elif isinstance(op, nengo.builder.learning_rules.SimOja):
        output = learning_rules.sim_oja(op, signals, dt)
    elif isinstance(op, nengo.builder.learning_rules.SimVoja):
        output = learning_rules.sim_voja(op, signals, dt)
    else:
        raise NotImplementedError

    assert output is not None

    if isinstance(output, tf.Tensor):
        output = [output]
    elif isinstance(output, tuple):
        output = list(output)
    return output


def time_update(op, signals, dt):
    signals[op.step] = tf.assign_add(signals[op.step], 1)
    # TODO: is there really no way to multiply an int by a float?
    signals[op.time] = tf.assign(signals[op.time],
                                 tf.to_float(signals[op.step]) * dt)

    return signals[op.step], signals[op.time]


def reset(op, signals):
    if DEBUG:
        print("reset")
        print(op)
        print("dst", signals[op.dst])
        print("val", op.value)

    # convert value to appropriate dtype
    value = np.asarray(op.value, dtype=op.dst.dtype)

    # convert value to appropriate shape (note: this can change the
    # number of elements, which we use to broadcast scalars up to full
    # matrices)
    value = np.resize(value, op.dst.shape)

    if DEBUG:
        print("val", value)

    return assign_view(signals, op.dst, tf.constant(value))


def copy(op, signals):
    return assign_view(signals, op.dst, op.src)


def sliced_copy(op, signals):
    if DEBUG:
        print("sliced_copy")
        print(op)
        print("dst", signals[op.dst])
        print("dst.base", signals[op.dst.base])
        print(op.dst_slice)
        print("src", signals[op.src])
        print(op.src_slice)

    return assign_view(signals, op.dst, op.src, dst_slice=op.dst_slice,
                       src_slice=op.src_slice, inc=op.inc)


def elementwise_inc(op, signals):
    if DEBUG:
        print("elementwise_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    return assign_view(signals, op.Y, signals[op.A] * signals[op.X], inc=True)


def dot_inc(op, signals):
    # note: this is matrix/vector (A) by vector (X) multiplication,
    # not matrix-matrix

    if DEBUG:
        print("dot_inc")
        print(op)
        print("dst", signals[op.Y])
        print("A", signals[op.A])
        print("X", signals[op.X])

    dot = tf.mul(signals[op.A], signals[op.X])
    if op.A.ndim == 2:
        dot = tf.reduce_sum(dot, axis=1)

    return assign_view(signals, op.Y, dot, inc=True)


def sim_py_func(op, signals):
    if DEBUG:
        print("sim_py_func")
        print(op)
        print("t", op.t)
        print("x", op.x)
        print("fn", op.fn)

    inputs = []
    if op.t is not None:
        inputs += [signals[op.t]]
    if op.x is not None:
        inputs += [signals[op.x]]

    if op.output is None:
        def noop_func(*args):
            op.fn(*args)
            return args

        node_outputs = tf.py_func(
            noop_func, inputs, [x.dtype for x in inputs],
            name=utils.function_name(op.fn))
    else:
        node_outputs = tf.py_func(
            utils.align_func(op.fn, op.output),
            inputs, tf.as_dtype(op.output.dtype),
            name=utils.function_name(op.fn))

        assign_view(signals, op.output, node_outputs)

    node_outputs = ([node_outputs] if isinstance(node_outputs, tf.Tensor) else
                    node_outputs)
    return node_outputs


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
        assign_view(signals, op.states[i], result[i + 1])

    # we need the control_dependencies to force the state update operators
    # to run (otherwise they look like unused nodes and get optimized out)
    with tf.control_dependencies([signals[s] for s in op.states]):
        output = assign_view(signals, op.output, result[0])

    return output


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

    return assign_view(signals, op.output, result, inc=op.mode == "inc")


def get_variable(tensor):
    """Trace a Tensor backwards to find the base Variable."""
    if tensor.dtype._is_ref_dtype:
        return tensor

    my_base = None
    for input in set(tensor.op.inputs):
        base = get_variable(input)
        if base is not None:
            if my_base is not None:
                raise BuildError("multiple base variables found for tensor")
            else:
                my_base = base

    return my_base


def assign_view(signals, dst, src, src_slice=Ellipsis, dst_slice=Ellipsis,
                inc=False):
    # TODO: optimize out slice(None,None,None)

    if isinstance(src, Signal):
        src = signals[src]
        if isinstance(src_slice, slice):
            src = src[src_slice]
        elif src_slice is not Ellipsis:
            # advanced indexing
            src = tf.gather(src, tf.constant(src_slice))

    dst_var = get_variable(signals[dst])
    assert dst_var is not None

    if DEBUG:
        print("assigning %s to %s" % (src, dst_var))

    if not dst.is_view and dst_slice is Ellipsis:
        if inc:
            result = tf.assign_add(dst_var, src)
        else:
            result = tf.assign(dst_var, src)
    else:
        # TODO: make sliced assignment work for multidimensional arrays
        assert dst.ndim == 1

        if dst_slice is Ellipsis:
            start = dst.elemoffset
            stride = dst.elemstrides[0]
            stop = dst.elemoffset + dst.size * dst.elemstrides[0]

            indices = tf.range(start, stop, stride)
        elif isinstance(dst_slice, slice):
            start = dst.elemoffset + dst_slice.start * dst.elemstrides[0]
            stride = dst.elemstrides[0] * dst_slice.step
            stop = dst.elemoffset + dst_slice.stop * dst.elemstrides[0]

            indices = tf.range(start, stop, stride)
        else:
            # advanced indexing
            indices = np.asarray(dst_slice)

            indices *= dst.elemstrides[0]
            indices += dst.elemoffset

            indices = tf.constant(indices)

        if DEBUG:
            print("indices", indices)

        if inc:
            result = tf.scatter_add(dst_var, indices, src)
        else:
            result = tf.scatter_update(dst_var, indices, src)

        # we also need to update the base signal, so that future operations
        # on the base get the updated value
        signals[dst.base] = result

    signals[dst] = result

    if DEBUG:
        print("result", signals[dst], dst)

    return result


class SignalDict(dict):
    """Map from Signal -> Tensor

    Takes care of view/base logic
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if isinstance(key, Signal) and key.is_view:
                # return a view on the base signal
                base = dict.__getitem__(self, key.base)

                if key.dtype != key.base.dtype:
                    base = tf.cast(base, key.dtype)

                if key.initial_value.ndim != key.base.ndim:
                    if key.size != key.base.size:
                        # TODO: support this
                        raise NotImplementedError(
                            "Slicing and reshaping the same signal is not "
                            "supported")

                    view = tf.reshape(base, key.shape)
                else:
                    offset = np.unravel_index(key.elemoffset, key.base.shape)
                    shape = np.asarray(key.shape)
                    strides = np.asarray(key.elemstrides)
                    end = offset + shape * strides

                    end_mask = np.int32(0)
                    for i, b in enumerate(end < 0):
                        if b:
                            end_mask += 1 << i

                    view = tf.strided_slice(
                        base, offset, end, strides, end_mask=end_mask)
                dict.__setitem__(self, key, view)
                return view
            else:
                raise

    def __setitem__(self, key, val):
        """Assign `val` to `key`.

        Unlike normal dicts, this means that you cannot add a new key
        to a SignalDict using __setitem__. This is by design, to avoid
        silent typos when debugging Simulator. Every key must instead
        be explicitly initialized with SignalDict.init.
        """

        assert key in self
        dict.__setitem__(self, key, val)

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s %s" % (repr(k), repr(self[k]))
                          for k in self])

    def init(self, signal):
        """Set up a permanent mapping from signal -> tensor."""
        if signal in self:
            raise SignalError("Cannot add signal twice")

        if signal.is_view:
            if signal.base not in self:
                self.init(signal.base)

            if DEBUG:
                print("init view of", self[signal.base], self[signal])

            # get a view onto the base data
            dict.__setitem__(self, signal, self[signal])
        else:
            name = utils.sanitize_name(signal.name)

            x = tf.Variable(signal.initial_value, name=name,
                            dtype=signal.dtype)

            if DEBUG:
                print("init base", x)

            dict.__setitem__(self, signal, x)

        return self[signal]
