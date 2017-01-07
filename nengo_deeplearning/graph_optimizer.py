from collections import defaultdict, OrderedDict

from nengo.builder.operator import (TimeUpdate, SimPyFunc, SlicedCopy, DotInc,
                                    ElementwiseInc)
from nengo.builder.neurons import SimNeurons
from nengo.builder.processes import SimProcess
from nengo.exceptions import BuildError
from nengo.utils.compat import iteritems
from nengo.utils.simulator import operator_depencency_graph
import numpy as np

from nengo_deeplearning import signals, neurons, processes, DEBUG


def create_op_signals(operators, float_type=np.float32):
    """Groups signals together into larger arrays, and represent each
    individual signal as a slice into that array."""

    base_sigs = OrderedDict()
    sig_map = {}

    def get_sig(sig):
        if DEBUG:
            print("getting signal for %s" % sig)

        if sig in sig_map:
            pass
        elif sig.is_view:
            if sig.initial_value.ndim != sig.base.ndim:
                # reshape view
                if sig.size != sig.base.size:
                    # TODO: support this?
                    raise NotImplementedError(
                        "Slicing and reshaping the same signal is not "
                        "supported")

                sig_map[sig] = get_sig(sig.base).reshape(sig.shape)
            else:
                # slice view
                assert np.all([x == 1 for x in sig.elemstrides[1:]])

                start = sig.elemoffset
                stride = sig.elemstrides[0]
                stop = start + sig.size * stride
                if stop < 0:
                    stop = None

                sig_map[sig] = get_sig(sig.base)[slice(start, stop, stride)]
        else:
            if sig.dtype in (np.float32, np.float64):
                dtype = float_type
            elif sig.dtype in (np.int32, np.int64):
                dtype = np.int32
            else:
                raise NotImplementedError

            # resize scalars to length 1 vectors
            shape = sig.shape if sig.shape != () else (1,)

            key = (dtype, (None,) + shape[1:])

            initial_value = sig.initial_value.astype(dtype, copy=False)

            if initial_value.shape != shape:
                initial_value = np.resize(initial_value, shape)

            if key in base_sigs:
                base_sigs[key] = np.concatenate(
                    (base_sigs[key], initial_value), axis=0)
            else:
                base_sigs[key] = initial_value

            indices = np.arange(base_sigs[key].shape[0] - shape[0],
                                base_sigs[key].shape[0])

            sig_map[sig] = signals.TensorSignal(indices, key, label=sig.name)

        # error checking
        tensor_sig = sig_map[sig]
        if tensor_sig.shape != (sig.shape if sig.shape != () else (1,)):
            raise BuildError("TensorSignal shape %s does not match Signal "
                             "shape %s" % (tensor_sig.shape, sig.shape))
        if not np.allclose(base_sigs[tensor_sig.key][tensor_sig.indices],
                           sig.initial_value):
            raise BuildError("TensorSignal values don't match Signal values")

        return tensor_sig

    for op in operators:
        if DEBUG:
            print("---")
            print(op)

        if isinstance(op, TimeUpdate):
            continue
        # op.sets = [create_sig(s) for s in op.sets]
        # op.incs = [create_sig(s) for s in op.incs]
        # op.reads = [create_sig(s) for s in op.reads]
        # op.updates = [create_sig(s) for s in op.updates]
        for s in op.all_signals:
            get_sig(s)

    # copy the base arrays to make them contiguous in memory
    for k in base_sigs:
        base_sigs[k] = np.array(base_sigs[k])

    if DEBUG:
        print("base sigs")
        print("\n".join([str((k, v.shape)) for k, v in base_sigs.items()]))

    return base_sigs, sig_map


def greedy_planner(operators, sig_map):
    # based on nengo_ocl greedy_planner

    dependency_graph = operator_depencency_graph(operators)

    # map unscheduled ops to their direct predecessors and successors
    predecessors_of = {}
    successors_of = {}
    for op in operators:
        if isinstance(op, TimeUpdate):
            # TimeUpdate is executed as part of the while loop, not part of
            # the step plan
            continue

        predecessors_of[op] = set()
        successors_of[op] = set()
    for op, dests in iteritems(dependency_graph):
        if isinstance(op, TimeUpdate):
            continue

        for op2 in dests:
            predecessors_of[op2].add(op)
        successors_of[op].update(dests)

    # the ops in `available` are ready to be scheduled (all predecessors
    # have been scheduled). they're grouped by op type (so we can try to
    # merge all the operators with the same type)
    # initialize it with the ops that have no predecessors
    available = defaultdict(set)
    for op in (op for op, dep in iteritems(predecessors_of) if len(dep) == 0):
        available[type(op)].add(op)

    merged_ops = ()
    while len(predecessors_of) > 0:
        if len(available) == 0:
            raise BuildError("Cycle detected during graph optimization")

        # pick the type that has the largest number of available ops
        chosen_type = sorted(available.items(), key=lambda x: len(x[1]))[-1][0]
        candidates = available[chosen_type]

        # figure out which ops can be merged
        chosen = ()

        for op in candidates:
            if mergeable(op, chosen, sig_map):
                # add op
                chosen += (op,)
                # inc_indices.update([x for s in op.incs for x in s.indices])

        assert len(chosen) > 0
        merged_ops += ((chosen_type, chosen),)

        # update predecessors and successors of remaining ops
        available[chosen_type].difference_update(chosen)
        if len(available[chosen_type]) == 0:
            del available[chosen_type]

        for op in chosen:
            for op2 in successors_of[op]:
                preds = predecessors_of[op2]
                preds.remove(op)
                if len(preds) == 0:
                    available[type(op2)].add(op2)
            del predecessors_of[op]
            del successors_of[op]

    # note: -1 because we skip TimeUpdate
    assert len(operators) - 1 == sum(len(p[1]) for p in merged_ops)

    if DEBUG:
        print("PLAN")
        print("\n".join([str(x) for x in merged_ops]))

    return merged_ops

# TODO: add a thing that tries to rearrange arrays to minimize the number
# of gathers needed


# TODO: add a "noop" planner for testing/debugging

def mergeable(op, chosen_ops, sig_map):
    # check if the ops share the same shape and dtype for input/output
    # signals
    if len(chosen_ops) == 0:
        return True

    # note: we only need to check against the first item in the list,
    # since we know the rest all match
    c = chosen_ops[0]

    if (len(op.sets) != len(c.sets) or len(op.incs) != len(c.incs) or
                len(op.reads) != len(c.reads) or
                len(op.updates) != len(c.updates)):
        return False

    for s0, s1 in zip(op.all_signals, c.all_signals):
        if sig_map[s0].key != sig_map[s1].key:
            return False

        # note: here we are comparing their display shapes (e.g., the
        # shape after any reshape operations), not the shape of
        # the base arrays
        if sig_map[s0].shape[1:] != sig_map[s1].shape[1:]:
            return False

    if isinstance(op, SlicedCopy):
        # can't merge incs and updates
        if op.inc != c.inc:
            return False
    elif isinstance(op, SimPyFunc):
        # for these we need to make a special check that the functions
        # all do/do not get time as input, otherwise we could end
        # up confusing a node that only gets a scalar float input with
        # a node that only gets time as input
        if op.t != c.t:
            return False
    elif isinstance(op, SimNeurons):
        if type(c.neurons) in neurons.TF_NEURON_IMPL:
            # for neuron types with a custom tensorflow implementation,
            # the types must match exactly
            if type(c.neurons) != type(op.neurons):
                return False
        else:
            # we can't merge generic with custom types
            if type(op.neurons) in neurons.TF_NEURON_IMPL:
                return False

            # all the states must have the same base (note: this is
            # checking different state signals within one op against
            # each other, rather than checking signals across ops)
            if np.any([sig_map[op.states[0]].key != sig_map[s].key
                       for s in op.states]):
                return False
    elif isinstance(op, SimProcess):
        # as with SimNeurons, we can merge ops if they have a custom
        # implementation, or merge generic processes, but can't mix
        # the two

        if type(c.process) in processes.TF_PROCESS_IMPL:
            if type(c.process) != type(op.process):
                return False
        else:
            if type(op.process) in processes.TF_PROCESS_IMPL:
                return False

        # processes must also have the same mode
        if op.mode != c.mode:
            return False
    elif isinstance(op, (DotInc, ElementwiseInc)):
        # for these operations we also enforce that the first dimensions
        # match (we know all the other dimensions match due to checks above).
        # this allows us to stack all the arguments into continuous array
        # array blocks, allowing for more efficient multiplication (mainly
        # because it allows us to take advantage of broadcasting)
        for s0, s1 in zip(op.all_signals, c.all_signals):
            if sig_map[s0].shape[0] != sig_map[s1].shape[0]:
                return False

    return True
