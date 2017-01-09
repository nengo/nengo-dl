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


def greedy_planner(operators):
    # based on nengo_ocl greedy_planner

    # TimeUpdate is executed as part of the simulation loop, not part
    # of the step plan
    operators = [op for op in operators if not isinstance(op, TimeUpdate)]

    dependency_graph = operator_depencency_graph(operators)

    # map unscheduled ops to their direct predecessors and successors
    predecessors_of = {}
    successors_of = {}
    for op in operators:
        predecessors_of[op] = set()
        successors_of[op] = set()
    for op, dests in iteritems(dependency_graph):
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
            if mergeable(op, chosen):
                # add op
                chosen += (op,)

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

    if DEBUG:
        print("PLAN")
        print("\n".join([str(x) for x in merged_ops]))

    # note: -1 because we skip TimeUpdate
    assert len(operators) == sum(len(p[1]) for p in merged_ops)

    return merged_ops


# TODO: add a "noop" planner for testing/debugging

def mergeable(op, chosen_ops):
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
        # if sig_map[s0].key != sig_map[s1].key:
        #     return False
        if s0.dtype != s1.dtype:
            return False
        if s0.base.shape[1:] != s1.base.shape[1:]:
            return False

        # note: here we are comparing their display shapes (e.g., the
        # shape after any reshape operations), not the shape of
        # the base arrays
        if s0.shape[1:] != s1.shape[1:]:
            return False

    if isinstance(op, SlicedCopy):
        # can't merge incs and updates
        if op.inc != c.inc:
            return False
    elif isinstance(op, (DotInc, ElementwiseInc)):
        # for these operations we also enforce that the first dimensions
        # match (we know all the other dimensions match due to checks above).
        # this allows us to stack all the arguments into continuous array
        # array blocks, allowing for more efficient multiplication (mainly
        # because it allows us to take advantage of broadcasting)
        for s0, s1 in zip(op.all_signals, c.all_signals):
            shape0 = s0.shape[0] if s0.shape != () else 1
            shape1 = s1.shape[0] if s1.shape != () else 1
            if shape0 != shape1:
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
            if len(op.states) > 0:
                dtype = op.states[0].dtype
                shape = op.states[0].base.shape[1:]
                if np.any([s.dtype != dtype or s.base.shape[1:] != shape
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

    return True


def noop_order_signals(plan, **kwargs):
    all_signals = list(set([s.base for op_type, ops in plan for op in ops
                            for s in op.all_signals]))

    # all_signals = [s.base for op_type, ops in plan for op in ops
    #                for s in op.all_signals]
    # all_signals = [s for i, s in enumerate(all_signals)
    #                if s not in all_signals[:i]]
    return all_signals, plan


def order_signals(plan, n_passes=10):
    """Orders signals and operators to try to structure reads in contiguous
    blocks."""

    # figure out all the read blocks in the plan (in theory we would like each
    # block to become a contiguous chunk in the base array)
    read_blocks = []
    for op_type, ops in plan:
        if op_type == SimNeurons:
            # add state signals to the reads
            for op in ops:
                op.reads += op.states

        for i in range(len(ops[0].reads)):
            read_blocks += [set(op.reads[i].base for op in ops)]

    # get rid of duplicate read blocks
    read_blocks = [
        x for i, x in enumerate(read_blocks) if not
        any([len(x.union(y)) == len(x) == len(y) for y in read_blocks[:i]])]

    # sort by the size of the block (descending order)
    read_blocks = sorted(
        read_blocks, key=lambda b: np.sum([s.size for s in b]))
    read_blocks = read_blocks[::-1]

    # get all the base signals
    all_signals = list(set([s.base for op_type, ops in plan for op in ops
                            for s in op.all_signals]))

    # mark the signals according to which blocks they are in
    signal_blocks = {s: tuple(s in b for b in read_blocks)
                     for s in all_signals}

    if len(read_blocks) == 0:
        # no reads, so nothing to reorder
        return plan, all_signals

    if DEBUG:
        print("all signals")
        print(all_signals)
        print("read blocks")
        print(read_blocks)
        print("signal blocks")
        print(signal_blocks)

    # basically we're going to repeatedly iterate over two steps
    # 1) order the ops within a group according to the order of their
    #    read signals
    # 2) order/group the signals according to operator groups

    # we iterate through the groups in order of increasing size, so that
    # if later reorderings (in 2) break a previous order, we tend to leave the
    # largest blocks in order.
    # similarly, we do multiple passes through this sorting because if a
    # later group breaks the ordering of an earlier one, it is possible that
    # on the next pass we can put the first group back into a valid ordering
    # based on the order established by the later group.

    sorted_reads = [(ops, i) for _, ops in plan
                    for i in range(len(ops[0].reads))]
    # note: we're sorting by the view size, not the base size
    sorted_reads = sorted(
        sorted_reads, key=lambda p: np.sum([op.reads[p[1]].size
                                            for op in p[0]]))

    if DEBUG:
        print("sorted reads")
        print("\n".join(str(x) for x in sorted_reads))

    # reorder the signals into continuous blocks (giving higher priority
    # to larger groups)
    all_signals = hamming_sort(all_signals, signal_blocks)

    if DEBUG:
        print("hamming sorted signals")
        print(all_signals)

    new_plan = {ops: ops for _, ops in plan}
    for n in range(n_passes):
        # TODO: detect if ops order didn't change (early termination)
        # TODO: every few iterations, eliminate the smallest unsatisfied block
        # TODO: do multiple passes actually help?
        if DEBUG:
            print("======== pass %d ========" % n)

        # reorder ops by signal order. this leaves the
        # hamming sort order unchanged, but it could upset the order
        # established in earlier read blocks. this is why we
        # have multiple passes, in the hopes that they can converge to
        # a mutual order.
        new_plan, all_signals = sort_ops_by_signals(
            sorted_reads, all_signals, new_plan, signal_blocks)

        if DEBUG:
            print("resorted ops")
            print("\n".join([str(x) for x in new_plan.values()]))

        if DEBUG:
            print("reordered signals")
            print(all_signals)

    # error checking
    assert len(new_plan) == len(plan)
    for ops, new_ops in new_plan.items():
        assert len(ops) == len(new_ops)
        for op in ops:
            assert op in new_ops

    if DEBUG:
        print("final sorted signals")
        print(all_signals)
        print("new plan")
        print("\n".join([str((op_type, new_plan[ops]))
                         for op_type, ops in plan]))

    return all_signals, [(op_type, new_plan[ops]) for op_type, ops in plan]


def hamming_sort(signals, blocks):
    signals = np.asarray(signals)
    blocks = np.asarray([blocks[s] for s in signals])

    sorted_signals = []
    curr_block = [False for _ in range(blocks.shape[1])]
    curr_block[0] = True
    active_block = None

    if DEBUG:
        print("hamming sort:")

    while True:
        if DEBUG:
            print("curr_block", curr_block)

        dists = np.sum(curr_block != blocks, axis=-1)

        # add any matching blocks to the sorted list
        zero_dists = dists == 0
        sorted_signals += [s for s in signals[zero_dists]]

        signals = signals[~zero_dists]
        blocks = blocks[~zero_dists]

        if len(signals) == 0:
            break

        # pick which block to go to next

        # start by picking all the blocks that are a continuation of the
        # active block (this is to give us some persistence, so it doesn't
        # jump around too much)
        if active_block is None:
            active_block = np.argmax(curr_block)

        next_blocks = blocks[blocks[:, active_block]]
        if len(next_blocks) == 0:
            # there are no remaining blocks that are a continuation of the
            # current block, so they're all up for grabs
            next_blocks = blocks
            active_block = None

        # get the unique blocks
        next_blocks = np.vstack(set(tuple(b) for b in next_blocks))

        if DEBUG:
            print("active block", active_block)
            print("next blocks")
            print(next_blocks)

        # then within all the blocks that are a potential continuation,
        # pick the ones with the smallest hamming distance
        next_dists = np.sum(next_blocks != curr_block, axis=1)
        next_blocks = next_blocks[next_dists == np.min(next_dists)]

        if DEBUG:
            print("hamming filter")
            print(next_blocks)

        # within all the blocks that have the same hamming distance, pick the
        # next block that matches along the highest indices
        for i in range(blocks.shape[1]):
            if len(next_blocks) == 1:
                break

            if np.any(np.logical_and(next_blocks[:, i], curr_block[i])):
                next_blocks = next_blocks[next_blocks[:, i]]

        # within the blocks that match curr_block equally, pick the next block
        # containing the largest read blocks
        for i in range(blocks.shape[1]):
            if len(next_blocks) == 1:
                break

            if np.any(next_blocks[:, i]):
                next_blocks = next_blocks[next_blocks[:, i]]
        else:
            raise BuildError("Something is wrong in hamming sort, no unique "
                             "next block")

        curr_block = next_blocks[0]

    return sorted_signals


def sort_signals_by_ops(sorted_reads, signals, new_plan, blocks):
    """Try to rearrange `signals` so that they are in the same order as
    operator reads."""

    for old_ops, read_block in sorted_reads:
        if DEBUG:
            print("sorting signals", [op.reads[read_block]
                                      for op in new_plan[old_ops]])
            print(read_block, new_plan[old_ops])

        if len(old_ops) == 1:
            continue

        ops = new_plan[old_ops]
        op_reads = [op.reads[read_block].base for op in ops]

        # iterative approach, because we want signals to bubble up as close as
        # possible to sorted order (even if they can't be fully sorted), so
        # that there are minimal changes to op order
        prev_index = signals.index(op_reads[0])
        for i in range(1, len(op_reads)):
            r_index = signals.index(op_reads[i])

            # if DEBUG:
            #     print("sort step")
            #     print(op_reads[i])
            #     print("prev", prev_index)
            #     print("start", r_index)

            move = False
            while r_index <= prev_index:
                r_index += 1
                move = True

                if (r_index > 0 and r_index < len(signals) and
                            blocks[signals[r_index]] !=
                            blocks[signals[r_index - 1]]):
                    break

            # if DEBUG:
            #     print("end", r_index)

            if move:
                pre = [x for x in signals[:r_index] if x is not op_reads[i]]
                post = signals[r_index:]
                signals = pre + [op_reads[i]] + post
                prev_index = r_index - 1
            else:
                prev_index = r_index

        if DEBUG:
            print("sorted signals", signals)

    return signals


def sort_ops_by_signals(sorted_reads, signals, new_plan, blocks):
    """Rearrange ops to match the order of signals.

    Note: this could screw up the order of other read blocks associated with
    these ops.  We iterate through the read blocks in increasing size so
    that the largest blocks win out.
    """

    if DEBUG:
        print("sort ops by signals")

    for old_ops, read_block in sorted_reads:
        if DEBUG:
            print("sorting ops", new_plan[old_ops])
            print("by", [op.reads[read_block] for op in new_plan[old_ops]])

        if len(old_ops) == 1:
            # then we have nothing to sort
            continue

        ops = new_plan[old_ops]

        # check if op reads are contiguous
        # reads = [op.reads[read_block].base for op in ops]
        # indices = sorted([signals.index(s) for s in reads])
        # if indices != list(range(np.min(indices), np.max(indices) + 1)):
        #     continue

        # note: the key is (signal index, view offset), so ops will be
        # sorted first by the order of the signals in the list, then by
        # the order of the views within each signal
        sorted_ops = sorted(
            ops, key=lambda op: (signals.index(op.reads[read_block].base),
                                 op.reads[read_block].elemoffset))

        new_plan[old_ops] = tuple(sorted_ops)

        if DEBUG:
            print("sorted ops")
            print(new_plan[old_ops])

        signals = sort_signals_by_ops(
            [x for x in sorted_reads
             if x[0] == old_ops and x[1] != read_block],
            signals, new_plan, blocks)

    return new_plan, signals


def create_signals(sigs, plan, float_type=np.float32):
    """Groups signals together into larger arrays, and represent each
    individual signal as a slice into that array."""

    base_arrays = OrderedDict()
    sig_map = {}

    # create all the base signals
    for sig in sigs:
        assert sig not in sig_map
        assert not sig.is_view

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

        if key in base_arrays:
            base_arrays[key] = np.concatenate(
                (base_arrays[key], initial_value), axis=0)
        else:
            base_arrays[key] = initial_value

        indices = np.arange(base_arrays[key].shape[0] - shape[0],
                            base_arrays[key].shape[0])

        sig_map[sig] = signals.TensorSignal(indices, key, label=sig.name)

    # add any signal views to the sig_map
    for _, ops in plan:
        for op in ops:
            for sig in op.all_signals:
                if sig.is_view:
                    if sig.initial_value.ndim != sig.base.ndim:
                        # reshape view
                        if sig.size != sig.base.size:
                            # TODO: support this?
                            raise NotImplementedError(
                                "Slicing and reshaping the same signal is not "
                                "supported")

                        sig_map[sig] = sig_map[sig.base].reshape(sig.shape)
                    else:
                        # slice view
                        assert np.all([x == 1 for x in sig.elemstrides[1:]])

                        start = sig.elemoffset
                        stride = sig.elemstrides[0]
                        stop = start + sig.size * stride
                        if stop < 0:
                            stop = None

                        sig_map[sig] = sig_map[sig.base][
                            slice(start, stop, stride)]
                else:
                    assert sig in sig_map

    # error checking
    for sig, tensor_sig in sig_map.items():

        if tensor_sig.shape != (sig.shape if sig.shape != () else (1,)):
            raise BuildError("TensorSignal shape %s does not match Signal "
                             "shape %s" % (tensor_sig.shape, sig.shape))
        if not np.allclose(base_arrays[tensor_sig.key][tensor_sig.indices],
                           sig.initial_value):
            raise BuildError("TensorSignal values don't match Signal values")

    # copy the base arrays to make them contiguous in memory
    for k in base_arrays:
        base_arrays[k] = np.array(base_arrays[k])

    if DEBUG:
        print("base arrays")
        print("\n".join([str((k, v.shape)) for k, v in base_arrays.items()]))

    return base_arrays, sig_map
