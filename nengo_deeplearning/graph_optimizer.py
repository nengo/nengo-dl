from collections import OrderedDict
import copy

from nengo.builder.operator import (TimeUpdate, SimPyFunc, SlicedCopy, DotInc,
                                    ElementwiseInc, Copy)
from nengo.builder.neurons import SimNeurons
from nengo.builder.processes import SimProcess
from nengo.exceptions import BuildError
from nengo.utils.compat import iteritems
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph
import numpy as np

from nengo_deeplearning import signals, neurons, processes, builder, DEBUG


def greedy_planner(operators):
    """Combine mergeable operators into groups that will be executed as a
     single computation.

     Parameters
     ----------
     operators : list of nengo.builder.operator.Operator
        all the `nengo` operators in a model (unordered)

    Returns
    -------
    list of tuple of `nengo.builder.operator.Operator`
        operators combined into mergeable groups and in execution order

    Notes
    -----
    Originally based on `nengo_ocl` greedy planner
     """

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
    # have been scheduled).
    # initialize it with the ops that have no predecessors
    available = [op for op, dep in iteritems(predecessors_of) if len(dep) == 0]

    plan = []
    groups = []
    while len(predecessors_of) > 0:
        # sort the available ops into mergeable groups
        for op in available:
            for g in groups:
                if mergeable(op, g):
                    g += [op]
                    break
            else:
                groups += [[op]]

        if len(groups) == 0:
            raise BuildError("Cycle detected during graph optimization")

        # pick the group that has the largest number of available ops
        groups = sorted(groups, key=lambda x: len(x))
        chosen = groups[-1]
        groups = groups[:-1]

        plan += [tuple(chosen)]

        # update predecessors and successors of remaining ops, and check for
        # any newly available ops
        available = []
        for op in chosen:
            for op2 in successors_of[op]:
                preds = predecessors_of[op2]
                preds.remove(op)
                if len(preds) == 0:
                    available += [op2]
            del predecessors_of[op]
            del successors_of[op]

    if DEBUG:
        print("GREEDY PLAN")
        print("\n".join([str(x) for x in plan]))

    # note: -1 because we skip TimeUpdate
    assert len(operators) == sum(len(ops) for ops in plan)

    return plan


def mergeable(op, chosen_ops):
    """Check if the given op can be merged with the candidate group

    Parameters
    ----------
    op : nengo.builder.operator.Operator
        the operator to be merged
    chosen_ops : list of nengo.builder.operator.Operator
        the operator group to be merged in to

    Returns
    -------
    bool
        True if `op` can be merged into `chosen_ops`, else False
    """

    if len(chosen_ops) == 0:
        return True

    # note: we only need to check against the first item in the list,
    # since we know the rest all match
    c = chosen_ops[0]

    # must share the same builder
    if builder.Builder.builders[type(op)] != builder.Builder.builders[type(c)]:
        return False

    # sets/incs/reads/updates must all match
    if (len(op.sets) != len(c.sets) or len(op.incs) != len(c.incs) or
            len(op.reads) != len(c.reads) or
            len(op.updates) != len(c.updates)):
        return False

    for s0, s1 in zip(op.all_signals, c.all_signals):
        # dtype of signals must match
        if s0.dtype != s1.dtype:
            return False

        # shape of signal base must match on all axes > 0
        if s0.base.shape[1:] != s1.base.shape[1:]:
            return False

        # note: here we are comparing their display shapes (e.g., the
        # shape after any reshape operations), not the shape of
        # the base arrays
        if s0.shape[1:] != s1.shape[1:]:
            return False

    # operator-specific checks
    if isinstance(op, (SlicedCopy, Copy)):
        # can't merge incs and updates
        if getattr(op, "inc", False) != getattr(c, "inc", False):
            return False
    elif isinstance(op, (DotInc, ElementwiseInc)):
        # for these operations we also enforce that the first dimensions
        # match (we know all the other dimensions match due to checks above).
        # this allows us to stack all the arguments into continuous array
        # blocks, allowing for more efficient multiplication (mainly
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

            # all the states must have the same base (note: this is checking
            # different state signals within one op against each other, rather
            # than checking signals across ops, which we already did)
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


def tree_planner(operators):
    """Create merged execution plan through exhaustive tree search.

    Unlike `greedy_planner`, this is guaranteed to find the shortest plan.
    However, depending on the structure of the operator graph, it can take
    a long time to execute.

    Parameters
     ----------
     operators : list of nengo.builder.operator.Operator
        all the `nengo` operators in a model (unordered)

    Returns
    -------
    list of tuple of `nengo.builder.operator.Operator`
        operators combined into mergeable groups and in execution order
    """

    def shortest_plan(ops, successors_of, predecessors_of, cache):
        if DEBUG:
            print("shortest_plan")
            print(ops)

        if len(ops) <= 1:
            # normal termination
            return [ops] if len(ops) == 1 else []
        elif ops in cache:
            # we've already found the shortest path for this set of ops
            # (plans are markovian)
            return cache[ops]

        # get the groups that could be scheduled next
        free = [op for op in ops if len(predecessors_of[op]) == 0]

        if DEBUG:
            print("free", free)

        available = []
        for op in free:
            for i, group in enumerate(available):
                if mergeable(op, group):
                    available[i] += (op,)
                    break
            else:
                available += [(op,)]

        if DEBUG:
            print("available")
            print(available)

        if len(available) == 0:
            raise BuildError("Cycle detected during graph optimization")

        # check what the shortest plan is after selecting each available group
        shortest = None
        for group in available:
            pred = {k: copy.copy(v) for k, v in predecessors_of.items()}
            for op in group:
                for op2 in successors_of[op]:
                    pred[op2].remove(op)

            if DEBUG:
                print("selecting", group)

            result = shortest_plan(
                tuple(op for op in ops if op not in group),
                successors_of, pred, cache)

            if shortest is None or len(result) + 1 < len(shortest):
                shortest = [group] + result

                if DEBUG:
                    print("new shortest plan detected")
                    print(shortest)

        cache[ops] = shortest

        return shortest

    operators = [op for op in operators if not isinstance(op, TimeUpdate)]
    dependency_graph = operator_depencency_graph(operators)

    predecessors_of = {}
    successors_of = {}
    for op in operators:
        predecessors_of[op] = set()
        successors_of[op] = set()
    for op in operators:
        dests = dependency_graph[op]
        for op2 in dests:
            predecessors_of[op2].add(op)
        successors_of[op].update(dests)

    tmp = shortest_plan(tuple(operators), successors_of, predecessors_of, {})

    if DEBUG:
        print("TREE PLAN")
        print("\n".join([str(x) for x in tmp]))

    return tmp


def noop_planner(operators):
    operators = [op for op in operators if not isinstance(op, TimeUpdate)]

    dependency_graph = operator_depencency_graph(operators)

    return [(op,) for op in toposort(dependency_graph)]


def order_signals(plan, n_passes=10):
    """Orders signals and operators to try to structure reads in contiguous
    blocks.

    Parameters
    ----------
    plan : list of tuple of `nengo.builder.operator.Operator`
        operator execution plan (e.g., output from `greedy_planner`)
    n_passes : int
        number of repeated passes through the operator reordering stage

    Returns
    -------
    list of `nengo.builder.signal.Signal`
        signals organized into the order in which we want them arranged in
        memory
    list of tuple of `nengo.builder.operator.Operator`
        input plan with operators reordered within groups to align with order
        of signals
    """

    # figure out all the read blocks in the plan (in theory we would like each
    # block to become a contiguous chunk in the base array)
    read_blocks = []
    for ops in plan:
        if type(ops[0]) == SimNeurons:
            # state signals are technically reads as well, they just aren't
            # marked as such, so we add them to the reads list
            for op in ops:
                op.reads += op.states

        for i in range(len(ops[0].reads)):
            read_blocks += [set(op.reads[i].base for op in ops)]

    # get rid of duplicate read blocks
    read_blocks = [
        x for i, x in enumerate(read_blocks) if not
        any([len(x.union(y)) == len(x) == len(y) for y in read_blocks[:i]])]

    # sort by the size of the block (descending order)
    # TODO: we should give some bonus to duplicate read blocks (since they're
    # affecting multiple operator groups)
    read_blocks = sorted(
        read_blocks, key=lambda b: np.sum([s.size for s in b]))
    read_blocks = read_blocks[::-1]

    # get all the unique base signals
    all_signals = list(set([s.base for ops in plan for op in ops
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

    sorted_reads = [(ops, i) for ops in plan
                    for i in range(len(ops[0].reads))]
    # note: we're sorting by the view size, not the base size
    sorted_reads = sorted(
        sorted_reads, key=lambda p: np.sum([op.reads[p[1]].size
                                            for op in p[0]]))

    if DEBUG:
        print("sorted reads")
        print("\n".join(str(x) for x in sorted_reads))

    # reorder the signals into contiguous blocks (giving higher priority
    # to larger groups)
    all_signals = hamming_sort(all_signals, signal_blocks)

    if DEBUG:
        print("hamming sorted signals")
        print(all_signals)

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

    new_plan = {ops: ops for ops in plan}
    for n in range(n_passes):
        # TODO: detect if ops order didn't change (early termination)
        # TODO: every few iterations, eliminate the smallest unsatisfied block
        if DEBUG:
            print("======== pass %d ========" % n)

        # reorder ops by signal order. this leaves the overall
        # hamming sort block order unchanged.
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
        print("\n".join([str(new_plan[ops]) for ops in plan]))

    return all_signals, [new_plan[ops] for ops in plan]


def hamming_sort(signals, blocks):
    """Reorder signals using heuristics to try to place signals that are read
    by the same operators into adjacent positions (giving priority to larger
    blocks).

    Parameters
    ----------
    list of `nengo.builder.signal.Signal`
        the signals to be sorted
    dict of {Signal: tuple of bool}
        dictionary indicating which read blocks each signal is a part of
    """

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

        # add any matching blocks to the sorted list
        zero_dists = np.all(blocks == curr_block, axis=1)
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
            # pick the largest block in the current block to become the new
            # active block (note: the boolean block arrays are sorted from
            # largest to smallest, so argmax(curr_block) gives us the index of
            # the first True block in the list, which is the largest. this
            # ordering is used in several places.)
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
        for i in range(blocks.shape[1] + 1):
            if len(next_blocks) == 1:
                break

            if np.any(next_blocks[:, i]):
                next_blocks = next_blocks[next_blocks[:, i]]
        else:
            raise BuildError("Something is wrong in hamming sort, no unique "
                             "next block")

        curr_block = next_blocks[0]

    return sorted_signals


def sort_ops_by_signals(sorted_reads, signals, new_plan, blocks):
    """Rearrange operators to match the order of signals.

    Note: the same operators can be associated with multiple read blocks if
    they have multiple inputs, so rearranging the operators according to one
    of those blocks could mess up the order with respect to the other read
    block.  We iterate through the read blocks in increasing size so
    that the largest blocks win out.

    Parameters
    ----------
    sorted_reads : list of (tuple of `nengo.builder.operator.Operator`, int)
        the operators that form each read block, sorted by increasing size of
        the read block. in the case that a group of operators participate in
        multiple read blocks, the integer distinguishes which one of those
        inputs this block is associated with.
    signals : list of `nengo.builder.operator.Signal`
        signals that have been arranged into a given order by other parts
        of the algorithm
    new_plan : dict of {tuple of Operator: tuple of Operator}
        mapping from original operator group to the sorted operators
    blocks : dict of {Signal: tuple of bool}
        indicates which read blocks each signal participates in

    Returns
    -------
    new_plan : dict of {tuple of Operator: tuple of Operator}
        mapping from original operator group to the sorted operators
    signals : list of `nengo.builder.operator.Signal`
        signals list, possibly rearranged to match new operator order
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
        # TODO: do we want to do this? I don't think so, because there is
        # a chance that non-contiguous signals could become contiguous after
        # the sort
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

        # after sorting the operators, we then rearrange all the other read
        # blocks associated with this group of operators to match the new
        # order. note that this could make smaller (earlier) blocks out
        # of order, which will hopefully be fixed on future passes. however,
        # it means that larger (later) blocks will align themselves to this
        # order if possible
        signals = sort_signals_by_ops(
            [x for x in sorted_reads
             if x[0] == old_ops and x[1] != read_block],
            signals, new_plan, blocks)

    return new_plan, signals


def sort_signals_by_ops(sorted_reads, signals, new_plan, blocks):
    """Attempts to rearrange `signals` so that it is in the same order as
    operator reads, without changing the overall block order.

    Parameters
    ----------
    sorted_reads : list of (tuple of `nengo.builder.operator.Operator`, int)
        the operators that form each read block, sorted by increasing size of
        the read block. in the case that a group of operators participate in
        multiple read blocks, the integer distinguishes which one of those
        inputs this block is associated with.
    signals : list of `nengo.builder.operator.Signal`
        signals to be sorted
    new_plan : dict of {tuple of Operator: tuple of Operator}
        mapping from original operator group to the sorted operators
    blocks : dict of {Signal: tuple of bool}
        indicates which read blocks each signal participates in

    Returns
    -------
    list of `nengo.builder.operator.Signal`
        sorted signals
    """

    for old_ops, read_block in sorted_reads:
        if DEBUG:
            print("sorting signals", [op.reads[read_block]
                                      for op in new_plan[old_ops]])
            print(read_block, new_plan[old_ops])

        ops = new_plan[old_ops]
        op_reads = [op.reads[read_block].base for op in ops]

        if len(set(op_reads)) == 1:
            # only one read signal, so nothing to sort
            continue

        # iterative approach, because we want signals to bubble up as close as
        # possible to sorted order (even if they can't be fully sorted), so
        # that there are minimal changes to op order
        # TODO: do we actually want that? or if things can't be sorted fully
        # should we just not bother?
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

                if (0 < r_index < len(signals) and
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


def noop_order_signals(plan, **kwargs):
    """A version of `order_signals` that doesn't do any reordering, for
    debugging."""

    all_signals = list(set([s.base for ops in plan for op in ops
                            for s in op.all_signals]))
    return all_signals, plan


def create_signals(sigs, plan, float_type):
    """Groups signals together into larger arrays, and represent each
    individual signal as a slice into that array.

    Parameters
    ----------
    signals : list of `nengo.builder.operator.Signal`
        base signals arranged into the order in which they should reside in
        memory (e.g., output from `order_signals`)
    plan : list of tuple of `nengo.builder.operator.Operator`
        operator execution plan (only used to get a list of all the operators)
    float_type : np.float32 or np.float64
        floating point precision to use for signals

    Returns
    -------
    base_arrays : dict of {(dtype, shape) : :class:`~numpy:numpy.ndarray`}
        combined arrays, containing the initial values for all signals
    sig_map : dict of {`nengo.builder.signal.Signal`:
                       :class:`.signals.TensorSignal}
        mapping from `nengo` `Signals` to `nengo_dl` `TensorSignals` (views
        into the base arrays)
    """

    # TODO: we don't need to separate base arrays by shape, we can just flatten
    # them all, and keep track of the shape inside the TensorSignal

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

        # key used to map signals to base arrays
        key = (dtype, (None,) + shape[1:])

        initial_value = sig.initial_value.astype(dtype, copy=False)

        # broadcast scalars up to full size
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
    all_signals = [sig for ops in plan for op in ops for sig in op.all_signals]
    for sig in all_signals:
        if sig.is_view:
            if sig.initial_value.ndim != sig.base.ndim:
                # reshape view
                if sig.size != sig.base.size:
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

                sig_map[sig] = sig_map[sig.base][slice(start, stop, stride)]
        else:
            # if it isn't a view, the signal should already be in sig_map
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
