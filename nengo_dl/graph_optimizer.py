from collections import OrderedDict, defaultdict
import logging

from nengo.synapses import Lowpass
from nengo.builder.operator import SimPyFunc, ElementwiseInc, DotInc, Reset
from nengo.builder.neurons import SimNeurons
from nengo.builder.processes import SimProcess
from nengo.exceptions import BuildError
from nengo.utils.compat import iteritems
from nengo.utils.graphs import toposort

try:
    from nengo.utils.simulator import operator_dependency_graph
except ImportError:
    # TODO: remove this and bump nengo version once there is a release
    from nengo.utils.simulator import (
        operator_depencency_graph as operator_dependency_graph)
import numpy as np

from nengo_dl import (signals, processes, builder, tensor_node, operators,
                      learning_rules, neurons)

logger = logging.getLogger(__name__)


def mergeable(op, chosen_ops):
    """Check if the given op can be merged with the candidate group

    Parameters
    ----------
    op : :class:`~nengo:nengo.builder.Operator`
        the operator to be merged
    chosen_ops : list of :class:`~nengo:nengo.builder.Operator`
        the operator group to be merged in to

    Returns
    -------
    bool
        True if ``op`` can be merged into ``chosen_ops``, else False
    """

    if len(chosen_ops) == 0:
        return True

    # note: we only need to check against the first item in the list,
    # since we know the rest all match
    c = next(iter(chosen_ops))

    # must share the same builder
    if builder.Builder.builders[type(op)] != builder.Builder.builders[type(c)]:
        return False

    # sets/incs/reads/updates must all match
    if (len(op.sets) != len(c.sets) or len(op.incs) != len(c.incs) or
            len(op.reads) != len(c.reads) or
            len(op.updates) != len(c.updates)):
        return False

    # signals must be mergeable into the same base array
    for s0, s1 in zip(op.all_signals, c.all_signals):
        # dtype of signals must match
        if s0.dtype != s1.dtype:
            return False

        # shape of signal base must match on all axes > 0
        if s0.base.shape[1:] != s1.base.shape[1:]:
            return False

        # display shape must also match (since we need the shape to be well
        # defined when we combine the signals)
        if s0.shape[1:] != s1.shape[1:]:
            return False

        # trainable/minibatched must match
        if s0.trainable != s1.trainable or s0.minibatched != s1.minibatched:
            return False

    # operator-specific checks
    if isinstance(op, ElementwiseInc):
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
    elif isinstance(op, DotInc):
        # if the matrix (A) is minibatched, then the first dimensions need
        # to match up (to allow us to transpose the dimensions)
        if op.A.minibatched:
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
        # neuron ops must all have the same type
        if type(c.neurons) != type(op.neurons):
            return False
    elif isinstance(op, SimProcess):
        # we can merge ops if they have a custom implementation, or merge
        # generic processes, but can't mix the two

        if type(c.process) in processes.SimProcessBuilder.TF_PROCESS_IMPL:
            if type(c.process) != type(op.process):
                return False
        elif type(op.process) in processes.SimProcessBuilder.TF_PROCESS_IMPL:
            return False

        # processes must also have the same mode
        if op.mode != c.mode:
            return False
    elif isinstance(op, tensor_node.SimTensorNode):
        # not possible to merge TensorNodes, since each one can be performing
        # an entirely different function. and unlike SimPyFunc, there is no
        # point trying to execute all those functions at once, because they're
        # already integrated into the Tensorflow graph.
        return False
    elif isinstance(op, (learning_rules.SimVoja, learning_rules.SimOja,
                         learning_rules.SimBCM)):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product
        attr = ("pre_decoded" if isinstance(op, learning_rules.SimVoja) else
                "pre_filtered")
        if getattr(op, attr).shape[0] != getattr(c, attr).shape[0]:
            return False

    return True


def greedy_planner(operators):
    """Combine mergeable operators into groups that will be executed as a
    single computation.

    Parameters
    ----------
    operators : list of :class:`~nengo:nengo.builder.Operator`
        all the ``nengo`` operators in a model (unordered)

    Returns
    -------
    list of tuple of :class:`~nengo:nengo.builder.Operator`
        operators combined into mergeable groups and in execution order

    Notes
    -----
    Originally based on ``nengo_ocl`` greedy planner
    """

    dependency_graph = operator_dependency_graph(operators)

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

    logger.debug("GREEDY PLAN")
    logger.debug("\n" + "\n".join([str(x) for x in plan]))

    assert len(operators) == sum(len(ops) for ops in plan)

    return plan


def tree_planner(op_list, max_depth=3):
    """Create merged execution plan through exhaustive tree search.

    The ``max_depth`` parameter scales the planner between full tree search
    and greedy search.  ``max_depth==1`` is equivalent to
    :func:`.greedy_planner`, and ``max_depth==len(op_list)`` is full tree
    search (guaranteed to find the optimal plan, but likely very slow).

    Parameters
    ----------
    op_list : list of :class:`~nengo:nengo.builder.Operator`
        all the ``nengo`` operators in a model (unordered)
    max_depth : int, optional
        the planner will search this many steps ahead before selecting which
        group to schedule next

    Returns
    -------
    list of tuple of :class:`~nengo:nengo.builder.Operator`
        operators combined into mergeable groups and in execution order
    """

    def shortest_plan(selected, successors_of, predecessors_of, cache,
                      max_depth, available):
        """Recursively check what the shortest plan is after selecting each
        available group."""

        shortest = (None, len(successors_of) + 1)
        nonempty_available = [x for x in enumerate(available) if len(x[1]) > 0]
        n_remaining = len(successors_of) - len(selected)
        for i, group in nonempty_available:
            new_len = n_remaining - len(group)

            if max_depth == 1 or new_len == 0:
                # we've reached the end, so just return the number
                # of remaining operators after selecting this group
                result, length = [], new_len
            else:
                # update the selected ops after adding group
                new_selected = selected | group

                try:
                    # check if we've already computed the shortest path
                    # for the selected ops and depth
                    result, length = cache[max_depth - 1][new_selected]

                except KeyError:
                    # update the list of available items after selecting
                    # this group
                    available[i] = set()
                    successors = [x for op in group for x in successors_of[op]]
                    for op in successors:
                        predecessors_of[op] -= 1

                        if predecessors_of[op] == 0:
                            available[mergeable_cache[op]].add(op)

                    # recursively find the best plan on the remaining
                    # operators
                    result, length = shortest_plan(
                        new_selected, successors_of, predecessors_of, cache,
                        max_depth - 1, available)

                    # return the available list to its original state for
                    # the next group (note: this is faster than copying
                    # the list)
                    for op in successors:
                        predecessors_of[op] += 1

                        if predecessors_of[op] == 1:
                            available[mergeable_cache[op]].remove(op)
                    available[i] = group

            if length + 1 < shortest[1]:
                # new shortest path found
                shortest = ([tuple(group)] + result, length + 1)

        if shortest[0] is None:
            raise BuildError("Cycle detected during graph optimization")

        cache[max_depth][selected] = shortest

        return shortest

    # compute operator dependency graph
    successors_of = operator_dependency_graph(op_list)

    # convert operators to integer indices (to save memory and make
    # lookup faster)
    op_codes = {op: np.uint32(i) for i, op in enumerate(op_list)}
    successors_of = {op_codes[k]: set(op_codes[x] for x in v)
                     for k, v in successors_of.items()}

    # track the number of incoming edges to each operator
    predecessors_of = {}
    for op in successors_of:
        predecessors_of[op] = 0
    for op, dests in successors_of.items():
        for op2 in dests:
            predecessors_of[op2] += 1

    # precompute which operators are theoretically mergeable (this doesn't mean
    # we can actually merge these ops, since they may be dependent on one
    # another)
    mergeable_cache = [None for _ in op_list]
    groups = []
    for j, op in enumerate(op_list):
        for i, g in enumerate(groups):
            if mergeable(op, g):
                mergeable_cache[j] = i
                g.append(op)
                break
        else:
            mergeable_cache[j] = len(groups)
            groups.append([op])
    groups = [[op_codes[x] for x in g] for g in groups]

    # find the ops that could be scheduled next in each merge group
    available = [set(op for op in g if predecessors_of[op] == 0)
                 for g in groups]

    plan = []
    while len(successors_of) > 0:
        # find the best plan of the given depth
        short_plan, _ = shortest_plan(
            frozenset(), successors_of, predecessors_of,
            [{} for _ in range(max_depth + 1)], max_depth, available)

        # select the first item in that plan (i.e., the best group to select
        # after looking ahead for max_depth steps)
        selected = short_plan[0]
        plan.append(selected)

        # update the operator availability
        available[mergeable_cache[next(iter(selected))]] = set()
        for op in selected:
            for op2 in successors_of[op]:
                predecessors_of[op2] -= 1

                if predecessors_of[op2] == 0:
                    available[mergeable_cache[op2]].add(op2)

            del predecessors_of[op]
            del successors_of[op]

    # convert indices back to operators
    plan = [tuple(op_list[x] for x in g) for g in plan]

    logger.debug("TREE PLAN")
    logger.debug("\n".join([str(x) for x in plan]))

    return plan


def noop_planner(operators):
    """Orders operators into a valid execution order, but does not perform
    any merging.

    Parameters
    ----------
    operators : list of :class:`~nengo:nengo.builder.Operator`
        all the ``nengo`` operators in a model (unordered)

    Returns
    -------
    list of tuple of :class:`~nengo:nengo.builder.Operator`
        operators in execution order
    """

    dependency_graph = operator_dependency_graph(operators)
    plan = [(op,) for op in toposort(dependency_graph)]

    logger.debug("NOOP PLAN")
    logger.debug("\n" + "\n".join([str(x) for x in plan]))

    return plan


def transitive_planner(op_list):
    """Create merged execution plan through transitive closure construction.

    This is something like a middle ground between :func:`.greedy_planner` and
    :func:`.tree_planner`; it can improve simulation time over the greedy
    planner, but comes with potentially significant build time increases.

    Parameters
    ----------
    op_list : list of :class:`~nengo:nengo.builder.Operator`
        all the ``nengo`` operators in a model (unordered)

    Returns
    -------
    list of tuple of :class:`~nengo:nengo.builder.Operator`
        operators combined into mergeable groups and in execution order
    """

    # note: importing this here since it only exists in nengo 2.4.0
    from nengo.utils.graphs import BidirectionalDAG

    n_ele = len(op_list)
    merge_groups = {}
    dg = operator_dependency_graph(op_list)
    op_codes = {op: np.uint32(i) for i, op in enumerate(op_list)}
    dg = {op_codes[k]: set(op_codes[x] for x in v) for k, v in dg.items()}
    op_codes = None  # so it will get garbage collected
    dg = BidirectionalDAG(dg)

    # fail fast here if the op graph has cycles
    toposort(dg.forward)

    op_builders = [builder.Builder.builders[type(op)] for op in op_list]

    # sort operators by builder (we'll only be interested in one builder type
    # at a time, because we can't merge operators between builder types anyway)
    ops_by_type = defaultdict(set)
    for i, op in enumerate(op_list):
        ops_by_type[op_builders[i]].add(np.uint32(i))

    # heuristic ordering for builder types (earlier items in the list will
    # have higher priority, meaning that we will choose to merge those ops
    # and potentially break lower-priority groups)
    order = [
        operators.SparseDotIncBuilder, operators.ElementwiseIncBuilder,
        neurons.SimNeuronsBuilder, processes.SimProcessBuilder,
        operators.SimPyFuncBuilder,
        learning_rules.SimOjaBuilder, learning_rules.SimVojaBuilder,
        learning_rules.SimBCMBuilder,
        operators.CopyBuilder, operators.ResetBuilder,
        tensor_node.SimTensorNodeBuilder]

    for builder_type in order:
        if builder_type not in ops_by_type:
            # no ops of this type in the model
            continue

        ops = ops_by_type[builder_type]

        # compute transitive closure
        trans = [None for _ in range(n_ele)]
        transitive_closure_recurse(dg.forward, ops, trans, builder_type,
                                   op_builders, {})

        # reduce it to the elements we care about (ops of the current
        # builder type)
        trans = {i: v for i, v in enumerate(trans[:len(op_list)]) if i in ops}

        while len(trans) > 0:
            # find all the ops that have no downstream dependents
            available = set(k for k, v in trans.items() if len(v) == 0)

            # sort those ops into mergeable groups
            groups = []
            for op in available:
                for g in groups:
                    if mergeable(op_list[op], (op_list[g[0]],)):
                        g.append(op)
                        break
                else:
                    groups.append([op])

            # merge the groups
            for g in groups:
                dg.merge(g, n_ele)
                merge_groups[n_ele] = g
                n_ele += 1

            # remove those ops from the transitive closure
            for op in available:
                del trans[op]

            # remove those ops from the transitive closure of upstream ops
            # note: we first remove all the duplicate aliased transitive sets,
            # to reduce the number of set operations we need to do
            unique_trans = {id(v): v for v in trans.values()}
            for t in unique_trans.values():
                t -= available

        # trans_reverse = [None for _ in range(n_ele)]
        # transitive_closure_recurse(dg.backward, ops, trans_reverse,
        #                            builder_type, op_builders, cache)
        # trans_reverse = {i: v for i, v in
        #                  enumerate(trans_reverse[:len(op_list)]) if i in ops}
        # group = None
        # for op in toposort(trans, trans_reverse):
        #     if group is None:
        #         group = [op]
        #         continue
        #
        #     if mergeable(op_list[op], (op_list[group[0]],)) and all(
        #             x not in trans[op] for x in group):
        #         group.append(op)
        #     else:
        #         dg.merge(group, n_ele)
        #         merge_groups[n_ele] = group
        #         n_ele += 1
        #         group = [op]
        #
        # dg.merge(group, n_ele)
        # merge_groups[n_ele] = group
        # n_ele += 1

        del ops_by_type[builder_type]

    assert len(ops_by_type) == 0

    # toposort the merged graph to come up with execution plan
    plan = toposort(dg.forward)
    plan = [tuple(op_list[x] for x in merge_groups[group]) for group in plan]

    logger.debug("TRANSITIVE PLAN")
    logger.debug("\n" + "\n".join([str(x) for x in plan]))

    return plan


def transitive_closure_recurse(dg, ops, trans, builder_type, op_builders,
                               cache):
    """Computes the transitive closure for the given graph, restricted to the
    operators with the given builder type.

    Parameters
    ----------
    dg : dict of {int: set of int}
        dependency graph where ``dg[a] = {b, c}`` indicates that operators
        ``b`` and ``c`` are dependent on ``a``
    ops : list of int
        the operators for which we want to compute the transitive closure
    trans : dict of {int: set of int}
        the transitive closure for the graph (will be filled in-place)
    builder_type : type
        one of the ``nengo_dl`` build classes (e.g.,
        :class:`~.operators.CopyBuilder`), specifying the type of operators
        to include in the transitive closure
    op_builders : list of type
        the build class for each operator
    cache : dict of {frozenset of int: set of int}
        stores base sets which ``trans`` will reference (to reduce memory
        usage, since many elements in ``trans`` will have the same value)

    Notes
    -----
    This function uses ints to refer to operators, where the int indicates
    the index of the operator in the overall op list (this is done to save
    memory).  See :func:`.transitive_planner`.
    """

    for op in ops:
        if trans[op] is not None:
            # this can occur if the downstream calculations of an earlier
            # op filled in the value for this op
            continue

        todo = [x for x in dg[op] if trans[x] is None]
        transitive_closure_recurse(dg, todo, trans, builder_type, op_builders,
                                   cache)

        merged = set(
            x for x in dg[op] if x < len(op_builders) and
            op_builders[x] == builder_type)

        unique_posts = {id(trans[x]): trans[x] for x in dg[op]}

        if len(merged) == 0 and len(unique_posts) == 1:
            trans[op] = next(iter(unique_posts.values()))
        else:
            for x in unique_posts.values():
                merged |= x

            key = frozenset(merged)
            try:
                trans[op] = cache[key]
            except KeyError:
                trans[op] = merged
                cache[key] = merged


def order_signals(plan, n_passes=10):
    """Orders signals and operators to try to structure reads in contiguous
    blocks.

    Parameters
    ----------
    plan : list of tuple of :class:`~nengo:nengo.builder.Operator`
        operator execution plan (e.g., output from ``greedy_planner``)
    n_passes : int, optional
        number of repeated passes through the operator reordering stage

    Returns
    -------
    list of :class:`~nengo:nengo.builder.Signal`
        signals organized into the order in which we want them arranged in
        memory
    list of tuple of :class:`~nengo:nengo.builder.Operator`
        input plan with operators reordered within groups to align with order
        of signals
    """

    # get all the unique base signals
    all_signals = sorted(set([s.base for ops in plan for op in ops
                              for s in op.all_signals]), key=lambda s: s.name)

    # figure out all the read blocks in the plan (in theory we would like each
    # block to become a contiguous chunk in the base array)
    read_blocks = OrderedDict()

    # note: reads[op] contains all the signals that are inputs to op. this is
    # generally equivalent to op.reads, but there are some ops that also
    # require their set/inc/updates as input. we don't want to modify
    # op.reads itself, because then if you pass the same model to the
    # Simulator the operators keep getting modified in-place.
    reads = {}
    for ops in plan:
        for op in ops:
            reads[op] = [x for x in op.reads]
            if type(op) == SimNeurons:
                # state signals are technically reads as well, they just aren't
                # marked as such, so we add them to the reads list
                reads[op] += op.states
            elif type(op) == SimProcess and isinstance(op.process, Lowpass):
                # the lowpass op has to read the output value as well (unless
                # we get a scatter_mul working)
                reads[op] += op.updates

        # the ith input signal for each op in the op group is one read group
        # (note that we only care about bases, since those are the things we
        # are trying to order)
        for i in range(len(reads[ops[0]])):
            read_blocks[(ops, i)] = set(reads[op][i].base for op in ops)

    if len(read_blocks) == 0:
        # no reads, so nothing to reorder
        return all_signals, plan

    # get rid of duplicate read blocks
    duplicates = [
        [y for y in read_blocks.values() if x == y]
        for x in read_blocks.values()]
    sorted_blocks = [
        (x, len(duplicates[i])) for i, x in enumerate(read_blocks.values())
        if duplicates[i][0] is x]

    # sort by the size of the block (descending order)
    # note: we multiply by the number of duplicates, since read blocks that
    # are read by multiple op groups will have a proportionally larger impact
    # on performance
    sorted_blocks = sorted(
        sorted_blocks, key=lambda b: np.sum([s.size for s in b[0]]) * b[1])
    sorted_blocks = [sorted_blocks[i][0] for i in
                     range(len(sorted_blocks) - 1, -1, -1)]

    # figure out which read blocks each signal participates in
    signal_blocks = defaultdict(list)
    for i, b in enumerate(sorted_blocks):
        for s in b:
            signal_blocks[s].append(i)
    signal_blocks = {s: frozenset(b) for s, b in signal_blocks.items()}

    logger.debug("all signals")
    logger.debug(all_signals)
    logger.debug("sorted blocks")
    logger.debug(sorted_blocks)
    logger.debug("signal blocks")
    logger.debug(signal_blocks)

    # list of the ops in each read block, sorted by the size of that read block
    sorted_reads = sorted(
        read_blocks.keys(),
        key=lambda p: -sorted_blocks.index(read_blocks[p]))

    logger.debug("sorted reads")
    logger.debug("\n".join(str(x) for x in sorted_reads))

    # reorder the signals into contiguous blocks (giving higher priority
    # to larger groups)
    sort_idxs = hamming_sort(signal_blocks)
    all_signals = sorted(all_signals, key=lambda s: sort_idxs[s])

    logger.debug("hamming sorted signals")
    logger.debug(all_signals)

    # now we want to order the ops and signals within the blocks established
    # by the hamming sort

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
    sig_idxs = {s: i for i, s in enumerate(all_signals)}

    logger.debug("plan")
    logger.debug("\n" + "\n".join([str(x) for x in new_plan.values()]))
    logger.debug("signal indices")
    logger.debug(sig_idxs)

    for n in range(n_passes):
        # TODO: every few iterations, eliminate the smallest unsatisfied block?
        logger.debug("======== pass %d ========", n)

        # save previous plan/idxs, so we can check if they change for
        # early termination
        prev_plan = {k: v for k, v in new_plan.items()}
        prev_sig_idxs = sig_idxs  # note: no copy necessary

        # reorder ops by signal order. this leaves the overall
        # hamming sort block order unchanged.
        new_plan, sig_idxs = sort_ops_by_signals(
            sorted_reads, all_signals, sig_idxs, new_plan, signal_blocks,
            reads)

        logger.debug("resorted ops")
        logger.debug("\n" + "\n".join([str(x) for x in new_plan.values()]))

        logger.debug("reordered signal indices")
        logger.debug(sig_idxs)

        if (all([x == y for ops in plan
                 for x, y in zip(new_plan[ops], prev_plan[ops])]) and
                all([sig_idxs[s] == prev_sig_idxs[s] for s in all_signals])):
            # if the plan didn't change and the signals didn't change, then
            # there is no point in continuing (they're not going to change
            # in the future)
            logger.debug("early termination")
            break

    sorted_signals = sorted(all_signals, key=lambda s: sig_idxs[s])

    # error checking
    # make sure that overall signal block order didn't change
    for s, s2 in zip(all_signals, sorted_signals):
        if s in signal_blocks or s2 in signal_blocks:
            assert signal_blocks[s] == signal_blocks[s2]

    # make sure that all ops are present
    assert len(new_plan) == len(plan)
    for ops, new_ops in new_plan.items():
        assert len(ops) == len(new_ops)
        # for op in ops:
        #     assert op in new_ops

    logger.debug("final sorted signals")
    logger.debug(sorted_signals)
    logger.debug("new plan")
    logger.debug("\n" + "\n".join([str(x) for x in new_plan.values()]))

    return sorted_signals, [new_plan[ops] for ops in plan]


def hamming_sort(blocks):
    """Reorder signals using heuristics to try to place signals that are read
    by the same operators into adjacent positions (giving priority to larger
    blocks).

    Parameters
    ----------
    blocks : dict of {:class:`~nengo:nengo.builder.Signal`: frozenset of int}
        dictionary indicating which read blocks each signal is a part of

    Returns
    -------
    dict of {:class:`~nengo:nengo.builder.Signal`: int}
        indices indicating where each signal should be in the sorted list
    """

    sorted_blocks = []
    curr_blocks = None
    active_block = None

    unique_blocks = set(blocks.values())

    n_unique = len(unique_blocks)

    logger.debug("hamming sort:")
    logger.debug("unique blocks")
    logger.debug(unique_blocks)

    while True:
        logger.debug("curr_blocks %s", curr_blocks)

        if curr_blocks is None:
            # first pass through loop, initialize with default first block
            # (the rest of the loop will figure out what the actual first
            # block will be)
            curr_blocks = frozenset([0])
        else:
            # add the selected block to the sorted list
            sorted_blocks.append(curr_blocks)
            unique_blocks.remove(curr_blocks)

        if len(sorted_blocks) == n_unique:
            break

        # pick which block to go to next

        # start by picking all the blocks that are a continuation of the
        # active block (this is to give us some persistence, so it doesn't
        # jump around too much)
        if active_block is None:
            # pick the largest block in the current block to become the new
            # active block (note: the blocks are sorted from largest to
            # smallest, so the smallest value in curr_blocks is the largest
            # block. this ordering is used in several places)
            active_block = min(curr_blocks)

        next_blocks = [b for b in unique_blocks if active_block in b]
        if len(next_blocks) == 0:
            # there are no remaining blocks that are a continuation of the
            # current block, so they're all up for grabs
            next_blocks = unique_blocks
            active_block = None

        # find all the matching blocks (blocks which contain all the same
        # elements as curr_blocks, plus something extra)
        matching = [b for b in next_blocks if len(b | curr_blocks) == len(b)]
        if len(matching) > 0:
            next_blocks = matching

        # then within all the matching blocks, pick the ones with the smallest
        # hamming distance
        next_dists = [len(curr_blocks ^ b) for b in next_blocks]
        min_dist = min(next_dists)
        next_blocks = [b for i, b in enumerate(next_blocks)
                       if next_dists[i] == min_dist]

        # within all the blocks that have the same hamming distance, pick the
        # next block that matches along the largest blocks
        for i in sorted(curr_blocks):
            if len(next_blocks) == 1:
                break

            if any(i in b for b in next_blocks):
                next_blocks = [b for b in next_blocks if i in b]

        # within the blocks that match curr_block equally, pick the next block
        # containing the largest read blocks
        if len(next_blocks) > 1:
            next_blocks = [frozenset(min(sorted(b) for b in next_blocks))]

        curr_blocks = next_blocks[0]

    # the sort index for each signal is just the position of its block in
    # the sorted block list (since we don't care about the order of
    # signals within each block). signals that aren't part of any read block
    # get a default value of -1.
    block_idxs = {b: i for i, b in enumerate(sorted_blocks)}
    sort_idxs = defaultdict(
        lambda: -1, [(s, block_idxs[b]) for s, b in blocks.items()])

    return sort_idxs


def sort_ops_by_signals(sorted_reads, sigs, sig_idxs, new_plan, blocks, reads):
    """Rearrange operators to match the order of signals.

    Note: the same operators can be associated with multiple read blocks if
    they have multiple inputs, so rearranging the operators according to one
    of those blocks could mess up the order with respect to the other read
    block.  We iterate through the read blocks in increasing size so
    that the largest blocks win out.

    Parameters
    ----------
    sorted_reads : list of tuple of (:class:`~nengo:nengo.builder.Operator`, \
                                     int)
        the operators that form each read block, sorted by increasing size of
        the read block. in the case that a group of operators participate in
        multiple read blocks, the integer distinguishes which one of those
        inputs this block is associated with.
    sigs : list of :class:`~nengo:nengo.builder.Signal`
        signals that have been arranged into a given order by other parts
        of the algorithm
    sig_idxs : dict of {:class:`~nengo:nengo.builder.Signal`: int}
        sorted indices of signals
    new_plan : dict of {tuple of :class:`~nengo:nengo.builder.Operator`: \
                        tuple of :class:`~nengo:nengo.builder.Operator`}
        mapping from original operator group to the sorted operators
    blocks : dict of {:class:`~nengo:nengo.builder.Signal`: frozenset of int}
        indicates which read blocks each signal participates in
    reads : dict of {:class:`~nengo:nengo.builder.Operator`: \
                     list of :class:`~nengo:nengo.builder.Signal`}
        the signals read by each operator

    Returns
    -------
    new_plan : dict of {tuple of :class:`~nengo:nengo.builder.Operator`: \
                        tuple of :class:`~nengo:nengo.builder.Operator`}
        mapping from original operator group to the sorted operators
    sig_idxs : dict of {:class:`~nengo:nengo.builder.Signal`: int}
        signal indices, possibly updated to match new op order
    """

    logger.log(logging.DEBUG - 1, "sort ops by signals")

    for old_ops, read_block in sorted_reads:
        logger.log(logging.DEBUG - 1, "-" * 30)
        logger.log(logging.DEBUG - 1, "sorting ops %s", new_plan[old_ops])
        logger.log(logging.DEBUG - 1, "by %s",
                   [reads[op][read_block] for op in new_plan[old_ops]])

        if len(old_ops) == 1:
            # then we have nothing to sort
            continue

        ops = new_plan[old_ops]

        # note: the key is (signal index, view offset), so ops will be
        # sorted first by the order of the signals in the list, then by
        # the order of the views within each signal
        sorted_ops = sorted(
            ops, key=lambda op: (sig_idxs[reads[op][read_block].base],
                                 reads[op][read_block].elemoffset))

        new_plan[old_ops] = tuple(sorted_ops)

        logger.log(logging.DEBUG - 1, "sorted ops")
        logger.log(logging.DEBUG - 1, new_plan[old_ops])

        # after sorting the operators, we then rearrange all the read
        # blocks associated with this group of operators to match the new
        # order. note that this could make smaller (earlier) blocks out
        # of order, which will hopefully be fixed on future passes. however,
        # it means that larger (later) blocks will align themselves to this
        # order if possible
        # note2: we include the current read block in the groups to be sorted,
        # because while we know that these ops are in the same relative order
        # as the signals, the signals may not be adjacent (sorting will try
        # to make them adjacent)
        sig_idxs = sort_signals_by_ops(
            [x for x in sorted_reads if x[0] == old_ops],
            sigs, sig_idxs, new_plan, blocks, reads)

    return new_plan, sig_idxs


def sort_signals_by_ops(sorted_reads, sigs, sig_idxs, new_plan, blocks, reads):
    """Attempts to rearrange ``sigs`` so that it is in the same order as
    operator reads, without changing the overall block order.

    Parameters
    ----------
    sorted_reads : list of tuple of (:class:`~nengo:nengo.builder.Operator`, \
                                     int)
        the operators that form each read block, sorted by increasing size of
        the read block. in the case that a group of operators participate in
        multiple read blocks, the integer distinguishes which one of those
        inputs this block is associated with.
    sigs : list of :class:`~nengo:nengo.builder.Signal`
        signals to be sorted
    sig_idxs : dict of {:class:`~nengo:nengo.builder.Signal`: int}
        sorted indices of signals
    new_plan : dict of {tuple of :class:`~nengo:nengo.builder.Operator`: \
                        tuple of :class:`~nengo:nengo.builder.Operator`}
        mapping from original operator group to the sorted operators
    blocks : dict of {:class:`~nengo:nengo.builder.Signal`: frozenset of int}
        indicates which read blocks each signal participates in
    reads : dict of {:class:`~nengo:nengo.builder.Operator`: \
                     list of :class:`~nengo:nengo.builder.Signal`}
        the signals read by each operator

    Returns
    -------
    sig_idxs : dict of {:class:`~nengo:nengo.builder.Signal`: int}
        sorted indices of signals
    """

    logger.log(logging.DEBUG - 1, "-" * 10)
    logger.log(logging.DEBUG - 1, "sort signals by ops")

    for old_ops, read_block in sorted_reads:
        logger.log(logging.DEBUG - 1, "sorting signals %s",
                   [reads[op][read_block] for op in new_plan[old_ops]])
        logger.log(logging.DEBUG - 1, "%d %s", read_block, new_plan[old_ops])

        ops = new_plan[old_ops]

        sort_vals = {s: i for i, s in
                     enumerate(reads[op][read_block].base for op in ops)}

        if len(sort_vals) == 1:
            # only one read signal, so nothing to sort
            continue

        sort_idxs = [sig_idxs[s] for s in sort_vals]
        min_index = min(sort_idxs)
        max_index = max(sort_idxs)

        if max_index - min_index != len(sort_idxs) - 1:
            # this block isn't contiguous, so it isn't sortable
            continue

        # we try to sort things into everything <= the first read block
        # in op_reads and everything after, with the op_reads signals in
        # the middle (ordered to match op_reads)
        curr_block = None
        curr_max = -1
        for i, s in enumerate(sigs[min_index:max_index + 1]):
            if blocks[s] != curr_block:
                prev_max = curr_max
                curr_max = -1
                curr_block = blocks[s]

            idx = sort_vals[s]
            if idx < prev_max:
                # if the sort position for this signal is less than the
                # end of the previous sorted block, then the list is not
                # sortable
                break

            # update the max sort index in this block
            curr_max = max(curr_max, idx)
        else:
            for i, s in enumerate(
                    sorted(sort_vals, key=lambda s: sort_vals[s])):
                sig_idxs[s] = min_index + i

            logger.log(logging.DEBUG - 1, "sorted indices %s", sig_idxs)

    return sig_idxs


def noop_order_signals(plan, **kwargs):
    """A version of :func:`.graph_optimizer.order_signals` that doesn't do any
    reordering, for debugging."""

    all_signals = list(set([s.base for ops in plan for op in ops
                            for s in op.all_signals]))
    return all_signals, plan


def create_signals(sigs, plan, float_type, minibatch_size):
    """Groups signal data together into larger arrays, and represent each
    individual signal as a slice into that array.

    Parameters
    ----------
    sigs : list of :class:`~nengo:nengo.builder.Signal`
        base signals arranged into the order in which they should reside in
        memory (e.g., output from ``order_signals``)
    plan : list of tuple of :class:`~nengo:nengo.builder.Operator`
        operator execution plan (only used to get a list of all the operators)
    float_type : ``np.float32`` or ``np.float64``
        floating point precision to use for signals
    minibatch_size : int
        number of items in each minibatch

    Returns
    -------
    base_arrays : dict of {object : :class:`~numpy:numpy.ndarray`}
        combined arrays, containing the initial values for all signals
    sig_map : dict of {:class:`~nengo:nengo.builder.Signal`: \
                       :class:`.signals.TensorSignal`}
        mapping from ``nengo`` Signals to ``nengo_dl`` TensorSignals (views
        into the base arrays)
    """

    base_arrays = OrderedDict()
    curr_keys = {}
    sig_map = {}
    sig_idxs = {s: i for i, s in enumerate(sigs)}

    # find the non-overlapping partitions of the signals
    breaks = []
    diff = defaultdict(int)
    for ops in plan:
        # note: we don't include Resets, otherwise the big reset block
        # overrides most of the partitioning
        if not isinstance(ops[0], Reset):
            for i in range(len(ops[0].all_signals)):
                op_sigs = [op.all_signals[i].base for op in ops]
                idxs = [sig_idxs[s] for s in op_sigs]
                diff[op_sigs[np.argmin(idxs)]] += 1
                diff[op_sigs[np.argmax(idxs)]] -= 1

    # find the partition points in signal list
    open = 0
    for i, s in enumerate(sigs):
        if s in diff:
            open += diff[s]

        if open == 0:
            breaks += [i + 1]

    # create all the base signals
    for i, sig in enumerate(sigs):
        assert sig not in sig_map
        assert not sig.is_view

        if i in breaks:
            # start a new array for all current bases
            for k in curr_keys:
                curr_keys[k] = object()

        # convert to appropriate dtype
        if np.issubdtype(sig.dtype, np.float):
            dtype = float_type
        elif np.issubdtype(sig.dtype, np.integer):
            dtype = np.int32
        else:
            raise NotImplementedError

        # resize scalars to length 1 vectors
        shape = sig.shape if sig.shape != () else (1,)

        # parameters of signal that affect the base array
        array_params = (dtype, shape[1:], sig.trainable, sig.minibatched)

        # key used to map signals to base arrays
        if array_params not in curr_keys:
            curr_keys[array_params] = object()
        key = curr_keys[array_params]

        initial_value = sig.initial_value.astype(dtype, copy=False)

        # broadcast scalars up to full size
        if initial_value.shape != shape:
            initial_value = np.resize(initial_value, shape)

        if sig.minibatched:
            # duplicate along minibatch dimension
            initial_value = np.tile(
                initial_value[..., None],
                tuple(1 for _ in shape) + (minibatch_size,))

        if key in base_arrays:
            base_arrays[key][0].append(initial_value)
            base_arrays[key][2] += shape[0]
        else:
            base_arrays[key] = [[initial_value], sig.trainable, shape[0]]

        n = base_arrays[key][-1]
        indices = np.arange(n - shape[0], n)

        sig_map[sig] = signals.TensorSignal(
            indices, key, dtype, shape, sig.minibatched, label=sig.name)

        logger.debug("created base signal")
        logger.debug(sig)
        logger.debug(sig_map[sig])

    for key in base_arrays:
        arrs, t, _ = base_arrays[key]
        base_arrays[key] = (np.concatenate(arrs, axis=0), t)

    # add any signal views to the sig_map
    all_views = [sig for ops in plan for op in ops for sig in op.all_signals
                 if sig.is_view]
    for sig in all_views:
        if sig.size == sig.base.size:
            # reshape view
            sig_map[sig] = sig_map[sig.base].reshape(sig.shape)
        else:
            if sig.shape[1:] != sig.base.shape[1:]:
                raise NotImplementedError(
                    "Slicing and reshaping the same signal is not "
                    "supported")

            # slice view
            assert np.all([x == 1 for x in sig.elemstrides[1:]])

            start = sig.elemoffset
            stride = sig.elemstrides[0]
            stop = start + sig.size * stride
            if stop < 0:
                stop = None

            sig_map[sig] = sig_map[sig.base][slice(start, stop, stride)]

    # error checking
    for sig, tensor_sig in sig_map.items():
        # tensorsignal shapes should match signal shapes
        assert tensor_sig.shape == (sig.shape if sig.shape != () else (1,))

        # tensorsignal values should match signal values
        initial_value = sig.initial_value
        if sig.minibatched:
            initial_value = initial_value[..., None]

        assert np.allclose(base_arrays[tensor_sig.key][0][tensor_sig.indices],
                           initial_value.astype(dtype))

    logger.debug("base arrays")
    logger.debug("\n".join([str((k, v[0].dtype, v[0].shape, v[1]))
                            for k, v in base_arrays.items()]))

    return base_arrays, sig_map
