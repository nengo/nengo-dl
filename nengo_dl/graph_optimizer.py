"""
These functions are used to restructure the Nengo operator graph so that it
can be simulated more efficiently when converted into a TensorFlow graph.
"""

import logging
import warnings
from collections import defaultdict

import numpy as np
from nengo.builder.operator import Copy, DotInc, ElementwiseInc, Reset, SparseDotInc
from nengo.builder.processes import SimProcess
from nengo.builder.transforms import ConvInc
from nengo.exceptions import BuildError, SignalError
from nengo.transforms import SparseMatrix
from nengo.utils.graphs import BidirectionalDAG, toposort
from nengo.utils.simulator import operator_dependency_graph

from nengo_dl import (
    builder,
    compat,
    learning_rule_builders,
    neuron_builders,
    op_builders,
    process_builders,
    tensor_node,
    transform_builders,
)

logger = logging.getLogger(__name__)


def mergeable(op, chosen_ops):
    """
    Check if the given op can be merged with the candidate group.

    Parameters
    ----------
    op : `~nengo.builder.Operator`
        The operator to be merged
    chosen_ops : list of `~nengo.builder.Operator`
        The operator group to be merged in to

    Returns
    -------
    mergeable : bool
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
    if (
        len(op.sets) != len(c.sets)
        or len(op.incs) != len(c.incs)
        or len(op.reads) != len(c.reads)
        or len(op.updates) != len(c.updates)
    ):
        return False

    # signals must be mergeable into the same base array
    for s0, s1 in zip(op.all_signals, c.all_signals):
        # dtype of signals must match
        if s0.dtype != s1.dtype:
            return False

        # sparse status of signals must match
        if s0.sparse != s1.sparse:
            return False

        # trainable/minibatched must match
        if s0.trainable != s1.trainable or s0.minibatched != s1.minibatched:
            return False

        if not s0.sparse:
            # note: for sparse signals we don't need to worry about the shape
            # of the signal, since what we'll be combining is the underlying
            # data (which is always a vector)

            # shape of signal base must match on all axes > 0
            if s0.base.shape[1:] != s1.base.shape[1:]:
                return False

            # display shape must also match (since we need the shape to be well
            # defined when we combine the signals)
            if s0.shape[1:] != s1.shape[1:]:
                return False

    # operator-specific checks
    return builder.Builder.builders[type(op)].mergeable(op, c)


def greedy_planner(operators):
    """
    Create merged execution plan through greedy selection.

    Always selects the largest available group of operators as the next to be
    executed.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        All the ``nengo`` operators in a model (unordered)

    Returns
    -------
    plan : list of tuple of `~nengo.builder.Operator`
        Operators combined into mergeable groups and in execution order

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
    for op, dests in dependency_graph.items():
        for op2 in dests:
            predecessors_of[op2].add(op)
        successors_of[op].update(dests)

    # the ops in `available` are ready to be scheduled (all predecessors
    # have been scheduled).
    # initialize it with the ops that have no predecessors
    available = [op for op, dep in predecessors_of.items() if len(dep) == 0]

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
        groups = sorted(groups, key=len)
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
    logger.debug("\n%s" * len(plan), *plan)

    assert len(operators) == sum(len(ops) for ops in plan)

    return plan


def tree_planner(op_list, max_depth=3):
    """
    Create merged execution plan through exhaustive tree search.

    The ``max_depth`` parameter scales the planner between full tree search
    and greedy search.  ``max_depth==1`` is equivalent to
    `.greedy_planner`, and ``max_depth==len(op_list)`` is full tree
    search (guaranteed to find the optimal plan, but likely very slow).

    Parameters
    ----------
    op_list : list of `~nengo.builder.Operator`
        All the ``nengo`` operators in a model (unordered)
    max_depth : int
        The planner will search this many steps ahead before selecting which
        group to schedule next

    Returns
    -------
    plan : list of tuple of `~nengo.builder.Operator`
        Operators combined into mergeable groups and in execution order
    """

    def shortest_plan(
        selected, successors_of, predecessors_of, cache, max_depth, available
    ):
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
                remaining_length = new_len
            else:
                # update the selected ops after adding group
                new_selected = selected | group

                try:
                    # check if we've already computed the shortest path
                    # for the selected ops and depth
                    remaining_length = cache[max_depth - 1][new_selected]

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
                    _, remaining_length = shortest_plan(
                        new_selected,
                        successors_of,
                        predecessors_of,
                        cache,
                        max_depth - 1,
                        available,
                    )

                    # return the available list to its original state for
                    # the next group (note: this is faster than copying
                    # the list)
                    for op in successors:
                        predecessors_of[op] += 1

                        if predecessors_of[op] == 1:
                            available[mergeable_cache[op]].remove(op)
                    available[i] = group

            if remaining_length + 1 < shortest[1]:
                # new shortest path found
                shortest = (tuple(group), remaining_length + 1)

        if shortest[0] is None:
            raise BuildError("Cycle detected during graph optimization")

        cache[max_depth][selected] = shortest[1]

        return shortest

    # compute operator dependency graph
    successors_of = operator_dependency_graph(op_list)

    # convert operators to integer indices (to save memory and make
    # lookup faster)
    op_codes = {op: np.uint32(i) for i, op in enumerate(op_list)}
    successors_of = {
        op_codes[k]: set(op_codes[x] for x in v) for k, v in successors_of.items()
    }

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
    available = [set(op for op in g if predecessors_of[op] == 0) for g in groups]

    plan = []
    while len(successors_of) > 0:
        # find the best plan of the given depth
        selected, _ = shortest_plan(
            frozenset(),
            successors_of,
            predecessors_of,
            [{} for _ in range(max_depth + 1)],
            max_depth,
            available,
        )

        # select the first item in that plan (i.e., the best group to select
        # after looking ahead for max_depth steps)
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
    # note: we sort g so that group order is deterministic
    plan = [tuple(op_list[x] for x in sorted(g)) for g in plan]

    logger.debug("TREE PLAN")
    logger.debug("\n%s" * len(plan), *plan)

    return plan


def transitive_planner(op_list):
    """
    Create merged execution plan through transitive closure construction.

    This is something like a middle ground between `.greedy_planner` and
    `.tree_planner`; it can improve simulation time over the greedy
    planner, but comes with potentially significant build time increases.

    Parameters
    ----------
    op_list : list of `~nengo.builder.Operator`
        All the ``nengo`` operators in a model (unordered)

    Returns
    -------
    plan : list of tuple of `~nengo.builder.Operator`
        Operators combined into mergeable groups and in execution order
    """

    n_ele = len(op_list)
    merge_groups = {}
    dg = operator_dependency_graph(op_list)
    op_codes = {op: np.uint32(i) for i, op in enumerate(op_list)}
    dg = {op_codes[k]: set(op_codes[x] for x in v) for k, v in dg.items()}
    op_codes = {}  # so it will get garbage collected
    dg = BidirectionalDAG(dg)

    # fail fast here if the op graph has cycles
    toposort(dg.forward)

    builder_types = [builder.Builder.builders[type(op)] for op in op_list]

    # sort operators by builder (we'll only be interested in one builder type
    # at a time, because we can't merge operators between builder types anyway)
    ops_by_type = defaultdict(set)
    for i, op in enumerate(op_list):
        ops_by_type[builder_types[i]].add(np.uint32(i))

    # heuristic ordering for builder types (earlier items in the list will
    # have higher priority, meaning that we will choose to merge those ops
    # and potentially break lower-priority groups)
    order = [
        op_builders.DotIncBuilder,
        op_builders.ElementwiseIncBuilder,
        neuron_builders.SimNeuronsBuilder,
        process_builders.SimProcessBuilder,
        op_builders.SimPyFuncBuilder,
        learning_rule_builders.SimOjaBuilder,
        learning_rule_builders.SimVojaBuilder,
        learning_rule_builders.SimBCMBuilder,
        op_builders.CopyBuilder,
        op_builders.ResetBuilder,
        tensor_node.SimTensorNodeBuilder,
        op_builders.SimProbeBuilder,
        op_builders.TimeUpdateBuilder,
    ]

    for builder_type in order:
        if builder_type not in ops_by_type:
            # no ops of this type in the model
            continue

        ops = ops_by_type[builder_type]

        # compute transitive closure
        trans = [None for _ in range(n_ele)]
        transitive_closure_recurse(
            dg.forward, ops, trans, builder_type, builder_types, {}
        )

        # reduce it to the elements we care about (ops of the current
        # builder type)
        trans = {i: v for i, v in enumerate(trans[: len(op_list)]) if i in ops}

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

        del ops_by_type[builder_type]

    assert len(ops_by_type) == 0, "Builder type missing from `order`"

    # toposort the merged graph to come up with execution plan
    plan = toposort(dg.forward)
    plan = [tuple(op_list[x] for x in merge_groups[group]) for group in plan]

    logger.debug("TRANSITIVE PLAN")
    logger.debug("\n%s" * len(plan), *plan)

    return plan


def transitive_closure_recurse(dg, ops, trans, builder_type, builder_types, cache):
    """
    Computes the transitive closure for the given graph, restricted to the
    operators with the given builder type.

    Parameters
    ----------
    dg : dict of {int: set of int}
        Dependency graph where ``dg[a] = {b, c}`` indicates that operators
        ``b`` and ``c`` are dependent on ``a``
    ops : list of int
        The operators for which we want to compute the transitive closure
    trans : dict of {int: set of int}
        The transitive closure for the graph (will be filled in-place)
    builder_type : type
        One of the ``nengo_dl`` build classes (e.g.,
        `~.op_builders.CopyBuilder`), specifying the type of operators
        to include in the transitive closure
    builder_types : list of type
        The build class for each operator
    cache : dict of {frozenset of int: set of int}
        Stores base sets which ``trans`` will reference (to reduce memory
        usage, since many elements in ``trans`` will have the same value)

    Notes
    -----
    This function uses ints to refer to operators, where the int indicates
    the index of the operator in the overall op list (this is done to save
    memory).  See `.transitive_planner`.
    """

    for op in ops:
        if trans[op] is not None:
            # this can occur if the downstream calculations of an earlier
            # op filled in the value for this op
            continue

        todo = [x for x in dg[op] if trans[x] is None]
        transitive_closure_recurse(dg, todo, trans, builder_type, builder_types, cache)

        merged = set(
            x
            for x in dg[op]
            if x < len(builder_types) and builder_types[x] == builder_type
        )

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


def noop_planner(operators):
    """
    Orders operators into a valid execution order, but does not perform
    any merging.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        All the ``nengo`` operators in a model (unordered)

    Returns
    -------
    plan : list of tuple of `~nengo.builder.Operator`
        Operators in execution order
    """

    dependency_graph = operator_dependency_graph(operators)
    plan = [(op,) for op in toposort(dependency_graph)]

    logger.debug("NOOP PLAN")
    logger.debug("\n%s" * len(plan), *plan)

    return plan


def order_signals(plan, n_passes=10):
    """
    Orders signals and operators to try to structure reads/writes in contiguous
    blocks.

    Parameters
    ----------
    plan : list of tuple of `~nengo.builder.Operator`
        Operator execution plan (e.g., output from ``greedy_planner``)
    n_passes : int
        Number of repeated passes through the operator reordering stage

    Returns
    -------
    signals : list of `~nengo.builder.Signal`
        Signals organized into the order in which we want them arranged in
        memory
    plan : list of tuple of `~nengo.builder.Operator`
        Input plan with operators reordered within groups to align with order
        of signals
    """

    # get all the unique base signals (we use a dict to drop the duplicate
    # bases without changing their order, so that signal order will be
    # deterministic for a given model)
    all_signals = list(
        {s.base: None for ops in plan for op in ops for s in op.all_signals}.keys()
    )

    # figure out all the read/write blocks in the plan (in theory we would like
    # each block to become a contiguous chunk in the base array)
    io_blocks = {}

    op_sigs = {}
    for ops in plan:
        # op_sigs stores the signals we care about during the sort process for
        # each op. at the moment this is equivalent to op.all_signals, but
        # keeping it as it provides a layer of indirection that has proven
        # useful in the past and may be in the future.
        for op in ops:
            op_sigs[op] = op.all_signals.copy()

        # the i'th signal for each op in the op group is one io group
        # (note that we only care about bases, since those are the things we
        # are trying to order)
        for i in range(len(op_sigs[ops[0]])):
            io_blocks[(ops, i)] = frozenset(op_sigs[op][i].base for op in ops)

    # get rid of duplicate io blocks
    duplicates = defaultdict(int)
    for x in io_blocks.values():
        duplicates[x] += 1

    # sort by the size of the block (descending order)
    # note: we multiply by the number of duplicates, since blocks that
    # are accessed by multiple op groups will have a proportionally larger
    # impact on performance
    sorted_blocks = [
        block
        for block, _ in sorted(
            duplicates.items(),
            key=lambda b: sum(s.size for s in b[0]) * b[1],
            reverse=True,
        )
    ]

    # figure out which io blocks each signal participates in
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

    # list of the ops in each io block, sorted by the size of that io block
    sorted_io = sorted(
        io_blocks.keys(), key=lambda p: -sorted_blocks.index(io_blocks[p])
    )

    logger.debug("sorted io")
    logger.debug("\n".join(str(x) for x in sorted_io))

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
    #    io signals
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
    logger.debug("\n%s" * len(new_plan), *new_plan.values())
    logger.debug("signal indices")
    logger.debug(sig_idxs)

    for n in range(n_passes):
        # TODO: every few iterations, eliminate the smallest unsatisfied block?
        logger.debug("======== pass %d ========", n)

        # save previous plan/idxs, so we can check if they change for
        # early termination
        prev_plan = new_plan.copy()
        prev_sig_idxs = sig_idxs  # note: no copy necessary

        # reorder ops by signal order. this leaves the overall
        # hamming sort block order unchanged.
        new_plan, sig_idxs = sort_ops_by_signals(
            sorted_io, all_signals, sig_idxs, new_plan, signal_blocks, op_sigs
        )

        logger.debug("resorted ops")
        logger.debug("\n%s" * len(new_plan), *new_plan.values())

        logger.debug("reordered signal indices")
        logger.debug(sig_idxs)

        if all(
            x == y for ops in plan for x, y in zip(new_plan[ops], prev_plan[ops])
        ) and all(sig_idxs[s] == prev_sig_idxs[s] for s in all_signals):
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
        assert set(ops) == set(new_ops)

    logger.debug("final sorted signals")
    logger.debug(sorted_signals)
    logger.debug("new plan")
    logger.debug("\n%s" * len(new_plan), *new_plan.values())
    logger.debug("blocks")
    logger.debug("\n%s", display_signal_blocks(new_plan, sorted_signals))

    return sorted_signals, [new_plan[ops] for ops in plan]


def hamming_sort(blocks):
    """
    Reorder signals using heuristics to try to place signals that are accessed
    by the same operators into adjacent positions (giving priority to larger
    blocks).

    Parameters
    ----------
    blocks : dict of {`~nengo.builder.Signal`: frozenset of int}
        Dictionary indicating which io blocks each signal is a part of

    Returns
    -------
    sort_idxs : dict of {`~nengo.builder.Signal`: int}
        Indices indicating where each signal should be in the sorted list
    """

    sorted_blocks = []
    curr_blocks = None
    active_block = None

    unique_blocks = set(blocks.values())

    n_unique = len(unique_blocks)

    logger.log(logging.DEBUG - 1, "hamming sort:")
    logger.log(logging.DEBUG - 1, "unique blocks")
    logger.log(logging.DEBUG - 1, unique_blocks)

    while True:
        logger.log(logging.DEBUG - 1, "curr_blocks %s", curr_blocks)

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
        next_blocks = [
            b for i, b in enumerate(next_blocks) if next_dists[i] == min_dist
        ]

        # within all the blocks that have the same hamming distance, pick the
        # next block that matches along the largest blocks
        for i in sorted(curr_blocks):
            if len(next_blocks) == 1:
                break

            if any(i in b for b in next_blocks):
                next_blocks = [b for b in next_blocks if i in b]

        # within the blocks that match curr_block equally, pick the next block
        # containing the largest io blocks
        if len(next_blocks) > 1:
            next_blocks = [frozenset(min(sorted(b) for b in next_blocks))]

        curr_blocks = next_blocks[0]

    # the sort index for each signal is just the position of its block in
    # the sorted block list (since we don't care about the order of
    # signals within each block). signals that aren't part of any io block
    # get a default value of -1.
    block_idxs = {b: i for i, b in enumerate(sorted_blocks)}
    sort_idxs = defaultdict(lambda: -1, [(s, block_idxs[b]) for s, b in blocks.items()])

    return sort_idxs


def sort_ops_by_signals(sorted_io, sigs, sig_idxs, new_plan, blocks, op_sigs):
    """
    Rearrange operators to match the order of signals.

    Note: the same operators can be associated with multiple IO blocks if
    they have multiple inputs, so rearranging the operators according to one
    of those blocks could mess up the order with respect to the other IO
    block.  We iterate through the IO blocks in increasing size so
    that the largest blocks win out.

    Parameters
    ----------
    sorted_io : list of tuple of (`~nengo.builder.Operator`, int)
        The operators that form each IO block, sorted by increasing size of
        the block. In the case that a group of operators participate in
        multiple IO blocks, the integer distinguishes which one of those
        blocks this block is associated with.
    sigs : list of `~nengo.builder.Signal`
        Signals that have been arranged into a given order by other parts
        of the algorithm
    sig_idxs : dict of {`~nengo.builder.Signal`: int}
        Sorted indices of signals
    new_plan : dict of {tuple of `~nengo.builder.Operator`: \
                        tuple of `~nengo.builder.Operator`}
        Mapping from original operator group to the sorted operators
    blocks : dict of {`~nengo.builder.Signal`: frozenset of int}
        Indicates which IO blocks each signal participates in
    op_sigs : dict of {`~nengo.builder.Operator`: \
                       list of `~nengo.builder.Signal`}
        The signals accessed by each operator

    Returns
    -------
    new_plan : dict of {tuple of `~nengo.builder.Operator`: \
                        tuple of `~nengo.builder.Operator`}
        Mapping from original operator group to the sorted operators
    sig_idxs : dict of {`~nengo.builder.Signal`: int}
        Signal indices, possibly updated to match new op order
    """

    logger.log(logging.DEBUG - 1, "sort ops by signals")

    for old_ops, io_block in sorted_io:
        logger.log(logging.DEBUG - 1, "-" * 30)
        logger.log(logging.DEBUG - 1, "sorting ops %s", new_plan[old_ops])
        logger.log(
            logging.DEBUG - 1,
            "by %s",
            [op_sigs[op][io_block] for op in new_plan[old_ops]],
        )

        if len(old_ops) == 1:
            # then we have nothing to sort
            continue

        ops = new_plan[old_ops]

        # note: the key is (signal index, view offset), so ops will be
        # sorted first by the order of the signals in the list, then by
        # the order of the views within each signal
        sorted_ops = sorted(
            ops,
            key=lambda op, io_block=io_block: (
                sig_idxs[op_sigs[op][io_block].base],
                op_sigs[op][io_block].elemoffset,
            ),
        )

        new_plan[old_ops] = tuple(sorted_ops)

        logger.log(logging.DEBUG - 1, "sorted ops")
        logger.log(logging.DEBUG - 1, new_plan[old_ops])

        # after sorting the operators, we then rearrange all the io
        # blocks associated with this group of operators to match the new
        # order. note that this could make smaller (earlier) blocks out
        # of order, which will hopefully be fixed on future passes. however,
        # it means that larger (later) blocks will align themselves to this
        # order if possible
        # note2: we include the current io block in the groups to be sorted,
        # because while we know that these ops are in the same relative order
        # as the signals, the signals may not be adjacent (sorting will try
        # to make them adjacent)
        sig_idxs = sort_signals_by_ops(
            [x for x in sorted_io if x[0] == old_ops],
            sigs,
            sig_idxs,
            new_plan,
            blocks,
            op_sigs,
        )

    return new_plan, sig_idxs


def sort_signals_by_ops(sorted_io, sigs, sig_idxs, new_plan, blocks, op_sigs):
    """
    Attempts to rearrange ``sigs`` so that it is in the same order as
    operator signals, without changing the overall block order.

    Parameters
    ----------
    sorted_io : list of tuple of (`~nengo.builder.Operator`, \
                                     int)
        The operators that form each io block, sorted by increasing size of
        the io block. In the case that a group of operators participate in
        multiple io blocks, the integer distinguishes which one of those
        blocks this block is associated with.
    sigs : list of `~nengo.builder.Signal`
        Signals to be sorted
    sig_idxs : dict of {`~nengo.builder.Signal`: int}
        Sorted indices of signals
    new_plan : dict of {tuple of `~nengo.builder.Operator`: \
                        tuple of `~nengo.builder.Operator`}
        Mapping from original operator group to the sorted operators
    blocks : dict of {`~nengo.builder.Signal`: frozenset of int}
        Indicates which io blocks each signal participates in
    op_sigs : dict of {`~nengo.builder.Operator`: \
                       list of `~nengo.builder.Signal`}
        The signals accessed by each operator

    Returns
    -------
    sig_idxs : dict of {`~nengo.builder.Signal`: int}
        Sorted indices of signals
    """

    logger.log(logging.DEBUG - 1, "-" * 10)
    logger.log(logging.DEBUG - 1, "sort signals by ops")

    for old_ops, io_block in sorted_io:
        logger.log(
            logging.DEBUG - 1,
            "sorting signals %s",
            [op_sigs[op][io_block] for op in new_plan[old_ops]],
        )
        logger.log(logging.DEBUG - 1, "%d %s", io_block, new_plan[old_ops])

        ops = new_plan[old_ops]

        sort_vals = {
            s: i for i, s in enumerate(op_sigs[op][io_block].base for op in ops)
        }

        if len(sort_vals) == 1:
            # only one accessed signal, so nothing to sort
            continue

        sort_idxs = [sig_idxs[s] for s in sort_vals]
        min_index = min(sort_idxs)
        max_index = max(sort_idxs)

        if max_index - min_index != len(sort_idxs) - 1:
            # this block isn't contiguous, so it isn't sortable
            continue

        # we try to sort things into everything <= the first io block
        # in op_sigs and everything after, with the op_sigs signals in
        # the middle (ordered to match op_sigs)
        curr_block = None
        curr_max = -1
        for i, s in enumerate(sigs[min_index : max_index + 1]):
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
                sorted(sort_vals, key=lambda x, sort_vals=sort_vals: sort_vals[x])
            ):
                sig_idxs[s] = min_index + i

            logger.log(logging.DEBUG - 1, "sorted indices %s", sig_idxs)

    return sig_idxs


def noop_order_signals(plan, **_):
    """A version of `.graph_optimizer.order_signals` that doesn't do any
    reordering, for debugging."""

    all_signals = list(
        compat.FrozenOrderedSet(
            s.base for ops in plan for op in ops for s in op.all_signals
        )
    )
    return all_signals, plan


def remove_unmodified_resets(operators):
    """
    Remove any Reset operators that are targeting a signal that is
    never modified.

    If a signal is reset, but never inced/updated after that, we can just set
    the default signal value to the reset value and remove the reset. Note:
    this wouldn't normally happen, but it can happen if we removed
    some of the incs (e.g. in `.remove_zero_incs`).

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators
    """

    _, incs, _, updates = signal_io_dicts(operators)

    new_operators = []
    for op in operators:
        if type(op) == Reset:
            target = op.dst
            if len(incs[target.base]) + len(updates[target.base]) == 0:
                target.initial_value.setflags(write=True)
                target.initial_value[...] = op.value
                target.initial_value.setflags(write=False)
            else:
                new_operators.append(op)
        else:
            new_operators.append(op)

    return new_operators


def remove_zero_incs(operators):
    """
    Remove any operators where we know the input (and therefore output) is
    zero.

    If the input to a DotInc/ElementwiseInc/Copy/ConvInc is zero then we know
    that the output of the op will be zero, so we can just get rid of it.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators
    """

    logger.debug("REMOVE_ZERO_INCS")
    logger.debug("input ops")
    logger.debug(operators)

    def all_zero(sig):
        data = sig.initial_value
        if sig.sparse:
            if not isinstance(data, SparseMatrix):
                data = data.tocoo()
            data = data.data

        return np.all(data == 0)

    sets, incs, _, updates = signal_io_dicts(operators)

    new_operators = []
    for op in operators:
        if isinstance(op, (DotInc, ElementwiseInc, Copy, ConvInc)):
            for src in op.reads:
                # check if the input is the output of a Node (in which case the
                # value might change, so we should never get rid of this op).
                # checking the name of the signal seems a bit fragile, but I
                # can't think of a better solution
                if src.name.startswith("<Node"):
                    continue

                # find any ops that modify src
                pred = sets[src.base] + incs[src.base]

                # the input (and therefore output) will be zero if the only
                # input is a Reset(0) op, or the only input is a constant
                # signal (not set/inc/updated) that is all zero
                zero_input = (
                    (
                        len(pred) == 1
                        and type(pred[0]) == Reset
                        and np.all(pred[0].value == 0)
                    )
                    or (
                        len(pred) == 0 and all_zero(src) and len(updates[src.base]) == 0
                    )
                    and not src.trainable
                )
                if zero_input:
                    if len(op.sets) > 0:
                        new_operators.append(Reset(op.sets[0]))
                    break
            else:
                new_operators.append(op)
        else:
            new_operators.append(op)

    logger.debug("new ops")
    logger.debug(new_operators)

    return new_operators


def remove_reset_incs(operators):
    """Replace ``y=Reset(0) + x`` with ``y=x``.

    If a signal is Reset and Inc'd, we can change that to a Set that combines
    the two ops (note: any other incs of that signal can proceed as normal)

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        modified list of operators

    Notes
    -----
    In practice, this modification can hurt more than it helps.  Inc
    operators are cheaper to compute the gradient for, and changing Incs to
    Incs and Sets splits up the Inc merge groups. It tends to provide the
    most value for models consisting of long linear chains of objects.
    """

    # note: not using signal_io_dict because we care about exact signal matches
    # in this case, not bases
    valid_inc_types = [
        ElementwiseInc,
        SparseDotInc,
        DotInc,
        Copy,
        SimProcess,
        ConvInc,
        op_builders.ResetInc,
    ]
    incs = defaultdict(list)
    for op in operators:
        for s in op.incs:
            if type(op) not in valid_inc_types:
                warnings.warn(f"Unknown incer type {type(op)} in remove_reset_incs")
            elif getattr(op, "dst_slice", None) is None:
                # don't include copy ops with dst_slice, as they aren't incrementing
                # the whole signal
                incs[s].append(op)

    new_operators = []
    ignore = []
    for op in operators:
        if op in ignore:
            # don't add this op to new_operators
            ignore.remove(op)
            continue

        if type(op) == Reset and np.all(op.value == 0) and len(incs[op.dst]) > 0:
            # pick the first op that increments dst, and change it to a set
            # (to take the place of the reset)
            incer = incs[op.dst][0]

            if isinstance(incer, ElementwiseInc):
                setter = op_builders.ElementwiseSet(
                    incer.A, incer.X, incer.Y, tag=incer.tag
                )
            elif isinstance(incer, SparseDotInc):
                # note: this needs to come before the DotInc condition, since
                # SparseDotInc is a subclass of DotInc
                setter = op_builders.SparseDotSet(
                    incer.A, incer.X, incer.Y, tag=incer.tag
                )
            elif isinstance(incer, DotInc):
                setter = op_builders.DotSet(incer.A, incer.X, incer.Y, tag=incer.tag)
            elif isinstance(incer, Copy):
                setter = Copy(
                    incer.src,
                    incer.dst,
                    src_slice=incer.src_slice,
                    dst_slice=incer.dst_slice,
                    inc=False,
                    tag=incer.tag,
                )
            elif isinstance(incer, SimProcess):
                setter = SimProcess(
                    incer.process,
                    incer.input,
                    incer.output,
                    incer.t,
                    mode="set",
                    state=incer.state,
                    tag=incer.tag,
                )
            elif isinstance(incer, ConvInc):
                setter = transform_builders.ConvSet(
                    incer.W, incer.X, incer.Y, incer.conv, tag=incer.tag
                )
            elif isinstance(incer, op_builders.ResetInc):
                setter = Reset(incer.dst, tag=incer.tag)
                # setting the value separately to bypass float casting in Reset init
                setter.value = incer.value

            # replace incer with setter
            try:
                new_operators.remove(incer)
            except ValueError:
                ignore.append(incer)
            new_operators.append(setter)
        else:
            new_operators.append(op)

    return new_operators


def remove_constant_copies(operators):
    """
    Change Copies with constant input to Resets.

    If a Copy has no dependencies, or just one Reset() dependency, then
    we can change it to an op that just directly sets the output signal to
    the Copy input value.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators
    """

    sets, incs, _, updates = signal_io_dicts(operators)

    new_operators = []
    ignore = []
    for op in operators:
        if op in ignore:
            # don't add this op to new_operators
            ignore.remove(op)
            continue

        if isinstance(op, Copy):
            src = op.src

            try:
                dst = op.dst if op.dst_slice is None else op.dst[op.dst_slice]
            except SignalError:
                # Copy is implementing advanced indexing, which cannot be applied
                # directly to a signal
                new_operators.append(op)
                continue

            # check if the input is the output of a Node (in which case the
            # value might change, so we should never get rid of this op).
            # checking the name of the signal seems a bit fragile, but I can't
            # think of a better solution
            if src.name.startswith("<Node"):
                new_operators.append(op)
                continue

            pred = sets[src.base] + incs[src.base]
            if len(pred) == 0 and not op.src.trainable and len(updates[src.base]) == 0:
                # no predecessors means that the src is constant. but we still
                # need to keep the signal if it is trainable (since
                # changing it to a reset op would make it not trainable).
                # we also need to check if anything is updating src (which
                # wouldn't be in the predecessors).
                val = op.src.initial_value[op.src_slice]
            elif len(pred) == 1 and type(pred[0]) == Reset:
                # if the only predecessor is a Reset, we can just use that
                # set value
                val = pred[0].value

                # remove the reset operator
                try:
                    new_operators.remove(pred[0])
                except ValueError:
                    ignore.append(pred[0])
            else:
                new_operators.append(op)
                continue

            new_op = op_builders.ResetInc(dst) if op.inc else Reset(dst)
            # note: we need to set the value separately to bypass the float()
            # casting in Reset
            new_op.value = val

            new_operators.append(new_op)
        else:
            new_operators.append(op)

    return new_operators


def remove_identity_muls(operators):
    """
    Change y=x*1 ops to y=x Copy ops.

    If one of the inputs to a DotInc/ElementwiseInc is 1 then we can skip
    the multiplication and change it to a Copy op.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators
    """

    sets, incs, _, updates = signal_io_dicts(operators)

    def is_identity(x, sig):
        if isinstance(x, float) or x.shape == ():
            return x == 1

        d = x.shape[0]
        if sig.ndim == 1:
            return x.shape == (d,) and np.allclose(x, 1)

        if isinstance(x, SparseMatrix):
            return (
                x.shape == (d, d)
                and np.allclose(x.data, 1)
                and np.allclose(x.indices, [[i, i] for i in range(d)])
            )

        return (
            x.shape == (d, d)
            and np.allclose(x.diagonal(), 1)
            and np.allclose(np.sum(np.abs(x)), d)  # all non-diagonal elements are 0
        )

    new_operators = []
    for op in operators:
        if isinstance(op, (DotInc, ElementwiseInc)):
            # we can check A or X for elementwise inc, since either being 1
            # means that this is a noop. but if X is 1 on a dotinc this is
            # still a useful op, and we shouldn't remove it
            srcs = op.reads if isinstance(op, ElementwiseInc) else [op.A]
            for src in srcs:
                # check if the input is the output of a Node (in which case the
                # value might change, so we should never get rid of this op).
                # checking the name of the signal seems a bit fragile, but I
                # can't think of a better solution
                if src.name.startswith("<Node"):
                    continue

                # find any ops that modify src
                pred = sets[src.base] + incs[src.base]

                # the input will be one if the only input is a Reset(1) op, or
                # the only input is a constant signal (not set/inc/updated)
                # that is an identity value
                identity_input = (
                    (
                        len(pred) == 1
                        and type(pred[0]) == Reset
                        and is_identity(pred[0].value, src)
                    )
                    or (
                        len(pred) == 0
                        and is_identity(src.initial_value, src)
                        and len(updates[src.base]) == 0
                    )
                    and not src.trainable
                )

                if identity_input:
                    other_src = [x for x in op.reads if x is not src][0]
                    new_operators.append(
                        Copy(
                            other_src,
                            op.Y,
                            inc=len(op.incs) > 0,
                            tag=f"{op.tag}.identity_mul",
                        )
                    )
                    break
            else:
                new_operators.append(op)
        else:
            new_operators.append(op)

    return new_operators


def signal_io_dicts(operators):
    """
    Organizes operators into dictionaries according to the signals they
    set/inc/read/update.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    sets : dict of {`~nengo.builder.Signal`: \
                    list of `~nengo.builder.Operator`}
        A dictionary indicating all the Operators that set each signal.
    incs : dict of {`~nengo.builder.Signal`: \
                    list of `~nengo.builder.Operator`}
        A dictionary indicating all the Operators that inc each signal.
    reads : dict of {`~nengo.builder.Signal`: \
                     list of `~nengo.builder.Operator`}
        A dictionary indicating all the Operators that read each signal.
    updates : dict of {`~nengo.builder.Signal`: \
                       list of `~nengo.builder.Operator`}
        A dictionary indicating all the Operators that update each signal.
    """

    # note: we manually initialize the arrays because we want there to be
    # an entry for all the signal bases, but get an error if we try to
    # access any non-base signals
    sets = {s.base: [] for op in operators for s in op.all_signals}
    incs = {s.base: [] for op in operators for s in op.all_signals}
    reads = {s.base: [] for op in operators for s in op.all_signals}
    updates = {s.base: [] for op in operators for s in op.all_signals}

    for op in operators:
        for s in op.sets:
            sets[s.base].append(op)
        for s in op.incs:
            incs[s.base].append(op)
        for s in op.reads:
            reads[s.base].append(op)
        for s in op.updates:
            updates[s.base].append(op)

    return sets, incs, reads, updates


def display_signal_blocks(operators, all_signals):
    """
    Creates a visual depiction of the signals blocks read by each operator
    group.

    Parameters
    ----------
    operators : list of tuple of `~nengo.builder.Operator`
        Operator execution plan
    all_signals : list of `~nengo.builder.Signal`
        Base signals arranged into some order

    Returns
    -------
    signal_blocks : str
        A string where each row corresponds to one operator group, and the
        non-blank characters in the line indicate that the operator group
        reads/writes that signal (with a number used to distinguish the
        different signal blocks within the operator group).
    """

    sig_idxs = {s: i for i, s in enumerate(all_signals)}
    output = np.asarray([[" " for _ in all_signals] for _ in operators])
    for n, group in enumerate(operators):
        for i in range(len(group[0].all_signals)):
            sig_group = [sig_idxs[op.all_signals[i].base] for op in group]
            output[n, sig_group] = str(i)

    return "\n".join("".join(line) for line in output)


# the default simplifications that will be applied. exposed as a variable here to make
# it easier for users to add something to the defaults (e.g.
# `simplifications=default_simplifications + (a_thing,)`), rather than having to
# manually specify and track changes to the defaults
default_simplifications = (
    remove_unmodified_resets,
    remove_zero_incs,
    remove_identity_muls,
    remove_constant_copies,
    remove_reset_incs,
)
