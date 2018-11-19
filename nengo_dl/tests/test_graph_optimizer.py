# pylint: disable=missing-docstring

from nengo.exceptions import BuildError
from nengo.neurons import LIF, LIFRate, Izhikevich, AdaptiveLIF
from nengo.synapses import Lowpass, Triangle, Alpha, LinearFilter
from nengo.builder.learning_rules import SimBCM
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import (SimPyFunc, DotInc, Copy, Reset,
                                    ElementwiseInc)
from nengo.builder.processes import SimProcess
from nengo.builder.signal import Signal
import numpy as np
import pytest

from nengo_dl import op_builders
from nengo_dl.graph_optimizer import (
    mergeable, greedy_planner, tree_planner, transitive_planner, noop_planner,
    order_signals, noop_order_signals,
    remove_unmodified_resets, remove_zero_incs, remove_constant_copies,
    remove_identity_muls)
from nengo_dl.tensor_node import SimTensorNode
from nengo_dl.tests import dummies


# pylint: disable=redefined-outer-name
@pytest.fixture(params=[greedy_planner, tree_planner, transitive_planner])
def planner(request):
    return request.param


def test_mergeable():
    # anything is mergeable with an empty list
    assert mergeable(None, [])

    # ops with different numbers of sets/incs/reads/updates are not mergeable
    assert not mergeable(dummies.Op(sets=[dummies.Signal()]), [dummies.Op()])
    assert not mergeable(dummies.Op(incs=[dummies.Signal()]), [dummies.Op()])
    assert not mergeable(dummies.Op(reads=[dummies.Signal()]), [dummies.Op()])
    assert not mergeable(dummies.Op(updates=[dummies.Signal()]), [dummies.Op()])
    assert mergeable(dummies.Op(sets=[dummies.Signal()]),
                     [dummies.Op(sets=[dummies.Signal()])])

    # check matching dtypes
    assert not mergeable(dummies.Op(sets=[dummies.Signal(dtype=np.float32)]),
                         [dummies.Op(sets=[dummies.Signal(dtype=np.float64)])])

    # shape mismatch
    assert not mergeable(dummies.Op(sets=[dummies.Signal(shape=(1, 2))]),
                         [dummies.Op(sets=[dummies.Signal(shape=(1, 3))])])

    # display shape mismatch
    assert not mergeable(
        dummies.Op(sets=[dummies.Signal(base_shape=(2, 2), shape=(4, 1))]),
        [dummies.Op(sets=[dummies.Signal(base_shape=(2, 2), shape=(1, 4))])])

    # first dimension mismatch
    assert mergeable(dummies.Op(sets=[dummies.Signal(shape=(3, 2))]),
                     [dummies.Op(sets=[dummies.Signal(shape=(4, 2))])])

    # Copy (inc must match)
    assert mergeable(Copy(dummies.Signal(), dummies.Signal(), inc=True),
                     [Copy(dummies.Signal(), dummies.Signal(), inc=True)])
    assert not mergeable(Copy(dummies.Signal(), dummies.Signal(), inc=True),
                         [Copy(dummies.Signal(), dummies.Signal(), inc=False)])

    # elementwise (first dimension must match)
    assert mergeable(
        ElementwiseInc(dummies.Signal(), dummies.Signal(), dummies.Signal()),
        [ElementwiseInc(dummies.Signal(), dummies.Signal(), dummies.Signal())])
    assert mergeable(
        ElementwiseInc(dummies.Signal(shape=(1,)), dummies.Signal(), dummies.Signal()),
        [ElementwiseInc(dummies.Signal(shape=()), dummies.Signal(), dummies.Signal())])
    assert not mergeable(
        ElementwiseInc(dummies.Signal(shape=(3,)), dummies.Signal(), dummies.Signal()),
        [ElementwiseInc(dummies.Signal(shape=(2,)), dummies.Signal(),
                        dummies.Signal())])

    # simpyfunc (t input must match)
    time = dummies.Signal()
    assert mergeable(SimPyFunc(None, None, time, None),
                     [SimPyFunc(None, None, time, None)])
    assert mergeable(SimPyFunc(None, None, None, dummies.Signal()),
                     [SimPyFunc(None, None, None, dummies.Signal())])
    assert not mergeable(SimPyFunc(None, None, dummies.Signal(), None),
                         [SimPyFunc(None, None, None, dummies.Signal())])

    # simneurons
    # check matching TF_NEURON_IMPL
    assert mergeable(SimNeurons(LIF(), dummies.Signal(), dummies.Signal()),
                     [SimNeurons(LIF(), dummies.Signal(), dummies.Signal())])
    assert not mergeable(SimNeurons(LIF(), dummies.Signal(), dummies.Signal()),
                         [SimNeurons(LIFRate(), dummies.Signal(), dummies.Signal())])

    # check custom with non-custom implementation
    assert not mergeable(SimNeurons(LIF(), dummies.Signal(), dummies.Signal()),
                         [SimNeurons(Izhikevich(), dummies.Signal(),
                                     dummies.Signal())])

    # check non-custom matching
    assert not mergeable(
        SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal()),
        [SimNeurons(AdaptiveLIF(), dummies.Signal(), dummies.Signal())])
    assert not mergeable(
        SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                   states=[dummies.Signal(dtype=np.float32)]),
        [SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                    states=[dummies.Signal(dtype=np.int32)])])
    assert mergeable(
        SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                   states=[dummies.Signal(shape=(3,))]),
        [SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                    states=[dummies.Signal(shape=(2,))])])
    assert not mergeable(
        SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                   states=[dummies.Signal(shape=(2, 1))]),
        [SimNeurons(Izhikevich(), dummies.Signal(), dummies.Signal(),
                    states=[dummies.Signal(shape=(2, 2))])])

    # simprocess
    # mode must match
    assert not mergeable(
        SimProcess(Lowpass(0), None, dummies.Signal(), dummies.Signal(),
                   mode="inc"),
        [SimProcess(Lowpass(0), None, dummies.Signal(), dummies.Signal(),
                    mode="set")])

    # check that lowpass match
    assert mergeable(SimProcess(Lowpass(0), None, None, dummies.Signal()),
                     [SimProcess(Lowpass(0), None, None, dummies.Signal())])

    # check that lowpass and linear don't match
    assert not mergeable(SimProcess(Lowpass(0), None, None, dummies.Signal()),
                         [SimProcess(Alpha(0), None, None, dummies.Signal())])

    # check that two linear do match
    assert mergeable(
        SimProcess(Alpha(0.1), dummies.Signal(), None, dummies.Signal()),
        [SimProcess(LinearFilter([1], [1, 1, 1]), dummies.Signal(), None,
                    dummies.Signal())])

    # check custom and non-custom don't match
    assert not mergeable(SimProcess(Triangle(0), None, None, dummies.Signal()),
                         [SimProcess(Alpha(0), None, None, dummies.Signal())])

    # check non-custom matching
    assert mergeable(SimProcess(Triangle(0), None, None, dummies.Signal()),
                     [SimProcess(Triangle(0), None, None, dummies.Signal())])

    # simtensornode
    a = SimTensorNode(None, dummies.Signal(), None, dummies.Signal())
    assert not mergeable(a, [a])

    # learning rules
    a = SimBCM(dummies.Signal((4,)), dummies.Signal(), dummies.Signal(), dummies.Signal(),
               dummies.Signal())
    b = SimBCM(dummies.Signal((5,)), dummies.Signal(), dummies.Signal(), dummies.Signal(),
               dummies.Signal())
    assert not mergeable(a, [b])


def test_planner_mergeable(planner):
    # check that mergeable operators are merged
    input0 = dummies.Signal()
    input1 = dummies.Signal()
    output0 = dummies.Signal()
    output1 = dummies.Signal()
    operators = [Copy(input0, output0, inc=True),
                 Copy(input1, output1, inc=True)]
    plan = planner(operators)
    assert len(plan) == 1
    assert type(plan[0][0]) == Copy
    assert len(plan[0]) == 2


def test_planner_unmergeable(planner):
    # check that non-mergeable operators aren't merged
    input0 = dummies.Signal()
    operators = [Copy(input0, dummies.Signal(dtype=np.float32)),
                 Copy(input0, dummies.Signal(dtype=np.int32))]
    plan = planner(operators)
    assert len(plan) == 2
    assert type(plan[0][0]) == Copy
    assert len(plan[0]) == 1
    assert type(plan[1][0]) == Copy
    assert len(plan[1]) == 1


def test_planner_chain(planner):
    # test a chain
    a = dummies.Signal(label="a")
    b = dummies.Signal(label="b")
    c = dummies.Signal(label="c")
    d = dummies.Signal(label="d")
    operators = [Copy(a, b, inc=True) for _ in range(3)]
    operators += [SimPyFunc(c, lambda x: x, None, b)]
    operators += [Copy(c, d, inc=True) for _ in range(2)]
    plan = planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 3
    assert len(plan[1]) == 1
    assert len(plan[2]) == 2


def test_planner_cycle(planner):
    inputs = [dummies.Signal() for _ in range(3)]
    operators = [Copy(inputs[0], inputs[1]), Copy(inputs[1], inputs[2]),
                 Copy(inputs[2], inputs[0])]

    with pytest.raises(BuildError):
        planner(operators)


def test_planner_size():
    # check that operators are selected according to number of available ops
    input0 = dummies.Signal()
    operators = [Copy(input0, dummies.Signal(), inc=True)
                 for _ in range(2)]
    operators += [Copy(input0, dummies.Signal())]
    operators += [DotInc(input0, dummies.Signal(), dummies.Signal())
                  for _ in range(3)]
    plan = greedy_planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 3
    assert len(plan[1]) == 2
    assert len(plan[2]) == 1


def test_noop_planner():
    inputs = [dummies.Signal() for _ in range(3)]
    operators = [Copy(inputs[1], inputs[2]), Copy(inputs[0], inputs[1])]

    plan = noop_planner(operators)
    assert len(plan) == len(operators)
    assert plan[0] == (operators[1],)
    assert plan[1] == (operators[0],)


def contiguous(sigs, all_signals):
    indices = sorted([all_signals.index(s) for s in sigs])
    return indices == list(range(np.min(indices), np.max(indices) + 1))


def ordered(ops, all_signals, block=None):
    reads = {}
    for op in ops:
        reads[op] = [x for x in op.reads]
        if type(op) == SimNeurons:
            reads[op] += op.states
        elif type(op) == SimProcess and isinstance(op.process, Lowpass):
            reads[op] += op.updates

    if block is None:
        read_indices = [
            [all_signals.index(reads[op][i].base) * 10000 +
             reads[op][i].elemoffset for op in ops]
            for i in range(len(ops[0].reads))]
    else:
        read_indices = [
            [all_signals.index(reads[op][block].base) * 10000 +
             reads[op][block].elemoffset for op in ops]]

    return np.all(np.diff(read_indices, axis=1) > 0)


def test_order_signals_disjoint():
    # disjoint reads
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]

    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(5)),
        tuple(dummies.Op(reads=[inputs[5 + i]]) for i in range(5))]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_partial():
    # partially overlapping reads

    # two overlapping sets (A, A/B, B)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(4)),
        tuple(dummies.Op(reads=[inputs[2 + i]]) for i in range(4))]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:4], sigs)
    assert contiguous(inputs[2:6], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_partial2():
    # more complex partial overlap
    # (A, A/B, B/C, C)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(5)),
        tuple(dummies.Op(reads=[inputs[2 + i]]) for i in range(4)),
        tuple(dummies.Op(reads=[inputs[5 + i]]) for i in range(3)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert contiguous(inputs[2:6], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert ordered(new_plan[2], sigs)


def test_order_signals_partial3():
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in [0, 1, 2, 3]),
        tuple(dummies.Op(reads=[inputs[i]]) for i in [0, 4, 7]),
        tuple(dummies.Op(reads=[inputs[i]]) for i in [5, 6, 7])]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:4], sigs)
    assert contiguous([inputs[0], inputs[4], inputs[7]], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert ordered(new_plan[2], sigs)


def test_order_signals_partial_unsatisfiable():
    # this one will be unsatisfied, because after A it will pick A/B (because
    # B is the next biggest block). technically this could be satisfied if
    # we picked A/C next, but is there a way we could know that?
    # (A, A/B, A/C, B)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(7)),
        tuple(dummies.Op(reads=[inputs[5 + i]]) for i in range(5)),
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(3)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:7], sigs)
    assert not contiguous(inputs[5:], sigs)
    assert contiguous(inputs[:3], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[2], sigs)


def test_order_signals_subset():
    # ordering in which one read block is fully nested within another
    # (A, A/B)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(10)),
        tuple(dummies.Op(reads=[inputs[4 - i]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[:10], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_multiread():
    # signal sorting with operators that read from multiple signals
    # (no overlap)
    # (A, B, C) (where A and C are from the same op)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(3, 5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:3], sigs)
    assert contiguous(inputs[3:5], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_multiread_complex():
    # signal sorting with operators that read from multiple signals
    # (overlapping)
    # (C, B/C, A) (where A and B are from the same op)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(dummies.Op(reads=[inputs[i + 5]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:3], sigs)
    assert contiguous(inputs[5:], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_multiread_complex2():
    # (B, B/A, A, A/C, C) (where A and B are from the same op)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[2 + i], inputs[i]]) for i in range(4)),
        tuple(dummies.Op(reads=[inputs[5 + i]]) for i in range(3)),
    ]
    sigs, new_plan = order_signals(plan)

    assert contiguous(inputs[5:8], sigs)
    assert ordered(new_plan[1], sigs)

    # TODO: technically it is always possible to order both blocks properly,
    # but it requires you to know which of the two equally sized blocks should
    # have priority, and I'm not sure there's a way to determine that.
    assert (contiguous(inputs[:4], sigs) or
            contiguous(inputs[2:6], sigs))
    assert (ordered(new_plan[0], sigs, block=0) or
            ordered(new_plan[0], sigs, block=1))


def test_order_signals_multiread_unsatisfiable():
    # unsatisfiable order for block C (conflicts with A, which gets prioritized
    # because it is in a larger group)
    # (A, A/C, B, B/D)
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i], inputs[5 + i]]) for i in range(5)),
        tuple(dummies.Op(reads=[inputs[1 - i], inputs[5 + i]]) for i in range(2)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:], sigs)
    assert contiguous(inputs[:2], sigs)
    assert contiguous(inputs[5:7], sigs)
    assert ordered(new_plan[0], sigs)
    assert (ordered(new_plan[1], sigs, block=0) or
            ordered(new_plan[1], sigs, block=1))
    assert not ordered(new_plan[1], sigs)


def test_order_signals_views():
    base = dummies.Signal(shape=(6,), label="base")
    sig = dummies.Signal(shape=(7,), label="sig")
    sig2 = dummies.Signal(shape=(7,), label="sig2")
    views = [dummies.Signal(shape=(1,), base_shape=(5,), offset=1 + i,
                            label="view_%d" % i)
             for i in range(5)]
    for v in views:
        v.base = base
    plan = [
        (dummies.Op(reads=[base]), dummies.Op(reads=[views[1]]),
         dummies.Op(reads=[views[0]]), dummies.Op(reads=[sig2])),
        (dummies.Op(reads=[base]), dummies.Op(reads=[sig])),
        tuple(dummies.Op(reads=[views[i]]) for i in range(4, 2, -1)),
        (dummies.Op(reads=[views[4]]), dummies.Op(reads=[sig]))]
    sigs, new_plan = order_signals(plan)
    assert contiguous([base, sig, sig2], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert ordered(new_plan[2], sigs)
    assert ordered(new_plan[3], sigs)


def test_order_signals_duplicates():
    # test where read blocks contain duplicate signals
    inputs = [dummies.Signal(label=str(i)) for i in range(4)]
    plan = [
        tuple(dummies.Op(reads=[inputs[0]]) for _ in range(2)) +
        (dummies.Op(reads=[inputs[2]]),),
        tuple(dummies.Op(reads=[inputs[1]]) for _ in range(2)) +
        (dummies.Op(reads=[inputs[3]]),)
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous([inputs[0], inputs[2]], sigs)
    assert contiguous([inputs[1], inputs[3]], sigs)

    # note: not possible for these to be in increasing order, since they
    # contain duplicates
    assert not ordered(new_plan[0], sigs)
    assert not ordered(new_plan[1], sigs)


def test_order_signals_noreads():
    # test with ops that don't have any reads
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i]]) for i in range(5)),
        tuple(dummies.Op(sets=[inputs[5 + i]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert ordered(new_plan[0], sigs)


def test_order_signals_neuron_states():
    # test with neuron states (should be treated as reads)

    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(SimNeurons(None, inputs[0], inputs[1], states=[x])
              for x in inputs[2::2]),
        tuple(SimNeurons(None, inputs[0], inputs[1], states=[x])
              for x in inputs[3::2])]
    sigs, new_plan = order_signals(plan)

    assert contiguous(inputs[2::2], sigs)
    assert contiguous(inputs[3::2], sigs)
    # note: block=0 is just a single signal, so it's always "ordered"
    assert ordered(new_plan[0], sigs, block=1)
    assert ordered(new_plan[1], sigs, block=1)


def test_order_signals_lowpass():
    # test that lowpass outputs are ordered as reads

    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    time = dummies.Signal()
    plan = [
        tuple(SimProcess(Lowpass(0.1), inputs[i], inputs[i + 1], time,
                         mode="update") for i in range(0, 4, 2)),
        tuple(SimProcess(Lowpass(0.1), inputs[i], inputs[i + 1], time,
                         mode="update") for i in range(5, 9, 2))]
    sigs, new_plan = order_signals(plan)

    assert contiguous(inputs[1:5:2], sigs)
    assert contiguous(inputs[6:10:2], sigs)

    assert ordered(new_plan[0], sigs, block=1)
    assert ordered(new_plan[0], sigs, block=2)
    assert ordered(new_plan[1], sigs, block=1)
    assert ordered(new_plan[1], sigs, block=2)


def test_order_signals_duplicate_read_blocks():
    # test that order_signal prioritizes read blocks that are duplicated in
    # multiple op groups
    inputs = [dummies.Signal(label=str(i)) for i in range(10)]
    plan = [
        tuple(dummies.Op(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(dummies.Op(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(dummies.Op(reads=[inputs[5 + i], inputs[4 - i]]) for i in range(5))]

    sigs, new_plan = order_signals(plan)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert (ordered(new_plan[2], sigs, block=0) or
            ordered(new_plan[2], sigs, block=1))
    assert not ordered(new_plan[2], sigs)


def test_noop_order_signals():
    inputs = [dummies.Signal(label="a"), dummies.Signal(label="b"),
              dummies.Signal(label="c", base_shape=(2,))]
    plan = [(dummies.Op(reads=[x]),) for x in inputs]

    sigs, new_plan = noop_order_signals(plan)

    assert all(x == y for x, y in zip(plan, new_plan))
    assert len(sigs) == 3

    sigs.remove(inputs[0])
    sigs.remove(inputs[1])
    assert sigs[0].name == "c.base"


def test_remove_unmodified_resets():
    a = Signal([1])

    # check that unmodified reset gets removed
    operators = [Reset(a, 2)]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == []
    assert np.all(a.initial_value == 2)

    # check that reset + inc doesn't get removed
    operators = [Reset(a, 2), dummies.Op(incs=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators

    # check that reset + update doesn't get removed
    operators = [Reset(a, 2), dummies.Op(updates=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators

    # check that reset + read does get removed
    operators = [Reset(a, 3), dummies.Op(reads=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators[1:]
    assert np.all(a.initial_value == 3)


def test_remove_zero_incs():
    # check that zero inputs get removed (for A or X)
    operators = [DotInc(dummies.Signal(), dummies.Signal(initial_value=1),
                        dummies.Signal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == []

    operators = [DotInc(dummies.Signal(initial_value=1), dummies.Signal(),
                        dummies.Signal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == []

    # check that zero inputs (copy) get removed
    operators = [Copy(dummies.Signal(), dummies.Signal(), dummies.Signal(), inc=True)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == []

    # check that node inputs don't get removed
    x = dummies.Signal(label="<Node lorem ipsum")
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that zero inputs + trainable don't get removed
    x = dummies.Signal()
    x.trainable = True
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that updated input doesn't get removed
    x = dummies.Signal()
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal()),
                 dummies.Op(updates=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that inc'd input doesn't get removed
    x = dummies.Signal()
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal()),
                 dummies.Op(incs=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that set'd input doesn't get removed
    x = dummies.Signal()
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal()),
                 dummies.Op(sets=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that Reset(0) input does get removed
    x = dummies.Signal()
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal()),
                 Reset(x)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators[1:]

    # check that Reset(1) input does not get removed
    x = dummies.Signal()
    operators = [DotInc(dummies.Signal(initial_value=1), x, dummies.Signal()),
                 Reset(x, 1)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that set's get turned into a reset
    x = dummies.Signal()
    operators = [Copy(dummies.Signal(), x)]
    new_operators = remove_zero_incs(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is x
    assert new_operators[0].value == 0


def test_remove_constant_copies():
    # check that Copy with no inputs gets turned into Reset
    x = dummies.Signal()
    operators = [Copy(dummies.Signal(), x)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is x
    assert new_operators[0].value == 0

    # check that Copy with Node input doesn't get changed
    x = dummies.Signal(label="<Node lorem ipsum")
    operators = [Copy(x, dummies.Signal())]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check that Copy with trainable input doesn't get changed
    x = dummies.Signal()
    x.trainable = True
    operators = [Copy(x, dummies.Signal())]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with updated input doesn't get changed
    x = dummies.Signal()
    operators = [Copy(x, dummies.Signal()), dummies.Op(updates=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with inc'd input doesn't get changed
    x = dummies.Signal()
    operators = [Copy(x, dummies.Signal()), dummies.Op(incs=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with set input doesn't get changed
    x = dummies.Signal()
    operators = [Copy(x, dummies.Signal()), dummies.Op(sets=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with read input/output does get changed
    x = dummies.Signal()
    y = dummies.Signal()
    operators = [Copy(x, y), dummies.Op(reads=[x]),
                 dummies.Op(reads=[y])]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 3
    assert new_operators[1:] == operators[1:]
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is y
    assert new_operators[0].value == 0

    # check Copy with Reset input does get changed
    x = dummies.Signal()
    y = dummies.Signal()
    operators = [Copy(x, y), Reset(x, 2)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is y
    assert new_operators[0].value == 2

    # check that slicing is respected
    x = dummies.Signal()
    y = Signal(initial_value=[0, 0])
    operators = [Copy(x, y, dst_slice=slice(1, 2)), Reset(x, 2)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst.shape == (1,)
    assert new_operators[0].dst.is_view
    assert new_operators[0].dst.elemoffset == 1
    assert new_operators[0].dst.base is y
    assert new_operators[0].value == 2

    # check that CopyInc gets turned into ResetInc
    x = dummies.Signal()
    y = dummies.Signal()
    operators = [Copy(x, y, inc=True), Reset(x, 2)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], op_builders.ResetInc)
    assert new_operators[0].dst is y
    assert new_operators[0].value == 2
    assert len(new_operators[0].incs) == 1
    assert len(new_operators[0].sets) == 0


@pytest.mark.parametrize("Op", [DotInc, ElementwiseInc])
def test_remove_identity_muls(Op):
    # check that identity input signals get removed
    As = [1.0, np.diag(np.ones(3)) if Op == DotInc else np.ones(3)]
    for A in As:
        x = dummies.Signal(shape=(1,) if isinstance(A, float) else A.shape[:1])
        y = dummies.Signal(shape=(1,) if isinstance(A, float) else A.shape[:1])
        a = Signal(A)
        a.trainable = False
        operators = [Op(a, x, y)]
        new_operators = remove_identity_muls(operators)
        assert len(new_operators) == 1
        new_op = new_operators[0]
        assert isinstance(new_op, Copy)
        assert new_op.src is x
        assert new_op.dst is y
        assert new_op.inc

    # check that identity x gets removed for elementwiseinc
    if Op == ElementwiseInc:
        a = dummies.Signal()
        x = dummies.Signal(initial_value=1)
        y = dummies.Signal()
        operators = [Op(a, x, y)]
        new_operators = remove_identity_muls(operators)
        assert len(operators) == 1
        new_op = new_operators[0]
        assert isinstance(new_op, Copy)
        assert new_op.src is a
        assert new_op.dst is y
        assert new_op.inc

    # check that reset inputs get removed
    for A in As:
        x = dummies.Signal(shape=(1,) if isinstance(A, float) else A.shape[:1])
        y = dummies.Signal(shape=(1,) if isinstance(A, float) else A.shape[:1])
        a = dummies.Signal(shape=(1,) if isinstance(A, float) else A.shape)
        r = Reset(a)
        r.value = A
        operators = [Op(a, x, y), r]
        new_operators = remove_identity_muls(operators)
        assert len(new_operators) == 2
        assert new_operators[1:] == operators[1:]
        new_op = new_operators[0]
        assert isinstance(new_op, Copy)
        assert new_op.src is x
        assert new_op.dst is y
        assert new_op.inc

    # check that non-identity inputs don't get removed
    a = Signal(np.ones((3, 3)))
    a.trainable = False
    operators = [Op(a, dummies.Signal(shape=(3,)),
                    dummies.Signal(shape=(3,)))]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that node inputs don't get removed
    x = dummies.Signal(label="<Node lorem ipsum")
    operators = [Op(x, dummies.Signal(), dummies.Signal())]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that identity inputs + trainable don't get removed
    x = Signal(1.0)
    x.trainable = True
    operators = [Op(x, dummies.Signal(), dummies.Signal())]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that updated input doesn't get removed
    x = dummies.Signal()
    operators = [Op(x, dummies.Signal(), dummies.Signal()),
                 dummies.Op(updates=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that inc'd input doesn't get removed
    x = dummies.Signal()
    operators = [Op(x, dummies.Signal(), dummies.Signal()),
                 dummies.Op(incs=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that set'd input doesn't get removed
    x = dummies.Signal()
    operators = [Op(x, dummies.Signal(), dummies.Signal()),
                 dummies.Op(sets=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators
