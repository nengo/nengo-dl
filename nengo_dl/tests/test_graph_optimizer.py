from nengo.exceptions import BuildError
from nengo.neurons import LIF, LIFRate, Izhikevich, AdaptiveLIF
from nengo.synapses import Lowpass, Triangle, Alpha
from nengo.builder.learning_rules import SimBCM
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import (SimPyFunc, DotInc, Copy, Reset,
                                    ElementwiseInc)
from nengo.builder.processes import SimProcess
from nengo.builder.signal import Signal
import numpy as np
import pytest

from nengo_dl import builder, nengo_version, op_builders
from nengo_dl.graph_optimizer import (
    mergeable, greedy_planner, tree_planner, transitive_planner, noop_planner,
    order_signals, noop_order_signals, create_signals,
    remove_unmodified_resets, remove_zero_incs, remove_constant_copies,
    remove_identity_muls)
from nengo_dl.tensor_node import SimTensorNode


@pytest.fixture(
    params=[greedy_planner, tree_planner] +
    ([transitive_planner] if nengo_version >= (2, 4, 0) else []))
def planner(request):
    return request.param


class DummySignal(object):
    def __init__(self, shape=None, dtype=None, base_shape=None, offset=0,
                 trainable=False, label=""):
        self.shape = (1,) if shape is None else shape
        self.dtype = np.float32 if dtype is None else dtype
        self.base = (self if base_shape is None else
                     DummySignal(shape=base_shape, dtype=self.dtype,
                                 label="%s.base" % label))
        self.elemoffset = offset
        self.name = label
        self.ndim = len(self.shape)
        self.is_view = base_shape is not None
        self.size = np.prod(self.shape)
        self.trainable = trainable
        self.minibatched = not trainable

    @property
    def initial_value(self):
        return np.zeros(self.base.shape, self.dtype)

    def may_share_memory(self, other):
        return False

    def __repr__(self):
        return "DummySignal(%s)" % self.name


class DummyOp(object):
    def __init__(self, sets=None, incs=None, reads=None, updates=None):
        self.sets = [] if sets is None else sets
        self.incs = [] if incs is None else incs
        self.reads = [] if reads is None else reads
        self.updates = [] if updates is None else updates

        self.all_signals = self.sets + self.incs + self.reads + self.updates

    def __repr__(self):
        rep = "DummyOp("
        if len(self.sets) > 0:
            rep += "sets=%s" % self.sets
        if len(self.incs) > 0:
            rep += "incs=%s" % self.incs
        if len(self.reads) > 0:
            rep += "reads=%s" % self.reads
        if len(self.updates) > 0:
            rep += "updates=%s" % self.updates
        rep += ")"
        return rep


@builder.Builder.register(DummyOp)
class DummyBuilder(builder.OpBuilder):
    pass


def test_mergeable():
    # anything is mergeable with an empty list
    assert mergeable(None, [])

    # ops with different numbers of sets/incs/reads/updates are not mergeable
    assert not mergeable(DummyOp(sets=[DummySignal()]), [DummyOp()])
    assert not mergeable(DummyOp(incs=[DummySignal()]), [DummyOp()])
    assert not mergeable(DummyOp(reads=[DummySignal()]), [DummyOp()])
    assert not mergeable(DummyOp(updates=[DummySignal()]), [DummyOp()])
    assert mergeable(DummyOp(sets=[DummySignal()]),
                     [DummyOp(sets=[DummySignal()])])

    # check matching dtypes
    assert not mergeable(DummyOp(sets=[DummySignal(dtype=np.float32)]),
                         [DummyOp(sets=[DummySignal(dtype=np.float64)])])

    # shape mismatch
    assert not mergeable(DummyOp(sets=[DummySignal(shape=(1, 2))]),
                         [DummyOp(sets=[DummySignal(shape=(1, 3))])])

    # display shape mismatch
    assert not mergeable(
        DummyOp(sets=[DummySignal(base_shape=(2, 2), shape=(4, 1))]),
        [DummyOp(sets=[DummySignal(base_shape=(2, 2), shape=(1, 4))])])

    # first dimension mismatch
    assert mergeable(DummyOp(sets=[DummySignal(shape=(3, 2))]),
                     [DummyOp(sets=[DummySignal(shape=(4, 2))])])

    # Copy (inc must match)
    assert mergeable(Copy(DummySignal(), DummySignal(), inc=True),
                     [Copy(DummySignal(), DummySignal(), inc=True)])
    assert not mergeable(Copy(DummySignal(), DummySignal(), inc=True),
                         [Copy(DummySignal(), DummySignal(), inc=False)])

    # elementwise (first dimension must match)
    assert mergeable(
        ElementwiseInc(DummySignal(), DummySignal(), DummySignal()),
        [ElementwiseInc(DummySignal(), DummySignal(), DummySignal())])
    assert mergeable(
        ElementwiseInc(DummySignal(shape=(1,)), DummySignal(), DummySignal()),
        [ElementwiseInc(DummySignal(shape=()), DummySignal(), DummySignal())])
    assert not mergeable(
        ElementwiseInc(DummySignal(shape=(3,)), DummySignal(), DummySignal()),
        [ElementwiseInc(DummySignal(shape=(2,)), DummySignal(),
                        DummySignal())])

    # simpyfunc (t input must match)
    time = DummySignal()
    assert mergeable(SimPyFunc(None, None, time, None),
                     [SimPyFunc(None, None, time, None)])
    assert mergeable(SimPyFunc(None, None, None, DummySignal()),
                     [SimPyFunc(None, None, None, DummySignal())])
    assert not mergeable(SimPyFunc(None, None, DummySignal(), None),
                         [SimPyFunc(None, None, None, DummySignal())])

    # simneurons
    # check matching TF_NEURON_IMPL
    assert mergeable(SimNeurons(LIF(), DummySignal(), DummySignal()),
                     [SimNeurons(LIF(), DummySignal(), DummySignal())])
    assert not mergeable(SimNeurons(LIF(), DummySignal(), DummySignal()),
                         [SimNeurons(LIFRate(), DummySignal(), DummySignal())])

    # check custom with non-custom implementation
    assert not mergeable(SimNeurons(LIF(), DummySignal(), DummySignal()),
                         [SimNeurons(Izhikevich(), DummySignal(),
                                     DummySignal())])

    # check non-custom matching
    assert not mergeable(
        SimNeurons(Izhikevich(), DummySignal(), DummySignal()),
        [SimNeurons(AdaptiveLIF(), DummySignal(), DummySignal())])
    assert not mergeable(
        SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                   states=[DummySignal(dtype=np.float32)]),
        [SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                    states=[DummySignal(dtype=np.int32)])])
    assert mergeable(
        SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                   states=[DummySignal(shape=(3,))]),
        [SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                    states=[DummySignal(shape=(2,))])])
    assert not mergeable(
        SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                   states=[DummySignal(shape=(2, 1))]),
        [SimNeurons(Izhikevich(), DummySignal(), DummySignal(),
                    states=[DummySignal(shape=(2, 2))])])

    # simprocess
    # mode must match
    assert not mergeable(
        SimProcess(Lowpass(0), None, None, DummySignal(), mode="inc"),
        [SimProcess(Lowpass(0), None, None, DummySignal(), mode="set")])

    # check matching TF_PROCESS_IMPL
    # note: we only have one item in TF_PROCESS_IMPL at the moment, so no
    # such thing as a mismatch
    assert mergeable(SimProcess(Lowpass(0), None, None, DummySignal()),
                     [SimProcess(Lowpass(0), None, None, DummySignal())])

    # check custom vs non custom
    assert not mergeable(SimProcess(Lowpass(0), None, None, DummySignal()),
                         [SimProcess(Alpha(0), None, None, DummySignal())])

    # check non-custom matching
    assert mergeable(SimProcess(Triangle(0), None, None, DummySignal()),
                     [SimProcess(Alpha(0), None, None, DummySignal())])

    # simtensornode
    a = SimTensorNode(None, DummySignal(), None, DummySignal())
    assert not mergeable(a, [a])

    # learning rules
    a = SimBCM(DummySignal((4,)), DummySignal(), DummySignal(), DummySignal(),
               DummySignal())
    b = SimBCM(DummySignal((5,)), DummySignal(), DummySignal(), DummySignal(),
               DummySignal())
    assert not mergeable(a, [b])


def test_planner_mergeable(planner):
    # check that mergeable operators are merged
    input0 = DummySignal()
    input1 = DummySignal()
    output0 = DummySignal()
    output1 = DummySignal()
    operators = [Copy(input0, output0, inc=True),
                 Copy(input1, output1, inc=True)]
    plan = planner(operators)
    assert len(plan) == 1
    assert type(plan[0][0]) == Copy
    assert len(plan[0]) == 2


def test_planner_unmergeable(planner):
    # check that non-mergeable operators aren't merged
    input0 = DummySignal()
    operators = [Copy(input0, DummySignal(dtype=np.float32)),
                 Copy(input0, DummySignal(dtype=np.int32))]
    plan = planner(operators)
    assert len(plan) == 2
    assert type(plan[0][0]) == Copy
    assert len(plan[0]) == 1
    assert type(plan[1][0]) == Copy
    assert len(plan[1]) == 1


def test_planner_chain(planner):
    # test a chain
    a = DummySignal(label="a")
    b = DummySignal(label="b")
    c = DummySignal(label="c")
    d = DummySignal(label="d")
    operators = [Copy(a, b, inc=True) for _ in range(3)]
    operators += [SimPyFunc(c, lambda x: x, None, b)]
    operators += [Copy(c, d, inc=True) for _ in range(2)]
    plan = planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 3
    assert len(plan[1]) == 1
    assert len(plan[2]) == 2


def test_planner_cycle(planner):
    inputs = [DummySignal() for _ in range(3)]
    operators = [Copy(inputs[0], inputs[1]), Copy(inputs[1], inputs[2]),
                 Copy(inputs[2], inputs[0])]

    with pytest.raises(BuildError):
        planner(operators)


def test_planner_size():
    # check that operators are selected according to number of available ops
    input0 = DummySignal()
    operators = [Copy(input0, DummySignal(), inc=True)
                 for _ in range(2)]
    operators += [Copy(input0, DummySignal())]
    operators += [DotInc(input0, DummySignal(), DummySignal())
                  for _ in range(3)]
    plan = greedy_planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 3
    assert len(plan[1]) == 2
    assert len(plan[2]) == 1


def test_noop_planner():
    inputs = [DummySignal() for _ in range(3)]
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]

    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[5 + i]]) for i in range(5))]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_partial():
    # partially overlapping reads

    # two overlapping sets (A, A/B, B)
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(4)),
        tuple(DummyOp(reads=[inputs[2 + i]]) for i in range(4))]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:4], sigs)
    assert contiguous(inputs[2:6], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_partial2():
    # more complex partial overlap
    # (A, A/B, B/C, C)
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[2 + i]]) for i in range(4)),
        tuple(DummyOp(reads=[inputs[5 + i]]) for i in range(3)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert contiguous(inputs[2:6], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert ordered(new_plan[2], sigs)


def test_order_signals_partial3():
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in [0, 1, 2, 3]),
        tuple(DummyOp(reads=[inputs[i]]) for i in [0, 4, 7]),
        tuple(DummyOp(reads=[inputs[i]]) for i in [5, 6, 7])]
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(7)),
        tuple(DummyOp(reads=[inputs[5 + i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[i]]) for i in range(3)),
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(10)),
        tuple(DummyOp(reads=[inputs[4 - i]]) for i in range(5)),
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(DummyOp(reads=[inputs[i]]) for i in range(3, 5)),
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(DummyOp(reads=[inputs[i + 5]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:3], sigs)
    assert contiguous(inputs[5:], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)


def test_order_signals_multiread_complex2():
    # (B, B/A, A, A/C, C) (where A and B are from the same op)
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[2 + i], inputs[i]]) for i in range(4)),
        tuple(DummyOp(reads=[inputs[5 + i]]) for i in range(3)),
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i], inputs[5 + i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[1 - i], inputs[5 + i]]) for i in range(2)),
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
    base = DummySignal(shape=(6,), label="base")
    sig = DummySignal(shape=(7,), label="sig")
    sig2 = DummySignal(shape=(7,), label="sig2")
    views = [DummySignal(shape=(1,), base_shape=(5,), offset=1 + i,
                         label="view_%d" % i)
             for i in range(5)]
    for v in views:
        v.base = base
    plan = [
        (DummyOp(reads=[base]), DummyOp(reads=[views[1]]),
         DummyOp(reads=[views[0]]), DummyOp(reads=[sig2])),
        (DummyOp(reads=[base]), DummyOp(reads=[sig])),
        tuple(DummyOp(reads=[views[i]]) for i in range(4, 2, -1)),
        (DummyOp(reads=[views[4]]), DummyOp(reads=[sig]))]
    sigs, new_plan = order_signals(plan)
    assert contiguous([base, sig, sig2], sigs)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert ordered(new_plan[2], sigs)
    assert ordered(new_plan[3], sigs)


def test_order_signals_duplicates():
    # test where read blocks contain duplicate signals
    inputs = [DummySignal(label=str(i)) for i in range(4)]
    plan = [
        tuple(DummyOp(reads=[inputs[0]]) for _ in range(2)) +
        (DummyOp(reads=[inputs[2]]),),
        tuple(DummyOp(reads=[inputs[1]]) for _ in range(2)) +
        (DummyOp(reads=[inputs[3]]),)
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(5)),
        tuple(DummyOp(sets=[inputs[5 + i]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert ordered(new_plan[0], sigs)


def test_order_signals_neuron_states():
    # test with neuron states (should be treated as reads)

    inputs = [DummySignal(label=str(i)) for i in range(10)]
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

    inputs = [DummySignal(label=str(i)) for i in range(10)]
    time = DummySignal()
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
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(DummyOp(reads=[inputs[i], inputs[5 + i]]) for i in range(3)),
        tuple(DummyOp(reads=[inputs[5 + i], inputs[4 - i]]) for i in range(5))]

    sigs, new_plan = order_signals(plan)
    assert ordered(new_plan[0], sigs)
    assert ordered(new_plan[1], sigs)
    assert (ordered(new_plan[2], sigs, block=0) or
            ordered(new_plan[2], sigs, block=1))
    assert not ordered(new_plan[2], sigs)


def test_noop_order_signals():
    inputs = [DummySignal(label="a"), DummySignal(label="b"),
              DummySignal(label="c", base_shape=(2,))]
    plan = [(DummyOp(reads=[x]),) for x in inputs]

    sigs, new_plan = noop_order_signals(plan)

    assert all(x == y for x, y in zip(plan, new_plan))
    assert len(sigs) == 3

    sigs.remove(inputs[0])
    sigs.remove(inputs[1])
    assert sigs[0].name == "c.base"


def test_create_signals():
    # check that floats/ints get split into different arrays
    sigs = [DummySignal(dtype=np.float32), DummySignal(dtype=np.float32),
            DummySignal(dtype=np.int32), DummySignal(dtype=np.int32)]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that floats all get converted to same precision and combined
    sigs = [DummySignal(dtype=np.float32), DummySignal(dtype=np.float32),
            DummySignal(dtype=np.float64), DummySignal(dtype=np.float64)]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert np.all([sig_map[x].dtype == np.float32 for x in sigs])
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key == sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that ints all get converted to same precision and combined
    sigs = [DummySignal(dtype=np.int32), DummySignal(dtype=np.int32),
            DummySignal(dtype=np.int64), DummySignal(dtype=np.int64)]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert np.all([sig_map[x].dtype == np.int32 for x in sigs])
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key == sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that different shapes go in different groups
    sigs = [DummySignal(shape=(10,)), DummySignal(shape=(5,)),
            DummySignal(shape=(10, 1)), DummySignal(shape=(5, 1))]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert bases[sig_map[sigs[0]].key][0].shape == (15, 10)
    assert bases[sig_map[sigs[2]].key][0].shape == (15, 1, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check trainable
    sigs = [DummySignal(trainable=True), DummySignal(trainable=True),
            DummySignal(trainable=False), DummySignal(trainable=False)]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert bases[sig_map[sigs[0]].key][0].shape == (2,)
    assert bases[sig_map[sigs[2]].key][0].shape == (2, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that scalars get upsized
    sigs = [DummySignal(shape=()), DummySignal(shape=(4,))]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert list(bases.values())[0][0].shape == (5, 10)


def test_create_signals_views():
    sigs = [DummySignal(shape=(2, 2), base_shape=(4,)),
            DummySignal(shape=(2, 2), base_shape=(4,))]
    sigs += [sigs[0].base, sigs[1].base]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs)]
    bases, sig_map = create_signals(sigs[2:], plan, np.float32, 10)
    assert list(bases.values())[0][0].shape == (8, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key == sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key
    assert np.all(sig_map[sigs[0]].indices == (0, 1, 2, 3))
    assert np.all(sig_map[sigs[1]].indices == (4, 5, 6, 7))
    assert np.all(sig_map[sigs[0]].indices == sig_map[sigs[2]].indices)
    assert np.all(sig_map[sigs[1]].indices == sig_map[sigs[3]].indices)


def test_create_signals_partition():
    # check that signals are partitioned based on plan
    sigs = [DummySignal(), DummySignal(),
            DummySignal(), DummySignal()]
    plan = [tuple(DummyOp(reads=[x]) for x in sigs[:2]),
            tuple(DummyOp(reads=[x]) for x in sigs[2:])]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that signals are partioned for different read blocks
    plan = [tuple(DummyOp(reads=[sigs[i], sigs[2 + i]]) for i in range(2))]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that signals are partioned for different sig types
    plan = [tuple(DummyOp(reads=[sigs[i]], sets=[sigs[2 + i]])
                  for i in range(2))]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert sig_map[sigs[0]].key == sig_map[sigs[1]].key
    assert sig_map[sigs[1]].key != sig_map[sigs[2]].key
    assert sig_map[sigs[2]].key == sig_map[sigs[3]].key

    # check that resets are ignored
    sigs = [DummySignal(), DummySignal(), DummySignal(), DummySignal()]
    plan = [tuple(Reset(x) for x in sigs)]
    bases, sig_map = create_signals(sigs, plan, np.float32, 10)
    assert len(bases) == 4


def test_remove_unmodified_resets():
    a = Signal([1])

    # check that unmodified reset gets removed
    operators = [Reset(a, 2)]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == []
    assert np.all(a.initial_value == 2)

    # check that reset + inc doesn't get removed
    operators = [Reset(a, 2), DummyOp(incs=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators

    # check that reset + update doesn't get removed
    operators = [Reset(a, 2), DummyOp(updates=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators

    # check that reset + read does get removed
    operators = [Reset(a, 3), DummyOp(reads=[a])]
    new_ops = remove_unmodified_resets(operators)
    assert new_ops == operators[1:]
    assert np.all(a.initial_value == 3)


def test_remove_zero_incs():
    # check that zero inputs get removed
    operators = [DotInc(DummySignal(), DummySignal(), DummySignal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == []

    # check that zero inputs (copy) get removed
    operators = [Copy(DummySignal(), DummySignal(), DummySignal(), inc=True)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == []

    # check that node inputs don't get removed
    x = DummySignal(label="<Node lorem ipsum.out")
    operators = [DotInc(DummySignal(), x, DummySignal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that zero inputs + trainable don't get removed
    x = DummySignal()
    x.trainable = True
    operators = [DotInc(DummySignal(), x, DummySignal())]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that updated input doesn't get removed
    x = DummySignal()
    operators = [DotInc(DummySignal(), x, DummySignal()), DummyOp(updates=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that inc'd input doesn't get removed
    x = DummySignal()
    operators = [DotInc(DummySignal(), x, DummySignal()), DummyOp(incs=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that set'd input doesn't get removed
    x = DummySignal()
    operators = [DotInc(DummySignal(), x, DummySignal()), DummyOp(sets=[x])]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that Reset(0) input does get removed
    x = DummySignal()
    operators = [DotInc(DummySignal(), x, DummySignal()), Reset(x)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators[1:]

    # check that Reset(1) input does not get removed
    x = DummySignal()
    operators = [DotInc(DummySignal(), x, DummySignal()), Reset(x, 1)]
    new_operators = remove_zero_incs(operators)
    assert new_operators == operators

    # check that set's get turned into a reset
    x = DummySignal()
    operators = [Copy(DummySignal(), x)]
    new_operators = remove_zero_incs(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is x
    assert new_operators[0].value == 0


def test_remove_constant_copies():
    # check that Copy with no inputs gets turned into Reset
    x = DummySignal()
    operators = [Copy(DummySignal(), x)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is x
    assert new_operators[0].value == 0

    # check that Copy with Node input doesn't get changed
    x = DummySignal(label="<Node lorem ipsum.out")
    operators = [Copy(x, DummySignal())]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check that Copy with trainable input doesn't get changed
    x = DummySignal()
    x.trainable = True
    operators = [Copy(x, DummySignal())]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with updated input doesn't get changed
    x = DummySignal()
    operators = [Copy(x, DummySignal()), DummyOp(updates=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with inc'd input doesn't get changed
    x = DummySignal()
    operators = [Copy(x, DummySignal()), DummyOp(incs=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with set input doesn't get changed
    x = DummySignal()
    operators = [Copy(x, DummySignal()), DummyOp(sets=[x])]
    new_operators = remove_constant_copies(operators)
    assert new_operators == operators

    # check Copy with read input/output does get changed
    x = DummySignal()
    y = DummySignal()
    operators = [Copy(x, y), DummyOp(reads=[x]),
                 DummyOp(reads=[y])]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 3
    assert new_operators[1:] == operators[1:]
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is y
    assert new_operators[0].value == 0

    # check Copy with Reset input does get changed
    x = DummySignal()
    y = DummySignal()
    operators = [Copy(x, y), Reset(x, 2)]
    new_operators = remove_constant_copies(operators)
    assert len(new_operators) == 1
    assert isinstance(new_operators[0], Reset)
    assert new_operators[0].dst is y
    assert new_operators[0].value == 2

    # check that slicing is respected
    x = DummySignal()
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
    x = DummySignal()
    y = DummySignal()
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
        x = DummySignal(shape=(1,) if isinstance(A, float) else A.shape[:1])
        y = DummySignal(shape=(1,) if isinstance(A, float) else A.shape[:1])
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

    # check that reset inputs get removed
    for A in As:
        x = DummySignal()
        y = DummySignal()
        a = DummySignal()
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
    operators = [Op(a, DummySignal(shape=(3,)),
                    DummySignal(shape=(3,)))]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that node inputs don't get removed
    x = DummySignal(label="<Node lorem ipsum.out")
    operators = [Op(x, DummySignal(), DummySignal())]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that identity inputs + trainable don't get removed
    x = Signal(1.0)
    x.trainable = True
    operators = [Op(x, DummySignal(), DummySignal())]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that updated input doesn't get removed
    x = DummySignal()
    operators = [Op(x, DummySignal(), DummySignal()),
                 DummyOp(updates=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that inc'd input doesn't get removed
    x = DummySignal()
    operators = [Op(x, DummySignal(), DummySignal()),
                 DummyOp(incs=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators

    # check that set'd input doesn't get removed
    x = DummySignal()
    operators = [Op(x, DummySignal(), DummySignal()),
                 DummyOp(sets=[x])]
    new_operators = remove_identity_muls(operators)
    assert new_operators == operators
