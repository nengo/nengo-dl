from nengo.neurons import LIF, LIFRate, Izhikevich, AdaptiveLIF
from nengo.synapses import Lowpass, Triangle, Alpha
from nengo.builder.operator import (SimPyFunc, DotInc, Copy, Reset)
from nengo.builder.neurons import SimNeurons
from nengo.builder.processes import SimProcess
import numpy as np
import pytest

from nengo_deeplearning import builder
from nengo_deeplearning.graph_optimizer import (
    mergeable, greedy_planner, tree_planner, order_signals, create_signals)
from nengo_deeplearning.tensor_node import SimTensorNode


class DummySignal(object):
    def __init__(self, shape=None, dtype=None, base_shape=None, offset=0,
                 trainable=False, label=""):
        self.shape = (1,) if shape is None else shape
        self.dtype = np.float32 if dtype is None else dtype
        self.base = (self if base_shape is None else
                     DummySignal(shape=base_shape, dtype=self.dtype))
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

    # elementwise/dotinc (first dimension must match)
    assert mergeable(DotInc(DummySignal(), DummySignal(), DummySignal()),
                     [DotInc(DummySignal(), DummySignal(), DummySignal())])
    assert mergeable(
        DotInc(DummySignal(shape=(1,)), DummySignal(), DummySignal()),
        [DotInc(DummySignal(shape=()), DummySignal(), DummySignal())])
    assert not mergeable(
        DotInc(DummySignal(shape=(3,)), DummySignal(), DummySignal()),
        [DotInc(DummySignal(shape=(2,)), DummySignal(), DummySignal())])

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


@pytest.mark.parametrize("planner", [greedy_planner, tree_planner])
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


@pytest.mark.parametrize("planner", [greedy_planner, tree_planner])
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


@pytest.mark.parametrize("planner", [greedy_planner])
def test_planner_size(planner):
    # check that operators are selected according to number of available ops
    input0 = DummySignal()
    operators = [Copy(input0, DummySignal(), inc=True)
                 for _ in range(2)]
    operators += [Copy(input0, DummySignal())]
    operators += [DotInc(input0, DummySignal(), DummySignal())
                  for _ in range(3)]
    plan = planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 3
    assert len(plan[1]) == 2
    assert len(plan[2]) == 1


@pytest.mark.parametrize("planner", [greedy_planner, tree_planner])
def test_planner_chain(planner):
    # test a chain
    input0 = DummySignal()
    input1 = DummySignal()
    output0 = DummySignal()
    output1 = DummySignal()
    operators = [Copy(input0, input1, inc=True)]
    operators += [Copy(input1, output0, inc=True) for _ in range(2)]
    operators += [Copy(output0, output1, inc=True) for _ in range(3)]
    plan = planner(operators)
    assert len(plan) == 3
    assert len(plan[0]) == 1
    assert len(plan[1]) == 2
    assert len(plan[2]) == 3


def contiguous(sigs, all_signals):
    indices = sorted([all_signals.index(s) for s in sigs])
    return indices == list(range(np.min(indices), np.max(indices) + 1))


def ordered(ops, all_signals, block=None):
    if block is None:
        read_indices = [
            [all_signals.index(op.reads[i].base) * 10000 +
             op.reads[i].elemoffset for op in ops]
            for i in range(len(ops[0].reads))]
    else:
        read_indices = [
            [all_signals.index(op.reads[block].base) * 10000 +
             op.reads[block].elemoffset for op in ops]]

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


def test_order_signals_partial_complex():
    # more complex partial overlap
    # (A, A/B, B/C, C)
    inputs = [DummySignal(label=str(i)) for i in range(10)]
    plan = [
        tuple(DummyOp(reads=[inputs[i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[2 + i]]) for i in range(5)),
        tuple(DummyOp(reads=[inputs[5 + i]]) for i in range(5)),
    ]
    sigs, new_plan = order_signals(plan)
    assert contiguous(inputs[:5], sigs)
    assert contiguous(inputs[5:], sigs)
    assert contiguous(inputs[2:7], sigs)
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
    assert contiguous(inputs[:4], sigs)
    assert contiguous(inputs[5:8], sigs)
    assert contiguous(inputs[2:6], sigs)
    assert ordered(new_plan[1], sigs)

    # note: technically it is always possible to order both properly, but it
    # requires you to know which of the two equally sized blocks should have
    # priority, and I'm not sure there's a way to determine that.
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
    assert ordered(new_plan[1], sigs, block=0)
    assert not ordered(new_plan[1], sigs, block=1)


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
