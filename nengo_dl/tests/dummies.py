"""Dummy objects for use in tests."""

from collections import defaultdict

import nengo
import numpy as np

from nengo_dl import builder, tensor_graph, signals


class Signal(object):
    def __init__(self, shape=None, dtype=None, base_shape=None, offset=0,
                 trainable=False, label="", initial_value=0):
        self.shape = (1,) if shape is None else shape
        self.dtype = np.float32 if dtype is None else dtype
        self.base = (self if base_shape is None else
                     Signal(shape=base_shape, dtype=self.dtype,
                            label="%s.base" % label))
        self.elemoffset = offset
        self.name = label
        self.ndim = len(self.shape)
        self.is_view = base_shape is not None
        self.size = np.prod(self.shape)
        self.trainable = trainable
        self.minibatched = not trainable
        self.init = initial_value

    @property
    def initial_value(self):
        return np.full(self.base.shape, self.init, dtype=self.dtype)

    def may_share_memory(self, _):
        return False

    def __repr__(self):
        return "DummySignal(%s)" % self.name


class Op(object):
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


@builder.Builder.register(Op)
class Builder(builder.OpBuilder):
    pass


class Probe(nengo.Probe):
    # pylint: disable=super-init-not-called
    def __init__(self, target=None):
        if target is not None:
            # bypass target validation
            nengo.Probe.target.data[self] = target


class Simulator(object):
    model = nengo.builder.Model()
    model.sig = defaultdict(lambda: defaultdict(lambda: Signal()))


class TensorGraph(tensor_graph.TensorGraph):
    # pylint: disable=super-init-not-called
    def __init__(self, plan=None, dtype=None, minibatch_size=None):
        self.plan = plan
        self.dtype = dtype
        self.minibatch_size = minibatch_size

        self.signals = signals.SignalDict(self.dtype, self.minibatch_size)
