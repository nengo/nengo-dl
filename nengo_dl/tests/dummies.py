"""Dummy objects for use in tests."""

from collections import defaultdict

import nengo
import numpy as np
from packaging import version

from nengo_dl import builder, signals, tensor_graph


class Signal(nengo.builder.signal.Signal):
    """
    Mock-up for `nengo.builder.Signal`.
    """

    def __init__(
        self,
        shape=None,
        dtype=None,
        base_shape=None,
        offset=0,
        trainable=False,
        label="",
        initial_value=0,
        sparse=False,
    ):  # pylint: disable=super-init-not-called
        self._shape = (1,) if shape is None else shape
        self._dtype = np.float32 if dtype is None else dtype
        self._base = (
            self
            if base_shape is None
            else Signal(shape=base_shape, dtype=self.dtype, label=f"{label}.base")
        )
        self._elemoffset = offset
        self.name = label
        self._ndim = len(self.shape)
        self._is_view = base_shape is not None
        self._size = np.prod(self.shape)
        self.trainable = trainable
        self.minibatched = not trainable
        self.init = initial_value
        self._sparse = sparse

    @property
    def initial_value(self):
        """Initial signal value."""
        return np.full(self.base.shape, self.init, dtype=self.dtype)

    @property
    def shape(self):
        """Signal shape."""
        return self._shape

    @property
    def dtype(self):
        """Signal dtype."""
        return self._dtype

    @property
    def elemoffset(self):
        """Signal offset."""
        return self._elemoffset

    @property
    def ndim(self):
        """Number of dimensions."""
        return self._ndim

    @property
    def is_view(self):
        """Whether or not this signal is a view of a base signal."""
        return self._is_view

    @property
    def size(self):
        """Size of signal."""
        return self._size

    @property
    def sparse(self):
        """Whether or not signal is sparse."""
        return self._sparse

    def may_share_memory(self, _):
        """Whether or not this signal shares memory with another."""
        return False

    def __repr__(self):
        return f"DummySignal({self.name})"


class Op:
    """
    Mock-up for `nengo.builder.Operator`.
    """

    def __init__(self, sets=None, incs=None, reads=None, updates=None):
        self.sets = [] if sets is None else sets
        self.incs = [] if incs is None else incs
        self.reads = [] if reads is None else reads
        self.updates = [] if updates is None else updates

        self.all_signals = self.sets + self.incs + self.reads + self.updates

    def __repr__(self):
        rep = "DummyOp("
        if len(self.sets) > 0:
            rep += f"sets={self.sets}"
        if len(self.incs) > 0:
            rep += f"incs={self.incs}"
        if len(self.reads) > 0:
            rep += f"reads={self.reads}"
        if len(self.updates) > 0:
            rep += f"updates={self.updates}"
        rep += ")"
        return rep

    def make_step(self, *args):
        """Raises an error (since this shouldn't be called)."""
        raise NotImplementedError()

    def init_signals(self, *args):
        """Raises an error (since this shouldn't be called)."""
        raise NotImplementedError()


@builder.Builder.register(Op)
class Builder(builder.OpBuilder):
    """
    Mock-up builder for `.Op`.
    """

    @staticmethod
    def mergeable(x, y):
        return True


class Probe(nengo.Probe):
    """
    Mock-up for `nengo.Probe`.
    """

    def __init__(self, target=None):
        # pylint: disable=super-init-not-called
        if target is not None:
            # bypass target validation
            nengo.Probe.target.data[self] = target

    @property
    def size_in(self):
        """Modified version of size_in that supports Signal targets."""

        return (
            self.target.size
            if isinstance(self.target, (Signal, nengo.builder.signal.Signal))
            else self.target.size_out
        )


class Simulator:
    """
    Mock-up for `nengo.Simulator`.
    """

    model = nengo.builder.Model()
    model.sig = defaultdict(lambda: defaultdict(Signal))


class TensorGraph(tensor_graph.TensorGraph):
    """
    Mock-up for `.tensor_graph.TensorGraph`.
    """

    def __init__(self, plan=None, dtype=None, minibatch_size=None):
        # pylint: disable=bad-super-call
        super(tensor_graph.TensorGraph, self).__init__(dtype=dtype)

        self.plan = plan
        self.minibatch_size = minibatch_size
        self.seed = 0
        self.model = nengo.builder.Model()
        self.invariant_inputs = {}

        self.signals = signals.SignalDict(self.dtype, self.minibatch_size)


def linear_net():
    """
    A simple network with an input, output, and no nonlinearity.
    """

    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=None, transform=1)
        p = nengo.Probe(b)

    return net, a, p


def DeterministicLIF(**kwargs):
    """
    A LIF neuron with fixed initial voltages (useful in tests where we want neurons
    to always have the same output given the same parameters, but don't want to
    fix the seed).
    """

    if version.parse(nengo.__version__) <= version.parse("3.0.0"):
        return nengo.LIF(**kwargs)
    else:
        return nengo.LIF(initial_state={"voltage": nengo.dists.Choice([0])}, **kwargs)
