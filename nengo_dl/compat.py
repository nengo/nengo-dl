# pylint: disable=unused-import,ungrouped-imports

"""Utilities to ease cross-compatibility between different versions of upstream
dependencies."""

import collections
import inspect

import nengo
import tensorflow as tf
from nengo._vendor.scipy.sparse import linalg_interface, linalg_onenormest
from packaging import version


class NoType:
    """A type that can never be instantiated."""

    def __init__(self, *arg, **kwargs):  # pragma: no cover
        raise RuntimeError("Cannot instantiate")


def make_dummy_type(name):
    """Return a NoType subclass with the given name."""
    return type(name, (NoType,), {})


# TensorFlow compatibility


class TFLogFilter:
    """
    Filter for cleaning up the TensorFlow log output.

    Parameters
    ----------
    err_on_deprecation : bool
        If True, treat deprecation log messages as errors.
    """

    def __init__(self, err_on_deprecation):
        self.err_on_deprecation = err_on_deprecation

    def filter(self, record):
        """
        Apply filter to log record.

        Parameters
        ----------
        record : `LogRecord`
            Message emitted by logger.

        Returns
        -------
        pass : bool
            If True, pass the ``record``, otherwise filter it out.

        Raises
        ------
        AttributeError
            If a deprecation message is detected and ``err_on_deprecation=True``.
        """

        if self.err_on_deprecation and (
            "deprecation.py" in record.pathname or "deprecated" in record.msg.lower()
        ):
            msg = record.getMessage()
            raise AttributeError(f"Deprecation warning detected:\n{msg}")

        return True


tf.get_logger().addFilter(TFLogFilter(err_on_deprecation=False))

if version.parse(tf.__version__) < version.parse("2.6.0rc0"):  # pragma: no cover
    from tensorflow.python.keras.engine.functional import Functional, _build_map
    from tensorflow.python.keras.layers import (
        BatchNormalizationV1,
        BatchNormalizationV2,
    )
else:
    from keras.engine.functional import Functional, _build_map
    from keras.layers import BatchNormalizationV1, BatchNormalizationV2

if version.parse(tf.__version__) < version.parse("2.5.0rc0"):

    def sub_layers(layer):
        """Get layers contained in ``layer``."""
        return layer._layers

else:

    def sub_layers(layer):
        """Get layers contained in ``layer``."""
        return layer._self_tracked_trackables

    # monkeypatch to fix bug when using TF2.5 with sphinx's doctest extension
    from tensorflow.python.autograph.impl.api import StackTraceMapper

    old_source_map = StackTraceMapper.get_effective_source_map

    def get_effective_source_map(self):
        """
        Sometimes the source file is unknown (e.g. when running code through Sphinx's
        doctest builder).

        This causes TensorFlow to crash (as of TF 2.5). So we convert any Nones
        to the string "unknown".
        """

        effective_source_map = old_source_map(self)

        # convert Nones to "unknown"
        effective_source_map = {
            key: tuple("unknown" if x is None else x for x in val)
            for key, val in effective_source_map.items()
        }
        return effective_source_map

    StackTraceMapper.get_effective_source_map = get_effective_source_map

if version.parse(tf.__version__) < version.parse("2.10.0rc0"):
    from tensorflow.python.training.tracking import base as trackable
else:
    from tensorflow.python.trackable import base as trackable


# Nengo compatibility

HAS_NENGO_3_2_1 = version.parse(nengo.__version__) >= version.parse("3.2.1.dev0")
HAS_NENGO_3_2_0 = version.parse(nengo.__version__) >= version.parse("3.2.0.dev0")
HAS_NENGO_3_1_0 = version.parse(nengo.__version__) >= version.parse("3.1.0.dev0")

if HAS_NENGO_3_2_0:  # pragma: no cover
    from nengo.builder.transforms import ConvTransposeInc
    from nengo.transforms import ConvolutionTranspose
    from nengo.utils.stdlib import FrozenOrderedSet
else:
    ConvTransposeInc = make_dummy_type("ConvTransposeInc")
    ConvolutionTranspose = make_dummy_type("ConvolutionTranspose")

    class FrozenOrderedSet(collections.abc.Set):
        """Backport of `nengo.utils.stdlib.FrozenOrderedSet`."""

        def __init__(self, data):
            self.data = dict((d, None) for d in data)

        def __contains__(self, elem):
            return elem in self.data

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __hash__(self):
            return self._hash()


if HAS_NENGO_3_1_0:
    from nengo.builder.probe import SimProbe
    from nengo.neurons import PoissonSpiking, RegularSpiking, StochasticSpiking, Tanh
    from nengo.transforms import NoTransform

    default_transform = None

    def conn_has_weights(conn):
        """Equivalent to conn.has_weights."""
        return conn.has_weights

    def neuron_state(neuron_op):
        """Equivalent to neuron_op.state."""
        return neuron_op.state

    def neuron_step(neuron_op, dt, J, output, state):  # pragma: no cover (runs in TF)
        """Equivalent to neuron_op.step."""
        neuron_op.neurons.step(dt, J, output, **state)

    def to_neurons(conn):
        """Equivalent to conn._to_neurons."""
        return conn._to_neurons

else:
    PoissonSpiking = make_dummy_type("PoissonSpiking")
    RegularSpiking = make_dummy_type("RegularSpiking")
    StochasticSpiking = make_dummy_type("StochasticSpiking")
    Tanh = make_dummy_type("Tanh")
    NoTransform = make_dummy_type("NoTransform")

    default_transform = 1

    def conn_has_weights(conn):
        """All connections have weights."""
        return True

    def neuron_state(neuron_op):
        """Look up keys from function signature."""
        names = list(inspect.signature(neuron_op.neurons.step_math).parameters.keys())[
            3:
        ]
        assert len(names) == len(neuron_op.states)
        return dict(zip(names, neuron_op.states))

    def neuron_step(neuron_op, dt, J, output, state):  # pragma: no cover (runs in TF)
        """Call step_math instead of step."""
        neuron_op.neurons.step_math(dt, J, output, *state.values())

    def to_neurons(conn):
        """Check whether the output of a connection is a neuron object."""
        return isinstance(conn.post_obj, nengo.ensemble.Neurons) or (
            isinstance(conn.pre_obj, nengo.Ensemble)
            and isinstance(conn.post_obj, nengo.Ensemble)
            and conn.solver.weights
        )

    class SimProbe(nengo.builder.Operator):
        """Backport of Nengo 3.1.0 SimProbe."""

        def __init__(self, signal, tag=None):
            super().__init__(tag=tag)
            self.sets = []
            self.incs = []
            self.reads = [signal]
            self.updates = []

        def make_step(self, signals, dt, rng):
            """Not used in NengoDL."""
            raise NotImplementedError

    from nengo_dl.builder import NengoBuilder as _NengoBuilder

    def build_probe(model, probe):
        """Copy of the base probe builder that adds the SimProbe op."""
        nengo.builder.probe.build_probe(model, probe)
        model.add_op(SimProbe(model.sig[probe]["in"]))

    _NengoBuilder.register(nengo.Probe)(build_probe)

    # monkeypatch fix for https://github.com/nengo/nengo/pull/1587
    linalg_onenormest.aslinearoperator = linalg_interface.aslinearoperator


try:
    from keras_spiking import Alpha, Lowpass, SpikingActivation
except ImportError:
    SpikingActivation = make_dummy_type("keras_spiking_SpikingActivation")
    Lowpass = make_dummy_type("keras_spiking_Lowpass")
    Alpha = make_dummy_type("keras_spiking_Alpha")
