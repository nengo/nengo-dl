# pylint: disable=unused-import,ungrouped-imports

"""
Utilities to ease cross-compatibility between different versions of upstream
dependencies.
"""

import collections
import inspect

import nengo
import tensorflow as tf
from nengo._vendor.scipy.sparse import linalg_interface, linalg_onenormest
from packaging import version
from tensorflow.python.eager import context

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

        # "tf.keras.backend.get_session is deprecated": this deprecation message
        # is raised incorrectly due to a bug, see
        # https://github.com/tensorflow/tensorflow/issues/33182
        if len(record.args) > 1 and record.args[1] == "tf.keras.backend.get_session":
            return False

        # "Output steps_run missing from loss dictionary": steps_run should
        # never have a loss defined
        if record.msg.startswith("Output steps_run missing from loss dictionary"):
            return False

        if self.err_on_deprecation and (
            "deprecation.py" in record.pathname or "deprecated" in record.msg.lower()
        ):
            msg = record.getMessage()
            raise AttributeError(f"Deprecation warning detected:\n{msg}")

        return True


tf.get_logger().addFilter(TFLogFilter(err_on_deprecation=False))

if version.parse(tf.__version__) < version.parse("2.3.0rc0"):

    def _build_map(outputs):
        """
        Vendored from ``tensorflow.python.keras.engine.functional._build_map``
        in TF 2.3.0 (with a few changes noted below).
        """
        finished_nodes = set()
        nodes_in_progress = set()
        nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
        layer_indices = {}  # layer -> in traversal order.
        for output in tf.nest.flatten(outputs):
            _build_map_helper(
                output,
                finished_nodes,
                nodes_in_progress,
                nodes_in_decreasing_depth,
                layer_indices,
            )

        # CHANGED: add `node.layer` alias for `node.outbound_layer`
        for node in nodes_in_decreasing_depth:
            node.layer = node.outbound_layer

        return nodes_in_decreasing_depth, layer_indices

    def _build_map_helper(
        tensor,
        finished_nodes,
        nodes_in_progress,
        nodes_in_decreasing_depth,
        layer_indices,
    ):
        """Recursive helper for `_build_map`."""
        layer, node_index, _ = tensor._keras_history  # pylint: disable=protected-access
        node = layer._inbound_nodes[node_index]  # pylint: disable=protected-access

        # Don't repeat work for shared subgraphs
        if node in finished_nodes:
            return

        # Prevent cycles.
        if node in nodes_in_progress:  # pragma: no cover
            raise ValueError(
                "The tensor "
                + str(tensor)
                + ' at layer "'
                + layer.name
                + '" is part of a cycle.'
            )

        # Store the traversal order for layer sorting.
        if layer not in layer_indices:
            layer_indices[layer] = len(layer_indices)

        # Propagate to all previous tensors connected to this node.
        nodes_in_progress.add(node)
        # CHANGED: `not node.is_input` to `node.inbound_layers`
        if node.inbound_layers:
            # CHANGED: `node.keras_inputs` to `tf.nest.flatten(node.input_tensors)`
            # CHANGED: `tensor` to `input_tensor` (for pylint)
            for input_tensor in tf.nest.flatten(node.input_tensors):
                _build_map_helper(
                    input_tensor,
                    finished_nodes,
                    nodes_in_progress,
                    nodes_in_decreasing_depth,
                    layer_indices,
                )

        finished_nodes.add(node)
        nodes_in_progress.remove(node)
        nodes_in_decreasing_depth.append(node)

    from tensorflow.keras import Model as Functional
    from tensorflow.python.keras.layers import (
        BatchNormalizationV1,
        BatchNormalizationV2,
    )
elif version.parse(tf.__version__) < version.parse("2.6.0rc0"):  # pragma: no cover
    from tensorflow.python.keras.engine.functional import Functional, _build_map
    from tensorflow.python.keras.layers import (
        BatchNormalizationV1,
        BatchNormalizationV2,
    )
else:
    from keras.engine.functional import Functional, _build_map
    from keras.layers import BatchNormalizationV1, BatchNormalizationV2


if version.parse(tf.__version__) < version.parse("2.3.0rc0"):
    from tensorflow.python.keras.engine import network

    # monkeypatch to fix bug in TF2.2, see
    # https://github.com/tensorflow/tensorflow/issues/37548
    old_conform = network.Network._conform_to_reference_input

    def _conform_to_reference_input(self, tensor, ref_input):
        keras_history = getattr(tensor, "_keras_history", None)

        tensor = old_conform(self, tensor, ref_input)

        if keras_history is not None:
            tensor._keras_history = keras_history

        return tensor

    network.Network._conform_to_reference_input = _conform_to_reference_input

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
        doctest builder). This causes TensorFlow to crash (as of TF 2.5). So we convert
        any Nones to the string "unknown".
        """

        effective_source_map = old_source_map(self)

        # convert Nones to "unknown"
        effective_source_map = {
            key: tuple("unknown" if x is None else x for x in val)
            for key, val in effective_source_map.items()
        }
        return effective_source_map

    StackTraceMapper.get_effective_source_map = get_effective_source_map


# Nengo compatibility

if version.parse(nengo.__version__) < version.parse("3.1.0"):
    PoissonSpiking = RegularSpiking = StochasticSpiking = Tanh = NoTransform = type(
        None
    )

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

else:
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


if version.parse(nengo.__version__) <= version.parse("3.1.0"):

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


else:
    from nengo.utils.stdlib import FrozenOrderedSet


def eager_enabled():
    """
    Check if we're in eager mode or graph mode.

    Note: this function differs from ``tf.executing_eagerly()`` in that it will still
    return ``True`` if we're inside a ``tf.function``. Essentially this checks whether
    the user has called ``tf.compat.v1.disable_eager_execution()`` or not.
    """

    return context.default_execution_mode == context.EAGER_MODE


try:
    from keras_spiking import Alpha, Lowpass, SpikingActivation
except ImportError:
    SpikingActivation = object()
    Lowpass = object()
    Alpha = object()
