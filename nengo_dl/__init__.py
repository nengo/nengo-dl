# pylint: disable=wrong-import-order,wrong-import-position,missing-docstring,ungrouped-imports
# isort: skip_file

__license__ = "Free for non-commercial use; see LICENSE.rst"
from nengo_dl.version import version as __version__
from nengo_dl.version import copyright as __copyright__

# filter out "INFO" level log messages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

del os

# need to explicitly import these to trigger the builder registration
from nengo_dl import (
    op_builders,
    neuron_builders,
    process_builders,
    learning_rule_builders,
    transform_builders,
)

# import into top-level namespace
from nengo_dl import callbacks, compat, converter, dists, losses
from nengo_dl.config import configure_settings
from nengo_dl.converter import Converter
from nengo_dl.neurons import (
    LeakyReLU,
    SoftLIFRate,
    SpikingLeakyReLU,
)
from nengo_dl.simulator import Simulator
from nengo_dl.tensor_node import Layer, TensorNode, tensor_layer
