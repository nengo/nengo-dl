# pylint: disable=wrong-import-order,wrong-import-position,missing-docstring,ungrouped-imports

__copyright__ = "2015-2020, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from nengo_dl.version import version as __version__

# check python version
import sys

if sys.version_info < (3, 5):
    raise ImportError(
        """
You are running Python version %s with NengoDL version %s. NengoDL requires
at least Python 3.5.

The fact that this version was installed on your system probably means that you
are using an older version of pip; you should consider upgrading with

 $ pip install pip setuptools --upgrade

There are two options for getting NengoDL working:

- Upgrade to Python >= 3.5

- Install an older version of NengoDL:

 $ pip install 'nengo-dl<2.0'
"""
        % (sys.version, __version__)
    )
del sys

# filter out "INFO" level log messages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

del os

import tensorflow as tf

# disable control flow v2 for performance reasons
# (see https://github.com/tensorflow/tensorflow/issues/33052)
tf.compat.v1.disable_control_flow_v2()

del tf

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
from nengo_dl.neurons import SoftLIFRate
from nengo_dl.simulator import Simulator
from nengo_dl.tensor_node import TensorNode, Layer, tensor_layer
