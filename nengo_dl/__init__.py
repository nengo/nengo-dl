# pylint: disable=wrong-import-order,wrong-import-position,missing-docstring,ungrouped-imports

__copyright__ = "2015-2018, Applied Brain Research"
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

from nengo_dl.compat import tf_compat

tf_compat.logging.set_verbosity(tf_compat.logging.WARN)


# disable v2 behaviour (for now, pending full TF 2.0 compatibility)
tf_compat.disable_control_flow_v2()


del os
del tf_compat

# need to explicitly import these to trigger the builder registration
from nengo_dl import (
    op_builders,
    neuron_builders,
    process_builders,
    learning_rule_builders,
    transform_builders,
)

# import into top-level namespace
from nengo_dl import dists, callbacks
from nengo_dl.simulator import Simulator
from nengo_dl.tensor_node import TensorNode, tensor_layer, reshaped
from nengo_dl.config import configure_settings
from nengo_dl.neurons import SoftLIFRate

# apply tensorflow monkey patches
from nengo_dl import tensorflow_patch

tensorflow_patch.patch_dynamic_stitch_grad()
