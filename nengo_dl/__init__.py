# flake8: noqa

import os
import pkg_resources
import warnings

__copyright__ = "2015-2018, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from .version import version as __version__

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# check GPU support
installed_dists = [d.project_name for d in pkg_resources.working_set]
if ("tensorflow-gpu" not in installed_dists and
        "tf-nightly-gpu" not in installed_dists):
    warnings.warn("No GPU support detected. It is recommended that you "
                  "install tensorflow-gpu (`pip install tensorflow-gpu`).")

# need to explicitly import these to trigger the builder registration
from nengo_dl import (
    op_builders, neuron_builders, process_builders, learning_rule_builders)

# import into top-level namespace
from nengo_dl import dists
from nengo_dl.simulator import Simulator
from nengo_dl.tensor_node import (
    TensorNode, tensor_layer, reshaped)
from nengo_dl.utils import configure_settings
from nengo_dl.neurons import SoftLIFRate

# apply tensorflow monkey patches
from nengo_dl import tensorflow_patch
tensorflow_patch.patch_dynamic_stitch_grad()
tensorflow_patch.patch_state_grads()
