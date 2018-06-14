__copyright__ = "2015-2018, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from nengo_dl.version import version as __version__

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

# filter out "INFO" level log messages
import tensorflow as tf  # pylint: disable=wrong-import-order,wrong-import-position
tf.logging.set_verbosity(tf.logging.WARN)
del tf  # we don't want a nengo_dl.tf attribute
