import os
import warnings

__copyright__ = "2015, Applied Brain Research"
__license__ = ("Free for non-commercial use; see "
               "https://pythonhosted.org/nengo/license.html")
from .version import version as __version__  # noqa: E402,F401

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# import builtins
# builtins.profile = lambda x: x

# check GPU support
from tensorflow.python.client import device_lib  # noqa: E402

if not any(["gpu" in x.name for x in device_lib.list_local_devices()]):
    warnings.warn("No GPU support detected. It is recommended that you "
                  "install tensorflow-gpu (`pip install tensorflow-gpu`).")

# check nengo version
from nengo.version import version_info as nengo_version  # noqa: E402

minimum_nengo_version = (2, 3, 1)
latest_nengo_version = (2, 4, 1)
if nengo_version < minimum_nengo_version:  # pragma: no cover
    raise ValueError(
        "`nengo_dl` does not support `nengo` version %s. Upgrade "
        "with 'pip install --upgrade --no-deps nengo'." %
        (nengo_version,))
elif nengo_version > latest_nengo_version:  # pragma: no cover
    warnings.warn(
        "This version of `nengo_dl` has not been tested with your `nengo` "
        "version %s. The latest fully supported version is %s" %
        (nengo_version, latest_nengo_version))

# need to explicitly import these to trigger the builder registration
from nengo_dl import (  # noqa: F401
    op_builders, neurons, processes, learning_rules)

# import into top-level namespace
from nengo_dl.simulator import Simulator  # noqa: F401
from nengo_dl.tensor_node import (  # noqa: F401
    TensorNode, tensor_layer, reshaped)
from nengo_dl.utils import configure_settings  # noqa: F401
from nengo_dl.neurons import SoftLIFRate  # noqa: F401

# fix tensorflow bugs
from nengo_dl import tensorflow_patch  # noqa: E402
tensorflow_patch.patch_dynamic_stitch_grad()
tensorflow_patch.patch_state_grads()
