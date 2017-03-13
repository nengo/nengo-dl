import os
import warnings

__copyright__ = "2015, Applied Brain Research"
__license__ = ("Free for non-commercial use; see "
               "https://pythonhosted.org/nengo/license.html")
from .version import version as __version__  # noqa: E402,F401

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEBUG = False
# TODO: change the debug system to use a proper logger

# check GPU support
from tensorflow.python.client import device_lib  # noqa: E402

if not any(["gpu" in x.name for x in device_lib.list_local_devices()]):
    default_device = "/cpu:0"
    warnings.warn("No GPU support detected. It is recommended that you "
                  "install tensorflow-gpu (`pip install tensorflow-gpu`).")
else:
    default_device = "/gpu:0"

# check nengo version
from nengo.version import version_info  # noqa: E402

minimum_nengo_version = (2, 3, 1)
latest_nengo_version = (2, 3, 2)
if version_info < minimum_nengo_version:
    raise ValueError(
        "`nengo_dl` does not support `nengo` version %s. Upgrade "
        "with 'pip install --upgrade --no-deps nengo'."
        % version_info)
elif version_info > latest_nengo_version:
    warnings.warn("This version of `nengo_dl` has not been tested "
                  "with your `nengo` version (%s). The latest fully "
                  "supported version is %s" % (
                      version_info, latest_nengo_version))

# need to explicitly import these to trigger the builder registration
from nengo_dl import (  # noqa: F401
    operators, neurons, processes, learning_rules)

# import into top-level namespace
from nengo_dl.simulator import Simulator  # noqa: F401
from nengo_dl.tensor_node import TensorNode  # noqa: F401

# fix tensorflow bugs
from nengo_dl import tensorflow_patch  # noqa: E402
tensorflow_patch.patch_dynamic_stitch_grad()
