import os
import warnings

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

# need to explicitly import these to trigger the builder registration
from nengo_deeplearning import (  # noqa: F401
    operators, neurons, processes, learning_rules)

from nengo_deeplearning.simulator import Simulator  # noqa: F401

# check nengo version
from nengo.version import version_info  # noqa: E402

minimum_nengo_version = (2, 3, 0)
latest_nengo_version = (2, 3, 1)
if version_info < minimum_nengo_version:
    raise ValueError(
        "`nengo_deeplearning` does not support `nengo` version %s. Upgrade "
        "with 'pip install --upgrade --no-deps nengo'."
        % version_info)
elif version_info > latest_nengo_version:
    warnings.warn("This version of `nengo_deeplearning` has not been tested "
                  "with your `nengo` version (%s). The latest fully "
                  "supported version is %s" % (
                      version_info, latest_nengo_version))
