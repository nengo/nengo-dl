import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEBUG = False

from nengo_deeplearning.builder import Builder  # noqa: F401
from nengo_deeplearning.simulator import Simulator  # noqa: F401

# need to explicitly import these to trigger the builder registration
from nengo_deeplearning import (  # noqa: F401
    operators, neurons, processes, learning_rules)
