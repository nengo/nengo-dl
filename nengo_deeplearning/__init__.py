import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEBUG = False

from nengo_deeplearning.builder import Builder
from nengo_deeplearning.simulator import Simulator

# need to explicitly import these to trigger the builder registration
from nengo_deeplearning import operators, neurons, processes, learning_rules
