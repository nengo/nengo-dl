import os

import tensorflow as tf

from nengo_deeplearning import Simulator


# TODO: set this up so step_blocks, unroll_simulation can be controlled by
# environment variables in travis ci

class TestSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        if os.environ.get("NENGO_DL_TEST_PRECISION", "64") == "32":
            dtype = tf.float32
        else:
            dtype = tf.float64

        if os.environ.get("NENGO_DL_TEST_UNROLL", "False") == "False":
            unroll = False
        else:
            unroll = True

        step_blocks = os.environ.get("NENGO_DL_TEST_STEP_BLOCKS", "None")
        if step_blocks == "None":
            step_blocks = None
        else:
            step_blocks = int(step_blocks)

        super(TestSimulator, self).__init__(
            *args, dtype=dtype, unroll_simulation=unroll,
            step_blocks=step_blocks, **kwargs)
