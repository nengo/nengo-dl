import os

import tensorflow as tf

from nengo_deeplearning import Simulator


class TestSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        if os.environ.get("NENGO_DL_TEST_PRECISION", "32") == "32":
            dtype = tf.float32
        else:
            dtype = tf.float64

        if os.environ.get("NENGO_DL_TEST_UNROLL", "False") == "False":
            unroll = False
        else:
            unroll = True

        step_blocks = os.environ.get("NENGO_DL_TEST_STEP_BLOCKS", "50")
        if step_blocks == "None":
            step_blocks = None
        else:
            step_blocks = int(step_blocks)

        super(TestSimulator, self).__init__(
            *args, dtype=dtype, unroll_simulation=unroll,
            step_blocks=step_blocks, **kwargs)
