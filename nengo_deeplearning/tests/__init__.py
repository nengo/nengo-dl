import os

import tensorflow as tf

from nengo_deeplearning import Simulator


class TestSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        if "NENGO_DL_TEST_PRECISION" in os.environ:
            if os.environ["NENGO_DL_TEST_PRECISION"] == "32":
                kwargs.setdefault("dtype", tf.float32)
            else:
                kwargs.setdefault("dtype", tf.float64)

        if "NENGO_DL_TEST_UNROLL" in os.environ:
            if os.environ["NENGO_DL_TEST_UNROLL"] == "True":
                kwargs.setdefault("unroll_simulation", True)
            else:
                kwargs.setdefault("unroll_simulation", False)

        if "NENGO_DL_TEST_STEP_BLOCKS" in os.environ:
            step_blocks = os.environ["NENGO_DL_TEST_STEP_BLOCKS"]
            if step_blocks == "None":
                kwargs.setdefault("step_blocks", None)
            else:
                kwargs.setdefault("step_blocks", int(step_blocks))

        kwargs.setdefault("device", "/cpu:0")
        if "NENGO_DL_TEST_DEVICE" in os.environ:
            kwargs.setdefault("device", os.environ["NENGO_DL_TEST_DEVICE"])

        super(TestSimulator, self).__init__(*args, **kwargs)
