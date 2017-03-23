import logging
import os

import nengo.tests.test_synapses
import numpy as np
import tensorflow as tf

from nengo_dl import simulator


# set looser tolerances on synapse tests
def allclose(*args, **kwargs):
    kwargs.setdefault('atol', 5e-7)
    return nengo.utils.testing.allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose

# cast output of run_synapse to float64. this is necessary because
# Synapse.filt bases its internal dtypes on the dtype of its inputs, and
# we don't want to downcast everything there to float32.
nengo_run_synapse = nengo.tests.test_synapses.run_synapse


def run_synapse(*args, **kwargs):
    output = nengo_run_synapse(*args, **kwargs)
    return tuple(x.astype(np.float64) for x in output)


nengo.tests.test_synapses.run_synapse = run_synapse


class Simulator(simulator.Simulator):
    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.WARNING)

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
        else:
            kwargs.setdefault("step_blocks", 10)

        if "NENGO_DL_TEST_DEVICE" in os.environ:
            kwargs.setdefault("device", os.environ["NENGO_DL_TEST_DEVICE"])
        else:
            kwargs.setdefault("device", "/cpu:0")

        super(Simulator, self).__init__(*args, **kwargs)
