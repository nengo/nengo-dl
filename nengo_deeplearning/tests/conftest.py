import nengo.tests.test_synapses
import numpy as np
# import tensorflow as tf

# from nengo_deeplearning.tests import TestSimulator


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


# # add options to control simulator arguments
# def pytest_addoption(parser):
#     parser.addoption("--precision", action="store", default="32",
#                      help="floating point precision for simulator")
#     parser.addoption("--unroll", action="store", default="False",
#                      help="unroll_simulation Simulator argument")
#     parser.addoption("--step_blocks", action="store", default="None",
#                      help="step_blocks Simulator argument")
#
#
# def pytest_configure(config):
#     if config.getoption("precision"):
#         if config.getoption("precision") == "32":
#             TestSimulator.dtype = tf.float32
#         else:
#             TestSimulator.dtype = tf.float64
#
#     if config.getoption("unroll"):
#         TestSimulator.unroll = bool(config.getoption("unroll"))
#
#     if config.getoption("step_blocks"):
#         tmp = config.getoption("step_blocks")
#         if tmp == "None":
#             TestSimulator.step_blocks = None
#         else:
#             TestSimulator.step_blocks = int(tmp)
