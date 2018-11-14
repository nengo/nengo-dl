# pylint: disable=missing-docstring

import nengo
import tensorflow as tf

from nengo_dl import config


def test_simulator_fixture(Simulator, pytestconfig):
    with Simulator(nengo.Network()) as sim:
        assert sim.tensor_graph.dtype == (
            tf.float32 if pytestconfig.getoption("--dtype") == "float32" else
            tf.float64)
        assert sim.unroll == pytestconfig.getoption("--unroll_simulation")
        assert sim.tensor_graph.device == pytestconfig.getoption("--device")
        assert (config.get_setting(sim.model, "inference_only") ==
                pytestconfig.getoption("--inference-only"))

    # check that manually specified values aren't overridden
    with nengo.Network() as net:
        config.configure_settings(dtype=tf.float64, inference_only=True)

    with Simulator(net, unroll_simulation=5, device="/cpu:0") as sim:
        assert sim.tensor_graph.dtype == tf.float64
        assert sim.unroll == 5
        assert sim.tensor_graph.device == "/cpu:0"
        assert config.get_setting(sim.model, "inference_only")
