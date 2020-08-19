# pylint: disable=missing-docstring

import tensorflow as tf

from nengo_dl import config, simulator


def make_test_sim(request):
    """
    A Simulator factory to be used in tests.

    The factory allows simulator arguments to be controlled via pytest command line
    arguments.

    This is used in the ``conftest.Simulator`` fixture, or can be be passed
    to the ``nengo_simloader`` option when running the Nengo core tests.
    """

    dtype = request.config.getoption("--dtype")
    unroll = request.config.getoption("--unroll-simulation")
    device = request.config.getoption("--device")
    inference_only = request.config.getoption("--inference-only")

    def TestSimulator(net, *args, **kwargs):
        kwargs.setdefault("unroll_simulation", unroll)
        kwargs.setdefault("device", device)
        kwargs.setdefault("progress_bar", False)

        if net is not None and config.get_setting(net, "inference_only") is None:
            with net:
                config.configure_settings(inference_only=inference_only)

        if net is not None and config.get_setting(net, "dtype") is None:
            with net:
                config.configure_settings(dtype=dtype)

        return simulator.Simulator(net, *args, **kwargs)

    return TestSimulator
