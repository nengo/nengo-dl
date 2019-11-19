import pytest
import tensorflow as tf

from nengo_dl import utils
from nengo_dl.tests import make_test_sim


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "training: mark test as requiring training functionality"
    )
    config.addinivalue_line(
        "markers", "performance: mark tests that depend on a specific benchmark machine"
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu", False) and not utils.tf_gpu_installed:
        pytest.skip("This test requires tensorflow-gpu")
    elif (
        hasattr(item, "fixturenames")
        and "Simulator" not in item.fixturenames
        and item.config.getvalue("--simulator-only")
    ):
        pytest.skip("Only running tests that require a Simulator")
    elif item.get_closest_marker("training", False) and item.config.getvalue(
        "--inference-only"
    ):
        pytest.skip("Skipping training test in inference-only mode")
    elif item.get_closest_marker("performance", False) and not item.config.getvalue(
        "--performance"
    ):
        pytest.skip("Skipping performance test")

    tf.keras.backend.clear_session()


def pytest_addoption(parser):
    parser.addoption(
        "--simulator-only",
        action="store_true",
        default=False,
        help="Only run tests involving Simulator",
    )
    parser.addoption(
        "--inference-only",
        action="store_true",
        default=False,
        help="Run tests in inference-only mode",
    )
    parser.addoption(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Simulator float precision",
    )
    parser.addoption(
        "--unroll-simulation",
        default=1,
        type=int,
        help="`unroll_simulation` parameter for Simulator",
    )
    parser.addoption("--device", default=None, help="`device` parameter for Simulator")
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )


@pytest.fixture(scope="session")
def Simulator(request):
    """
    Simulator class to be used in tests (use this instead of ``nengo_dl.Simulator``).
    """

    return make_test_sim(request)
