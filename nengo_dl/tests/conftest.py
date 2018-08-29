from nengo.conftest import seed  # pylint: disable=unused-import
import pytest

from nengo_dl import tests


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests (use this instead of
    ``nengo_dl.Simulator``).
    """

    return tests.Simulator


def pytest_runtest_setup(item):
    if getattr(item.obj, "gpu", False) and not item.config.getvalue("--gpu"):
        pytest.skip("GPU tests not requested")
    elif ("Simulator" not in item.fixturenames and
          item.config.getvalue("--simulator-only")):
        pytest.skip("Only running tests that require a Simulator")
    elif getattr(item.obj, "training", False) and item.config.getvalue(
            "--inference-only"):
        pytest.skip("Skipping training test in inference-only mode")


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False,
                     help="Run GPU tests")
    parser.addoption("--simulator-only", action="store_true", default=False,
                     help="Only run tests involving Simulator")
    parser.addoption("--inference-only", action="store_true", default=False,
                     help="Don't run tests that require training/gradients")
