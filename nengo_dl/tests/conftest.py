from nengo.conftest import seed  # noqa
import pytest

from nengo_dl import tests


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests (use this instead of
    ``nengo_dl.Simulator``).
    """

    return tests.Simulator


def pytest_runtest_setup(item):
    if (getattr(item.obj, 'gpu', None) and not item.config.getvalue('--gpu')):
        pytest.skip("GPU tests not requested")


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False,
                     help="run GPU tests")
