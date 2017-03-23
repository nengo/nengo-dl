from nengo.conftest import seed  # noqa
import pytest

from nengo_dl import tests


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests (use this instead of
    ``nengo_dl.Simulator``).
    """

    return tests.Simulator
