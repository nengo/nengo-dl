# pylint: disable=missing-docstring

from importlib import reload
import sys

from nengo import version as nengo_version
import pytest

import nengo_dl
from nengo_dl import version


def test_nengo_version_check():
    # note: we rely on travis-ci to test this against different nengo versions

    if version.dev or nengo_version.dev is None:
        # nengo_dl should be compatible with all non-development nengo
        # versions, and a nengo_dl dev version should be compatible with all
        # (dev or non-dev) nengo versions
        assert nengo_version.version_info <= version.latest_nengo_version
        with pytest.warns(None) as w:
            reload(version)
        assert not any("This version of `nengo_dl` has not been tested with "
                       "your `nengo` version" in str(x.message) for x in w)
    else:
        # a development version of nengo with a non-development nengo_dl
        # version should cause a warning (we don't want to mark a nengo_dl
        # release as compatible with a nengo dev version, since the nengo
        # version may change and no longer be compatible with our nengo_dl
        # release).

        # note: we assume that a nengo dev version means the latest dev
        # version
        assert nengo_version.version_info >= version.latest_nengo_version

        with pytest.warns(UserWarning) as w:
            reload(version)

        assert any("This version of `nengo_dl` has not been tested with "
                   "your `nengo` version" in str(x.message) for x in w)

    # check that it still works without nengo (faking an import error by
    # messing up sys.modules)
    saved = sys.modules["nengo.version"]
    sys.modules["nengo.version"] = None
    with pytest.warns(None) as w:
        reload(version)
    assert len(w) == 0
    sys.modules["nengo.version"] = saved


@pytest.mark.gpu
def test_tensorflow_gpu_warning():
    # we assume that if the --gpu flag is set then tensorflow-gpu is installed,
    # so there should be no warning
    with pytest.warns(None) as w:
        reload(nengo_dl)

    assert len(w) == 0
