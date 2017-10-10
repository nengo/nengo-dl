import sys

from nengo import version as nengo_version
import pytest

from nengo_dl import version

if sys.version_info >= (3, 4):
    from importlib import reload


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
