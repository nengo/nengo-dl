"""
We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""

import warnings

name = "nengo-dl"
version_info = (3, 1, 0)  # (major, minor, patch)
dev = None  # set to None for releases

version = "{v}{dev}".format(
    v=".".join(str(v) for v in version_info),
    dev=(".dev%d" % dev) if dev is not None else "",
)

# check nengo version
try:
    import nengo.version
except ImportError:
    # nengo not installed, can't check version
    pass
else:
    minimum_nengo_version = (3, 0, 0)

    # for release versions of nengo-dl, this should be the latest released
    # nengo version. for dev versions of nengo-dl, this should be the current
    # nengo dev version.
    latest_nengo_version = (3, 0, 0)

    if nengo.version.version_info < minimum_nengo_version:  # pragma: no cover
        raise ValueError(
            "`nengo_dl` does not support `nengo` version %s. Upgrade "
            "with 'pip install --upgrade --no-deps nengo'." % (nengo.version.version,)
        )
    elif nengo.version.version_info > latest_nengo_version:  # pragma: no cover
        warnings.warn(
            "This version of `nengo_dl` has not been tested with your `nengo` "
            "version (%s). The latest fully supported version is %d.%d.%d."
            % ((nengo.version.version,) + latest_nengo_version)
        )
