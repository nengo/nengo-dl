"""
We use semantic versioning (see http://semver.org/).
Additionally, '.dev0' will be added to the version unless the code base
represents a release version. Commits for which the version doesn't have
'.dev0' should be git tagged with the version.
"""

import warnings

name = "nengo-dl"
version_info = (2, 1, 0)  # (major, minor, patch)
dev = False

version = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev='.dev0' if dev else '')

# check nengo version
try:
    import nengo.version

    minimum_nengo_version = (2, 7, 0)
    latest_nengo_version = (2, 8, 0)
    if nengo.version.version_info < minimum_nengo_version:  # pragma: no cover
        raise ValueError(
            "`nengo_dl` does not support `nengo` version %s. Upgrade "
            "with 'pip install --upgrade --no-deps nengo'." %
            (nengo.version.version,))
    elif nengo.version.version_info > latest_nengo_version:  # pragma: no cover
        warnings.warn(
            "This version of `nengo_dl` has not been tested with your `nengo` "
            "version (%s). The latest fully supported version is %d.%d.%d." %
            ((nengo.version.version,) + latest_nengo_version))
except ImportError:
    # nengo not installed, can't check version
    pass
