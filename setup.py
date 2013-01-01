#!/usr/bin/env python
import io
import runpy
import os
import sys

try:
    from pip import get_installed_distributions
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'pip'/'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")

if "bdist_wheel" in sys.argv:
    # when building wheels we have to pick a requirement ahead of time (can't
    # check it at install time). so we'll go with tensorflow (non-gpu), since
    # that is the safest option
    tf_req = "tensorflow"
else:
    # check if one of the tensorflow packages is already installed (so that we
    # don't force tensorflow to be installed if e.g. tensorflow-gpu is already
    # there)
    tf_dists = ["tf-nightly-gpu", "tf-nightly", "tensorflow-gpu"]
    installed_dists = [d.project_name for d in get_installed_distributions()]
    for d in tf_dists:
        if d in installed_dists:
            tf_req = d
            break
    else:
        tf_req = "tensorflow"


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(
    root, 'nengo_dl', 'version.py'))['version']

setup(
    name="nengo_dl",
    version=version,
    author="Daniel Rasmussen",
    author_email="daniel.rasmussen@appliedbrainresearch.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/nengo/nengo_dl",
    license="Free for non-commercial use",
    description="Deep learning integration for Nengo",
    long_description=read('README.rst', 'CHANGES.rst'),
    install_requires=["nengo>=2.7.0", "numpy>=1.11", "%s>=1.3.0" % tf_req,
                      "progressbar2>=3.34.0",
                      "backports.tempfile;python_version<'3.4'"],
    entry_points={"nengo.backends": ["dl = nengo_dl:Simulator"]},
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: Free for non-commercial use',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering'],
)
