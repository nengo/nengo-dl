#!/usr/bin/env python
import io
import runpy
import os

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")

# check if tensorflow-gpu is installed (so that we don't force tensorflow to
# be installed if tensorflow-gpu is already there)
try:
    from tensorflow.python.client import device_lib  # noqa: E402

    if not any(["gpu" in x.name for x in device_lib.list_local_devices()]):
        raise ImportError()

    tf_req = "tensorflow-gpu>=1.2.0"
except ImportError:
    tf_req = "tensorflow>=1.2.0"


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
    install_requires=["nengo>=2.3.1", "numpy>=1.11", tf_req],
    entry_points={"nengo.backends":
                  ["dl = nengo_dl:Simulator"]},
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
