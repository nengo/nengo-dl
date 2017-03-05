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
    root, 'nengo_deeplearning', 'version.py'))['version']

setup(
    name="nengo_deeplearning",
    version=version,
    author="Daniel Rasmussen",
    author_email="daniel.rasmussen@appliedbrainresearch.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/nengo/nengo_deeplearning",
    license="Free for non-commercial use",
    description="Deep learning in Nengo",
    long_description=read('README.rst', 'CHANGES.rst'),
    install_requires=["nengo", "numpy>=1.11", "tensorflow"],
    entry_points={"nengo.backends":
                  ["deeplearning = nengo_deeplearning:Simulator"]},
)
