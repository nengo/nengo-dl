#!/usr/bin/env python
import io
import runpy
import os
import pkg_resources
import sys

try:
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
    installed_dists = [d.project_name for d in pkg_resources.working_set]
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

install_requires = [
    "nengo>=2.7.0",
    "numpy>=1.12.1",
    "%s>=1.4.0" % tf_req,
    "progressbar2>=3.37.1",
]
docs_require = [
    "jupyter>=1.0.0",
    "matplotlib>=2.0.0",
    "nbconvert!=5.4",  # doesn't work with nbsphinx TODO: remove this once there's an upstream fix; https://github.com/jupyter/nbconvert/issues/878
    "nbsphinx>=0.2.13",
    "nengo-sphinx-theme>=0.7.0",
    "numpydoc>=0.6.0",
    "Pillow>=4.1.1",
    "sphinx>=1.8.0",
    "sphinx-click>=1.4.1"
]
tests_require = [
    "click>=6.7",
    "codespell>=1.12.0",
    "coverage>=4.3.4",
    "matplotlib>=2.0.0",
    "nbval>=0.6.0",
    "pylint>=1.9.2",
    "pytest>=3.0.4",
    "pytest-xdist>=1.16.0",
]

setup(
    name="nengo-dl",
    version=version,
    author="Daniel Rasmussen",
    author_email="daniel.rasmussen@appliedbrainresearch.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://www.nengo.ai/nengo-dl",
    license="Free for non-commercial use",
    description="Deep learning integration for Nengo",
    long_description=read('README.rst', 'CHANGES.rst'),
    install_requires=install_requires,
    extras_require={
        "all": docs_require + tests_require,
        "docs": docs_require,
        "tests": tests_require,
    },
    python_requires=">=3.4",
    entry_points={"nengo.backends": ["dl = nengo_dl:Simulator"]},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
