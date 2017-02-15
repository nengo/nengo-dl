import io
import runpy
import os

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(root, 'nengo_deeplearning',
                                      'version.py'))['version']
description = "Deep learning in Nengo"
long_description = read('README.rst', 'CHANGES.rst')

url = "https://github.com/nengo/nengo_deeplearning"
setup(
    name="nengo_deeplearning",
    version=version,
    author="Daniel Rasmussen",
    author_email="daniel.rasmussen@appliedbrainresearch.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url=url,
    license="",
    description=description,
    long_description=long_description,
    install_requires=["nengo", "numpy", "tensorflow"],
    entry_points={"nengo.backends":
                  ["deeplearning = nengo_deeplearning:Simulator"]},
)
