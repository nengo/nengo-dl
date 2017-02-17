Installation
============

Requirements
------------
- ``numpy >= 1.11.0``
- ``nengo >= 2.3.1``
- ``tensorflow`` (custom fork, see below)

See the
`Nengo documentation <https://pythonhosted.org/nengo/getting_started.html>`_
for instructions on installing ``numpy`` and ``nengo``.

Installing TensorFlow
---------------------
In theory, TensorFlow can be installed through PyPI, via
``pip install tensorflow`` or ``pip install tensorflow-gpu`` (depending on
whether or not you want GPU support).

However, the current released version of TensorFlow (1.0.0) contains a serious
bug in its automatic differentiation, which means that if you try to train
a model using a gradient-based optimization method (which includes all the
standard deep learning optimization methods) the optimization will be
incorrect.

Fortunately, this bug is fixed in our `custom fork of TensorFlow
<https://github.com/drasmuss/tensorflow>`_. Unfortunately, this means that
TensorFlow needs to be installed from source, which is significantly more
complicated.

`Instructions for installing on Ubuntu or Mac OS
<https://www.tensorflow.org/install/install_sources>`_.

`Instructions for installing on Windows
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/README.md>`_.

Note that wherever it says to use ``https://github.com/tensorflow/tensorflow``,
you will instead use ``https://github.com/drasmuss/tensorflow``.

Once this bug is fixed in a released version of TensorFlow this custom
installation will no longer be necessary. Follow `this issue on GitHub
<https://github.com/tensorflow/tensorflow/issues/7397>`_ for updates.

Installing NengoDL
------------------
TODO: pypi release

Developer installation
----------------------
If you want to modify NengoDL, you will need to install it from source:

.. code-block:: bash

  git clone https://github.com/nengo/nengo_deeplearning.git
  cd nengo_deeplearning
  python setup.py develop --user

If you are in a virtual environment (recommended), you can omit the ``--user``
flag.