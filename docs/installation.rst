Installation
============

Requirements
------------
- ``python 2.7`` or ``python >= 3.4``
- ``numpy >= 1.11.0``
- ``nengo >= 2.3.1``
- ``tensorflow >= 1.0.0``

See the
`Nengo documentation <https://pythonhosted.org/nengo/getting_started.html>`_
for instructions on installing ``numpy`` and ``nengo``.

Installing TensorFlow
---------------------
Use ``pip install tensorflow`` to install the minimal version of TensorFlow,
or ``pip install tensorflow-gpu`` to include GPU support.

It is also possible to install TensorFlow from source.  This is significantly
more complicated but allows you to customize the installation to your
computer, which can improve simulation speeds.

`Instructions for installing on Ubuntu or Mac OS
<https://www.tensorflow.org/install/install_sources>`_.

`Instructions for installing on Windows
<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/README.md>`_.

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
