Installation
============

Requirements
------------
- ``python 2.7`` or ``python >= 3.4``
- ``numpy >= 1.11.0``
- ``nengo >= 2.5.0``
- ``tensorflow >= 1.2.0``

Installing NengoDL
------------------
To install NengoDL, we recommend using ``pip``:

.. code-block:: bash

  pip install nengo_dl

``pip`` will do its best to install all of NengoDL's requirements when it
installs NengoDL.  However, if anything goes wrong during this process you
can install the requirements manually.  See the
`Nengo documentation <https://www.nengo.ai/download.html>`_
for instructions on installing ``numpy`` and ``nengo``, and the ``tensorflow``
installation instructions below.

Developer installation
----------------------
If you want to modify NengoDL, or get the very latest updates, you will need to
perform a developer installation:

.. code-block:: bash

  git clone https://github.com/nengo/nengo_dl.git
  pip install -e ./nengo_dl

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
