Installation
============

Installing NengoDL
------------------
We recommend using ``pip`` to install NengoDL:

.. code-block:: bash

  pip install nengo-dl

That's it!

Requirements
------------
NengoDL works with Python 3.5 or later.  ``pip`` will do its best to install
all of NengoDL's requirements when it installs NengoDL.  However, if anything
goes wrong during this process you can install the requirements manually and
then try to ``pip install nengo-dl`` again.
See the `Nengo documentation <https://www.nengo.ai/download/>`_
for instructions on installing ``numpy`` and ``nengo``, and the ``tensorflow``
installation instructions below.

Developer installation
----------------------
If you want to modify NengoDL, or get the very latest updates, you will need to
perform a developer installation:

.. code-block:: bash

  git clone https://github.com/nengo/nengo-dl.git
  pip install -e ./nengo-dl

Installing TensorFlow
---------------------
Use ``pip install tensorflow`` to install the latest version of TensorFlow. GPU support
is included in this package as of version 2.1.0.

In order to use TensorFlow with GPU support you will need to install the appropriate
Nvidia drivers and CUDA/cuDNN. The precise steps for accomplishing this will depend
on your system. On Linux the correct Nvidia drivers (as of TensorFlow 2.2.0) can be
installed via ``sudo apt install nvidia-driver-440``, and on Windows simply using the
most up-to-date drivers should work.  For CUDA/cuDNN we recommend using
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ to
simplify the process. ``conda install tensorflow-gpu`` will install TensorFlow as
well as all the CUDA/cuDNN requirements.  If you run into any problems, see the
`TensorFlow GPU installation instructions <https://www.tensorflow.org/install/gpu>`_
for more details.

It is also possible to build TensorFlow from source.  This is significantly
more complicated but allows you to customize the installation to your
computer, which can improve simulation speeds.

`Instructions for installing on Ubuntu or Mac OS
<https://www.tensorflow.org/install/source>`_.

`Instructions for installing on Windows
<https://www.tensorflow.org/install/source_windows>`_.
