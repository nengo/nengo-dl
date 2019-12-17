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
Use ``pip install tensorflow`` to install the minimal version of TensorFlow,
or ``pip install tensorflow-gpu`` to include GPU support.

It is also possible to install TensorFlow from source.  This is significantly
more complicated but allows you to customize the installation to your
computer, which can improve simulation speeds.

`Instructions for installing on Ubuntu or Mac OS
<https://www.tensorflow.org/install/source>`_.

`Instructions for installing on Windows
<https://www.tensorflow.org/install/source_windows>`_.
