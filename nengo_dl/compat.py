# pylint: disable=unused-import,ungrouped-imports

"""
Utilities to ease cross-compatibility between different versions of upstream
dependencies.
"""

from distutils.version import LooseVersion

import nengo
import tensorflow as tf

if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    tf_compat = tf

    def tf_convolution(*args, **kwargs):
        """Convert "filters" kwarg to "filter"."""

        if "filters" in kwargs:
            kwargs["filter"] = kwargs.pop("filters")
        return tf.nn.convolution(*args, **kwargs)

    tf_math = tf

    def tf_shape(shape):
        """Convert ``tf.Dimension`` to int."""

        return tuple(x.value for x in shape)

    tf_uniform = tf.random_uniform

    RefVariable = tf.Variable
else:
    tf_compat = tf.compat.v1

    tf_convolution = tf.nn.convolution

    tf_math = tf.math

    def tf_shape(shape):
        """Return shape (elements are already ints)."""

        return shape

    tf_uniform = tf.random.uniform

    def RefVariable(*args, **kwargs):
        """Always returns RefVariables instead of (maybe) ResourceVariables."""

        return tf.compat.v1.Variable(*args, use_resource=False, **kwargs)


if LooseVersion(nengo.__version__) < "3.0.0":
    class SimPES(nengo.builder.Operator):  # pylint: disable=abstract-method
        r"""
        Calculate connection weight change according to the PES rule.

        Implements the PES learning rule of the form

        .. math:: \Delta \omega_{ij} = \frac{\kappa}{n} e_j a_i

        where

        * :math:`\kappa` is a scalar learning rate,
        * :math:`n` is the number of presynaptic neurons
        * :math:`e_j` is the error for the jth output dimension, and
        * :math:`a_i` is the activity of a presynaptic neuron.

        Parameters
        ----------
        pre_filtered : Signal
            The presynaptic activity, :math:`a_i`.
        error : Signal
            The error signal, :math:`e_j`.
        delta : Signal
            The synaptic weight change to be applied,
            :math:`\Delta \omega_{ij}`.
        learning_rate : float
            The scalar learning rate, :math:`\kappa`.
        tag : str (Default: None)
            A label associated with the operator, for debugging purposes.

        Attributes
        ----------
        pre_filtered : Signal
            The presynaptic activity, :math:`a_i`.
        error : Signal
            The error signal, :math:`e_j`.
        delta : Signal
            The synaptic weight change to be applied,
            :math:`\Delta \omega_{ij}`.
        learning_rate : float
            The scalar learning rate, :math:`\kappa`.
        tag : str (Default: None)
            A label associated with the operator, for debugging purposes.

        Notes
        -----
        1. sets ``[delta]``
        2. incs ``[]``
        3. reads ``[pre_filtered, error]``
        4. updates ``[]``
        """

        def __init__(self, pre_filtered, error, delta, learning_rate,
                     encoders=None, tag=None):
            super(SimPES, self).__init__(tag=tag)

            self.pre_filtered = pre_filtered
            self.error = error
            self.delta = delta
            self.learning_rate = learning_rate
            self.encoders = encoders

            # encoders not used in NengoDL (they'll be applied outside the op)
            assert encoders is None

            # note: in 3.0.0 the PES op changed from a set to an update
            self.sets = [delta]
            self.incs = []
            self.reads = [pre_filtered, error]
            self.updates = []

        def _descstr(self):
            return 'pre=%s, error=%s -> %s' % (
                self.pre_filtered, self.error, self.delta)

    # remove 'correction' from probeable attributes
    nengo.PES.probeable = ("error", "activities", "delta")

    class Convolution:
        """Dummy `nengo.transforms.Convolution` class."""

    class ConvInc:
        """Dummy `nengo.builder.transforms.ConvInc` class."""

    class SparseMatrix:
        """Dummy `nengo.transforms.SparseMatrix` class."""

    class SparseDotInc:
        """Dummy `nengo.builder.transforms.SparseDotInc` class."""

    def is_sparse(sig):
        """Check if Signal is sparse"""
        # always False since Sparse signals didn't exist, but we use getattr
        # so that dummies.Signal(sparse=False) will still work
        return getattr(sig, "sparse", False)
else:
    from nengo.builder.learning_rules import SimPES
    from nengo.builder.transforms import ConvInc, SparseDotInc
    from nengo.transforms import Convolution, SparseMatrix

    def is_sparse(sig):
        """Check if Signal is sparse"""
        return sig.sparse
