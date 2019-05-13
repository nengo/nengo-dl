"""Utilities to ease cross-compatibility between TF 1.0 and TF 2.0."""

from distutils.version import LooseVersion

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
