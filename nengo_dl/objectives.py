"""
Some common objective functions (for use with the ``objective`` argument in
`.Simulator.train` or `.Simulator.loss`).
"""

import tensorflow as tf


def mse(outputs, targets):
    """
    Compute Mean Squared Error between given outputs and targets.

    If any values in ``targets`` are ``nan``, that will be treated as
    zero error for those elements.

    Parameters
    ----------
    outputs : ``tf.Tensor``
        Output values from a Probe in a network.
    targets : ``tf.Tensor``
        Target values for a Probe in a network.

    Returns
    -------
    mse : ``tf.Tensor``
        Tensor representing the mean squared error.
    """

    targets = tf.where(tf.is_nan(targets), outputs, targets)
    return tf.reduce_mean(tf.square(targets - outputs))


class Regularize:
    """
    An objective function to apply regularization penalties.

    Parameters
    ----------
    order : int or str
        Order of the regularization norm (e.g. ``1`` for L1 norm, ``2`` for
        L2 norm).  See https://www.tensorflow.org/api_docs/python/tf/norm for
        a full description of the possible values for this parameter.
    axis : int or None
        The axis of the probed signal along which to compute norm.  If None
        (the default), the signal is flattened and the norm is computed across
        the resulting vector.  Note that these are only the axes with respect
        to the output on a single timestep (i.e. batch/time dimensions are not
        included).
    weight : float
        Scaling weight to apply to regularization penalty.

    Notes
    -----
    The mean will be computed across all the non-``axis`` dimensions after
    computing the norm (including batch/time) in order to compute the overall
    objective value.
    """

    def __init__(self, order=2, axis=None, weight=None):
        self.order = order
        self.axis = axis
        self.weight = weight

    def __call__(self, x):
        if self.axis is None:
            if x.get_shape().ndims > 3:
                # flatten signal (keeping batch/time dimension)
                x = tf.reshape(x, tf.concat([tf.shape(x)[:2], (-1,)], axis=0))
            axis = 2
        else:
            axis = self.axis + 2

        output = tf.reduce_mean(tf.norm(x, axis=axis, ord=self.order))

        if self.weight is not None:
            output *= self.weight

        return output
