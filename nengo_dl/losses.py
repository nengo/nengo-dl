"""
Some common loss functions (for use with the ``loss`` argument in
`.Simulator.compile`).
"""

import tensorflow as tf


def nan_mse(y_true, y_pred):
    """
    Compute Mean Squared Error between given outputs and targets.

    If any values in ``y_true`` are ``nan``, that will be treated as
    zero error for those elements.

    Parameters
    ----------
    y_true : ``tf.Tensor``
        Target values for a Probe in a network.
    y_pred : ``tf.Tensor``
        Output values from a Probe in a network.

    Returns
    -------
    mse : ``tf.Tensor``
        Tensor representing the mean squared error.
    """

    targets = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
    return tf.reduce_mean(tf.square(targets - y_pred))


class Regularize(tf.losses.Loss):
    """
    An objective function to apply regularization penalties.

    This is designed to be applied to a probed signal, e.g.

    .. testcode::

        with nengo.Network() as net:
            a = nengo.Node([0])
            b = nengo.Node(size_in=1)
            c = nengo.Connection(a, b, transform=1)
            p = nengo.Probe(c, "weights")
            ...

            # this part is optional, but we may often want to only keep the data from
            # the most recent timestep when probing in this way, to save memory
            nengo_dl.configure_settings(keep_history=True)
            net.config[p].keep_history = False

        with nengo_dl.Simulator(net) as sim:
            sim.compile(loss={p: nengo_dl.losses.Regularize()})
    """

    def __init__(self, order=2, axis=None):
        """
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

        Notes
        -----
        The mean will be computed across all the non-``axis`` dimensions after
        computing the norm (including batch/time) in order to compute the overall
        objective value.
        """

        super().__init__()

        self.order = order
        self.axis = axis

    def call(self, y_true, y_pred):
        """
        Invoke the loss function.

        Parameters
        ----------
        y_true : ``tf.Tensor``
            Ignored
        y_pred : ``tf.Tensor``
            The value to be regularized

        Returns
        -------
        output : ``tf.Tensor``
            Scalar regularization loss value.
        """
        if self.axis is None:
            if y_pred.shape.ndims > 3:
                # flatten signal (keeping batch/time dimension)
                y_pred = tf.reshape(
                    y_pred,
                    tf.concat(  # pylint: disable=no-value-for-parameter
                        [tf.shape(y_pred)[:2], (-1,)], axis=0
                    ),
                )
            axis = 2
        else:
            axis = self.axis + 2

        output = tf.reduce_mean(tf.norm(y_pred, axis=axis, ord=self.order))

        return output
