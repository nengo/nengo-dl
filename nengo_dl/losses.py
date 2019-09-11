"""
Some common loss functions (for use with the ``loss`` argument in
`.Simulator.compile`).
"""

import tensorflow as tf


class Regularize(tf.losses.Loss):
    """
    An objective function to apply regularization penalties.

    This is designed to be applied to a probed signal, e.g.

    .. code-block:: python

        with nengo.Network() as net:
            ...
            c = nengo.Connection(a, b)
            p = nengo.Probe(c, "weights")
            ...

            # this part is optional, but we may often want to only keep the data from
            # the most recent timestep when probing in this way, to save memory
            nengo_dl.configure_settings(keep_history=True)
            net.config[p].keep_history = False


        with nengo_dl.Simulator(net) as sim:
            sim.compile(loss={p: Regularize()})

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

    def __init__(self, order=2, axis=None):
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
        y_pure : ``tf.Tensor``
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
                    y_pred, tf.concat([tf.shape(y_pred)[:2], (-1,)], axis=0)
                )
            axis = 2
        else:
            axis = self.axis + 2

        output = tf.reduce_mean(tf.norm(y_pred, axis=axis, ord=self.order))

        return output
