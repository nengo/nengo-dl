"""
Build classes for Nengo transform operators.
"""

import warnings

import numpy as np
import tensorflow as tf

from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.compat import ConvInc


@Builder.register(ConvInc)
class ConvIncBuilder(OpBuilder):
    """
    Build a group of ``ConvInc`` operators.
    """

    # TODO: fix link to `~nengo.builder.transform.ConvInc` once it exists

    def __init__(self, ops, signals, config):
        super(ConvIncBuilder, self).__init__(ops, signals, config)

        self.conv = ops[0].conv

        if not self.conv.channels_last and config.cpu_only:
            # TensorFlow doesn't support channels first on CPU, so if
            # tensorflow-gpu isn't installed we need to force channels_last
            # TODO: check if this is supported in future versions
            warnings.warn(
                "TensorFlow does not support convolution with "
                "channels_last=False on the CPU; inputs will be transformed "
                "to channels_last=True",
                UserWarning,
            )
            force_last = True
        else:
            force_last = False

        # create data format string
        fmts = ["W", "HW", "DHW"]

        if self.conv.dimensions > len(fmts):
            raise NotImplementedError(
                "Convolutions > %d dimensions are not supported" % len(fmts)
            )

        fmt = fmts[self.conv.dimensions - 1]
        self.fmt = (
            "N" + fmt + "C" if self.conv.channels_last or force_last else "NC" + fmt
        )

        self.W_data = signals.combine([op.W for op in ops])
        # all the ops have the same input, so we just use one
        self.X_data = signals[ops[0].X]
        self.X_data = self.X_data.reshape(self.conv.input_shape.shape)
        self.Y_data = signals.combine([op.Y for op in ops])

        assert self.X_data.minibatched
        if self.W_data.minibatched:
            raise NotImplementedError(
                "Minibatched convolutional weights are not supported"
            )

        # set up X transformations
        if force_last:
            perm_x = np.arange(self.conv.dimensions + 2)

            # move channel dimension to the end
            perm_x[1:-1] = perm_x[2:]
            perm_x[-1] = 1

            self.perm_x = signals.constant(perm_x)
        else:
            self.perm_x = None

        # set up Y transformations
        if len(ops) > 1:
            if self.conv.channels_last or force_last:
                # separate channel dimension into output for each op
                reshape_y = (
                    (signals.minibatch_size,)
                    + self.conv.output_shape.spatial_shape
                    + (-1, len(ops))
                )

                # move ops to second axis (after batch)
                perm_y = np.arange(self.conv.dimensions + 3)
                perm_y[2:] = perm_y[1:-1]
                perm_y[1] = len(perm_y) - 1

                if force_last:
                    # switch back to channels-first (after batch/ops)
                    perm_y[-2:] = perm_y[2:-1]
                    perm_y[2] = len(perm_y) - 2
            else:
                # separate channel dimension into output for each op
                reshape_y = (
                    signals.minibatch_size,
                    -1,
                    len(ops),
                ) + self.conv.output_shape.spatial_shape

                # move ops to second axis (after batch)
                perm_y = np.arange(self.conv.dimensions + 3)
                perm_y[[1, 2]] = perm_y[[2, 1]]

            self.reshape_y = signals.constant(reshape_y)
            self.perm_y = signals.constant(perm_y)
        else:
            self.reshape_y = None

            if force_last:
                # switch back to channels-first (after batch)
                perm_y = np.arange(self.conv.dimensions + 2)
                perm_y[-2:] = perm_y[1:-1]
                perm_y[1] = len(perm_y) - 1
                self.perm_y = signals.constant(perm_y)
            else:
                self.perm_y = None

        # set up W transformations
        if len(ops) > 1:
            # move ops to end
            self.W_data = self.W_data.reshape((len(ops),) + self.conv.kernel_shape)
            self.perm_w = signals.constant(
                np.roll(np.arange(self.conv.dimensions + 3), -1)
            )

            # concatenate weights for each op along output channel dimension
            self.reshape_w = signals.constant(
                self.conv.kernel_size + (self.conv.input_shape.n_channels, -1)
            )
        else:
            self.perm_w = None
            self.reshape_w = None

    def build_step(self, signals):
        W = signals.gather(self.W_data)
        X = signals.gather(self.X_data)

        if self.perm_x is not None:
            # move channels to end
            X = tf.transpose(a=X, perm=self.perm_x)

        if self.perm_w is not None:
            # concatenate kernels along output channel dimension
            W = tf.transpose(a=W, perm=self.perm_w)
            W = tf.reshape(W, self.reshape_w)

        Y = tf.nn.convolution(
            input=X,
            filters=W,
            strides=self.conv.strides,
            data_format=self.fmt,
            padding=self.conv.padding.upper(),
        )

        if self.reshape_y is not None:
            Y = tf.reshape(Y, self.reshape_y)
        if self.perm_y is not None:
            Y = tf.transpose(a=Y, perm=self.perm_y)

        signals.scatter(self.Y_data, Y, mode="inc")

    @staticmethod
    def mergeable(x, y):
        # we allow convolutions to merge if they have the same input signal
        # (as then we can efficiently apply several kernels to the same input).
        # padding/strides/channels/shape also have to match.
        return (
            x.X is y.X
            and x.conv.input_shape.shape == y.conv.input_shape.shape
            and x.conv.strides == y.conv.strides
            and x.conv.padding == y.conv.padding
            and x.conv.channels_last == y.conv.channels_last
        )
