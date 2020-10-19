"""
Build classes for Nengo transform operators.
"""

import warnings

import numpy as np
import tensorflow as tf
from nengo.builder.transforms import ConvInc

from nengo_dl.builder import Builder, OpBuilder
from nengo_dl.compat import ConvTransposeInc


class ConvSet(ConvInc):
    """
    A version of `~nengo.builder.transforms.ConvInc` that overwrites the target
    rather than incrementing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def Y(self):
        """Y is stored in ``sets`` rather than ``incs``."""
        return self.sets[0]


class ConvTransposeSet(ConvTransposeInc):
    """
    A version of `~nengo.builder.transforms.ConvTransposeInc` that overwrites
    the target rather than incrementing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incs, self.sets = self.sets, self.incs

    @property
    def Y(self):
        """Y is stored in ``sets`` rather than ``incs``."""
        return self.sets[0]


@Builder.register(ConvInc)
@Builder.register(ConvSet)
@Builder.register(ConvTransposeInc)
@Builder.register(ConvTransposeSet)
class ConvIncBuilder(OpBuilder):
    """
    Build a group of `nengo.builder.transforms.ConvInc` operators.
    """

    @staticmethod
    def is_transpose_op(op):
        """Returns True if the given op performs transpose convolution."""
        return isinstance(op, (ConvTransposeInc, ConvTransposeSet))

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.conv = self.ops[0].conv
        self.n_ops = len(self.ops)
        self.mode = "inc" if type(self.ops[0]) == ConvInc else "update"
        self.transpose = self.is_transpose_op(self.ops[0])

        if not self.conv.channels_last and config.cpu_only:
            # TensorFlow doesn't support channels first on CPU, so if
            # GPU support isn't available we need to force channels_last
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
                f"Convolutions > {len(fmts)} dimensions are not supported"
            )

        fmt = fmts[self.conv.dimensions - 1]
        self.fmt = (
            "N" + fmt + "C" if self.conv.channels_last or force_last else "NC" + fmt
        )

        self.W_data = signals.combine([op.W for op in self.ops])
        # all the ops have the same input, so we just use one
        self.X_data = signals[self.ops[0].X]
        self.X_data = self.X_data.reshape(self.conv.input_shape.shape)
        self.Y_data = signals.combine([op.Y for op in self.ops])

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

            self.perm_x = tf.constant(perm_x)
        else:
            perm_x = []
            self.perm_x = None

        # set up Y transformations
        if len(self.ops) > 1:
            if self.conv.channels_last or force_last:
                # separate channel dimension into output for each op
                reshape_y = (
                    (signals.minibatch_size,)
                    + self.conv.output_shape.spatial_shape
                    + (-1, len(self.ops))
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
                    len(self.ops),
                ) + self.conv.output_shape.spatial_shape

                # move ops to second axis (after batch)
                perm_y = np.arange(self.conv.dimensions + 3)
                perm_y[[1, 2]] = perm_y[[2, 1]]

            self.reshape_y = tf.constant(reshape_y)
            self.perm_y = tf.constant(perm_y)
        else:
            self.reshape_y = None

            if force_last:
                # switch back to channels-first (after batch)
                perm_y = np.arange(self.conv.dimensions + 2)
                perm_y[-2:] = perm_y[1:-1]
                perm_y[1] = len(perm_y) - 1
                self.perm_y = tf.constant(perm_y)
            else:
                self.perm_y = None

        # set up W transformations
        if len(self.ops) > 1:
            # move ops to end
            self.W_data = self.W_data.reshape((len(self.ops),) + self.conv.kernel_shape)
            self.perm_w = tf.constant(np.roll(np.arange(self.conv.dimensions + 3), -1))

            # concatenate weights for each op along output channel dimension
            self.reshape_w = tf.constant(
                self.conv.kernel_size + (self.conv.input_shape.n_channels, -1)
            )
        else:
            self.perm_w = None
            self.reshape_w = None

        if self.transpose:
            output_space = list(self.conv.output_shape.spatial_shape)
            output_filters = sum(op.conv.n_filters for op in self.ops)

            output_shape = [None] + (
                output_space + [output_filters]
                if self.conv.channels_last
                else [output_filters] + output_space
            )
            if self.perm_x is not None:
                output_shape = [output_shape[i] for i in perm_x]
            # batch size will be filled in dynamically in build_step
            assert output_shape[0] is None
            self.output_shape = output_shape[1:]

            # swap channels, because conv_transpose order is for forward weights
            dims = self.conv.dimensions
            self.perm_w_t = tf.constant(list(range(dims)) + [dims + 1, dims])
        else:
            self.perm_w_t = None
            self.output_shape = None

    def build_step(self, signals):
        W = signals.gather(self.W_data)
        X = signals.gather(self.X_data)
        batch_size = X.shape[0]

        if self.perm_x is not None:
            # move channels to end
            X = tf.transpose(X, perm=self.perm_x)

        if self.perm_w is not None:
            # concatenate kernels along output channel dimension
            W = tf.transpose(W, perm=self.perm_w)
            W = tf.reshape(W, self.reshape_w)

        if self.transpose:
            Y = tf.nn.conv_transpose(
                input=X,
                filters=tf.transpose(W, perm=self.perm_w_t),
                output_shape=[batch_size] + self.output_shape,
                strides=self.conv.strides,
                data_format=self.fmt,
                padding=self.conv.padding.upper(),
            )
        else:
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
            Y = tf.transpose(Y, perm=self.perm_y)

        # tensorflow loses track of shape information during transposes for some reason
        if self.reshape_y is None:
            Y.set_shape((signals.minibatch_size,) + self.conv.output_shape.shape)
        else:
            Y.set_shape(
                (signals.minibatch_size, self.n_ops) + self.conv.output_shape.shape
            )

        signals.scatter(self.Y_data, Y, mode=self.mode)

    @staticmethod
    def mergeable(x, y):
        # we allow convolutions to merge if they have the same input signal
        # (as then we can efficiently apply several kernels to the same input).
        # padding/strides/channels/shape also have to match.
        return (
            x.X is y.X
            and x.conv.input_shape.shape == y.conv.input_shape.shape
            and x.conv.output_shape.spatial_shape == y.conv.output_shape.spatial_shape
            and x.conv.kernel_size == y.conv.kernel_size
            and x.conv.strides == y.conv.strides
            and x.conv.padding == y.conv.padding
            and x.conv.channels_last == y.conv.channels_last
            and ConvIncBuilder.is_transpose_op(x) == ConvIncBuilder.is_transpose_op(y)
        )
