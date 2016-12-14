from nengo.builder.signal import Signal, SignalError
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG


class SignalDict(dict):
    """Map from Signal -> Tensor

    Takes care of view/base logic
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if isinstance(key, Signal) and key.is_view:
                # return a view on the base signal
                base = dict.__getitem__(self, key.base)

                if key.dtype != key.base.dtype:
                    base = tf.cast(base, key.dtype)

                if key.initial_value.ndim != key.base.ndim:
                    if key.size != key.base.size:
                        # TODO: support this
                        raise NotImplementedError(
                            "Slicing and reshaping the same signal is not "
                            "supported")

                    view = tf.reshape(base, key.shape)
                else:
                    offset = np.unravel_index(key.elemoffset, key.base.shape)
                    shape = np.asarray(key.shape)
                    strides = np.asarray(key.elemstrides)
                    end = offset + shape * strides

                    end_mask = np.int32(0)
                    for i, b in enumerate(end < 0):
                        if b:
                            end_mask += 1 << i

                    view = tf.strided_slice(
                        base, offset, end, strides, end_mask=end_mask)
                dict.__setitem__(self, key, view)
                return view
            else:
                raise

    def __setitem__(self, key, val):
        """Assign `val` to `key`.

        Unlike normal dicts, this means that you cannot add a new key
        to a SignalDict using __setitem__. This is by design, to avoid
        silent typos when debugging Simulator. Every key must instead
        be explicitly initialized with SignalDict.init.
        """

        assert key in self
        dict.__setitem__(self, key, val)

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s %s" % (repr(k), repr(self[k]))
                          for k in self])

    def init(self, signal):
        """Set up a permanent mapping from signal -> tensor."""
        if signal in self:
            raise SignalError("Cannot add signal twice")

        if signal.is_view:
            if signal.base not in self:
                self.init(signal.base)

            if DEBUG:
                print("init view of", self[signal.base], self[signal])

            # get a view onto the base data
            dict.__setitem__(self, signal, self[signal])
        else:
            name = utils.sanitize_name(signal.name)

            x = tf.Variable(signal.initial_value, name=name,
                            dtype=signal.dtype)

            if DEBUG:
                print("init base", x)

            dict.__setitem__(self, signal, x)

        return self[signal]

    def assign_view(self, dst, src, src_slice=Ellipsis, dst_slice=Ellipsis,
                    inc=False):
        # TODO: optimize out slice(None,None,None)

        if isinstance(src, Signal):
            src = self[src]
            if isinstance(src_slice, slice):
                src = src[src_slice]
            elif src_slice is not Ellipsis:
                # advanced indexing
                src = tf.gather(src, tf.constant(src_slice))

        dst_var = self.get_variable(self[dst])
        assert dst_var is not None

        if DEBUG:
            print("assigning %s to %s" % (src, dst_var))

        if not dst.is_view and dst_slice is Ellipsis:
            if inc:
                result = tf.assign_add(dst_var, src)
            else:
                result = tf.assign(dst_var, src)
        else:
            # TODO: make sliced assignment work for multidimensional arrays
            assert dst.ndim == 1

            if dst_slice is Ellipsis:
                start = dst.elemoffset
                stride = dst.elemstrides[0]
                stop = dst.elemoffset + dst.size * dst.elemstrides[0]

                indices = tf.range(start, stop, stride)
            elif isinstance(dst_slice, slice):
                start = dst.elemoffset + dst_slice.start * dst.elemstrides[0]
                stride = dst.elemstrides[0] * dst_slice.step
                stop = dst.elemoffset + dst_slice.stop * dst.elemstrides[0]

                indices = tf.range(start, stop, stride)
            else:
                # advanced indexing
                indices = np.asarray(dst_slice)

                indices *= dst.elemstrides[0]
                indices += dst.elemoffset

                indices = tf.constant(indices)

            if DEBUG:
                print("indices", indices)

            if inc:
                result = tf.scatter_add(dst_var, indices, src)
            else:
                result = tf.scatter_update(dst_var, indices, src)

            # we also need to update the base signal, so that future operations
            # on the base get the updated value
            self[dst.base] = result

        self[dst] = result

        if DEBUG:
            print("result", self[dst], dst)

        return result

    def get_variable(self, tensor):
        """Trace a Tensor backwards to find the base Variable."""

        # TODO: is this still necessary, or did it get fixed by something
        # else during the debugging?

        if tensor.dtype._is_ref_dtype:
            return tensor

        my_base = None
        for input in set(tensor.op.inputs):
            base = self.get_variable(input)
            if base is not None:
                if my_base is not None:
                    raise BuildError(
                        "Multiple base variables found for tensor")
                else:
                    my_base = base

        return my_base
