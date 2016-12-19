from nengo.builder.signal import Signal, SignalError
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG


class SignalDict(dict):
    """Map from Signal -> Tensor

    Takes care of view/base logic
    """
    def __init__(self, dtype, *args, **kwargs):
        super(SignalDict, self).__init__(*args, **kwargs)
        self.dtype = dtype
        self.variables = {}

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

        If modifying a Variable, this modifies the underlying state of the
        variable.  If modifying a Tensor, no need to keep the state around
        so we just overwrite the entry.
        """

        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError("Tensor detected with wrong dtype (%s), should "
                             "be %s." % (val.dtype.base_dtype, self.dtype))

        if key in self.variables:
            self.assign_view(key, val)
        else:
            dict.__setitem__(self, key, val)

    def inc(self, key, val):
        # basically equivalent to self[key] += val, with some added logic
        # for efficiency
        if getattr(self[key], "zero_constant", False):
            self[key] = val
        else:
            self[key] += val

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s: %s" % (repr(k), repr(self[k]))
                          for k in self])

    def create_variable(self, signal):
        """Create a Variable (persistent state) for the given signal."""

        if signal in self.variables:
            raise SignalError("%s is already associated with a "
                              "Variable" % signal)

        if signal.is_view:
            if signal.base not in self.variables:
                if signal.base in self:
                    # TODO: what do we do if there's already a tensor signal?
                    # does this happen?
                    raise NotImplementedError
                self.create_variable(signal.base)

            if DEBUG:
                print("init view of", self.variables[signal.base],
                      self[signal])

            # get a view onto the base data
            dict.__setitem__(self, signal, self[signal])
        else:
            name = utils.sanitize_name(signal.name)

            x = tf.Variable(signal.initial_value, name=name,
                            dtype=utils.cast_dtype(signal.dtype, self.dtype))

            if DEBUG:
                print("init base", x)

            dict.__setitem__(self, signal, x)

        self.variables[signal] = self[signal]

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

        # get the base variable (e.g., if this is a slice)
        dst_var = self.get_variable(self[dst])
        if dst_var is None:
            raise BuildError("Attempted to assign to signal that is not based"
                             " on a Variable")

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
            dict.__setitem__(self, dst.base, result)

        dict.__setitem__(self, dst, result)

        if DEBUG:
            print("result", self[dst], dst)

        return result

    def get_variable(self, tensor):
        """Trace a Tensor backwards to find the base Variable."""

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
