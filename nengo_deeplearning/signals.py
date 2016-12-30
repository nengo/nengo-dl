from collections import defaultdict

from nengo.builder.signal import Signal, SignalError
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG


class TensorSignal(object):
    def __init__(self, indices, key, label="TensorSignal", display_shape=None):
        # indices into the base array corresponding to this signal
        self.indices = indices

        # dtype of base array
        self.key = key

        # shape
        self.display_shape = display_shape

        self.label = label

    @property
    def dtype(self):
        return self.key[0]

    @property
    def base_shape(self):
        return self.key[1]

    @property
    def shape(self):
        if self.display_shape is None:
            return (len(self.indices),) + self.base_shape[1:]
        return self.display_shape

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return "TensorSignal(key=%s, shape=%s, label=%s)" % (
            self.key, self.shape, self.label)

    def __getitem__(self, indices):
        if indices is Ellipsis:
            return self
        return TensorSignal(self.indices[indices], self.key,
                            label=self.label + ".slice")

    def broadcast(self, axis, length):
        assert axis in (0, 1)

        indices = self.indices
        indices = np.expand_dims(indices, axis)
        tile = [length if i == axis else 1 for i in range(indices.ndim)]
        indices = np.tile(indices, tile).flatten()

        if axis == 1:
            display_shape = self.shape + (length,)
        else:
            display_shape = (length,) + self.shape

        return TensorSignal(
            indices, self.key, display_shape=display_shape,
            label=self.label + ".broadcast(%d, %d)" % (axis, length))

    def reshape(self, shape):
        if np.prod(shape) != np.prod(self.shape):
            raise BuildError("Number of elements don't match in reshape")

        return TensorSignal(
            self.indices, self.key, display_shape=shape,
            label=self.label + ".reshape(%s)" % (shape,))

    def tile(self, length):
        # repeat along the first axis the given number of times
        indices = np.tile(self.indices, length)
        shape = (self.shape[0] * length,) + self.shape[1:]

        return TensorSignal(
            indices, self.key, display_shape=shape,
            label=self.label + ".tile(%d)" % length)


class SignalDict(dict):
    """Map from Signal -> Tensor

    Takes care of view/base logic
    """

    def __init__(self, sig_map, bases, dtype, *args, **kwargs):
        super(SignalDict, self).__init__(*args, **kwargs)
        self.bases = bases
        self.dtype = dtype
        self.sig_map = sig_map
        self.reads_by_base = defaultdict(list)

    def __getitem__(self, sigs):
        # try:
        #     return dict.__getitem__(self, key)
        # except KeyError:
        #     if isinstance(key, Signal) and key.is_view:
        #         # return a view on the base signal
        #         base = dict.__getitem__(self, key.base)
        #
        #         if key.dtype != key.base.dtype:
        #             base = tf.cast(base, key.dtype)
        #
        #         if key.initial_value.ndim != key.base.ndim:
        #             if key.size != key.base.size:
        #                 # TODO: support this
        #                 raise NotImplementedError(
        #                     "Slicing and reshaping the same signal is not "
        #                     "supported")
        #
        #             view = tf.reshape(base, key.shape)
        #         else:
        #             offset = np.unravel_index(key.elemoffset, key.base.shape)
        #             shape = np.asarray(key.shape)
        #             strides = np.asarray(key.elemstrides)
        #             end = offset + shape * strides
        #
        #             end_mask = np.int32(0)
        #             for i, b in enumerate(end < 0):
        #                 if b:
        #                     end_mask += 1 << i
        #
        #             view = tf.strided_slice(
        #                 base, offset, end, strides, end_mask=end_mask)
        #         dict.__setitem__(self, key, view)
        #         return view
        #     else:
        #         raise

        if not isinstance(sigs, (list, tuple)):
            sigs = [sigs]

        if len(sigs) == 0:
            return None

        return self.gather(sigs)

    def __setitem__(self, sigs, val):
        """Assign Tensor `val` to Signal `sig`.

        If modifying a Variable, this modifies the underlying state of the
        variable.  If modifying a Tensor, no need to keep the state around
        so we just overwrite the entry.
        """

        # if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
        #     raise BuildError("Tensor detected with wrong dtype (%s), should "
        #                      "be %s." % (val.dtype.base_dtype, self.dtype))
        #
        # if key in self.variables or (
        #             hasattr(key, "base") and key.base in self.variables):
        #     self.assign_view(key, val)
        # else:
        #     dict.__setitem__(self, key, val)

        if not isinstance(sigs, (list, tuple)):
            sigs = [sigs]

        self.scatter(sigs, val, mode="update")

    def _key_and_indices(self, sigs):
        assert isinstance(sigs, (list, tuple))
        assert isinstance(sigs[0], (Signal, TensorSignal))

        sigs = [self.sig_map[s] if isinstance(s, Signal) else s for s in sigs]

        key = sigs[0].key
        assert all([s.key == key for s in sigs])

        indices = np.concatenate([s.indices for s in sigs], axis=0)

        assert all([s.shape[1:] == sigs[0].shape[1:] for s in sigs])
        display_shape = (np.sum([s.shape[0]
                                 for s in sigs]),) + sigs[0].shape[1:]

        return key, indices, display_shape

    def scatter(self, sigs, val, mode="inc"):
        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError("Tensor detected with wrong dtype (%s), should "
                             "be %s." % (val.dtype.base_dtype, self.dtype))

        key, indices, _ = self._key_and_indices(sigs)

        if isinstance(val, tf.IndexedSlices):
            # source is already an indexed slice, so we just need to change
            # the indices
            assert len(indices) == val.indices.get_shape()[0]
            val.indices = tf.constant(indices)
        else:
            # source is a tensor (representing a dense subset of base), so
            # we change it to a sparse representation of the full base shape

            # reshape to appropriate shape (adding/removing empty dimensions)
            val = tf.reshape(val, (val.get_shape().as_list()[0],) + key[1][1:])

            val = tf.IndexedSlices(val, indices)

        if mode == "update":
            scatter_f = tf.scatter_update
        elif mode == "inc":
            scatter_f = tf.scatter_add
        elif mode == "mul":
            scatter_f = tf.scatter_mul

        if DEBUG:
            print("scatter")
            print("targets", [x for x in sigs])
            print("values", val)
            print("dst", self.bases[key])

        with tf.control_dependencies(self.reads_by_base[self.bases[key]]):
            self.bases[key] = scatter_f(self.bases[key], val.indices,
                                        val.values)

        if DEBUG:
            print("new dst", self.bases[key])

    def gather(self, sigs):
        key, indices, shape = self._key_and_indices(sigs)

        result = tf.gather(self.bases[key], indices)

        if shape is not None:
            result = tf.reshape(result, shape)

        # whenever we read from an array we use this to mark it as "read"
        # (so that any future writes to the array will be scheduled after
        # the read)
        self.reads_by_base[self.bases[key]] += [result]

        return result

    def combine(self, sigs):
        """Combines several TensorSignals into one."""

        key, indices, shape = self._key_and_indices(sigs)

        return TensorSignal(indices, key, display_shape=shape)

    # def __contains__(self, key):
    #     # differs from standard in that it returns True for views
    #     return dict.__contains__(self, key) or (
    #         hasattr(key, "base") and dict.__contains__(self, key.base))
    #
    # def inc(self, key, val):
    #     # basically equivalent to self[key] += val, with some added logic
    #     # for efficiency
    #     if getattr(self[key], "zero_constant", False):
    #         self[key] = val
    #     else:
    #         self[key] += val

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s: %s" % (repr(k), repr(self[k]))
                          for k in self])

        # def create_variable(self, signal):
        #     """Create a Variable (persistent state) for the given signal."""
        #
        #     if signal in self.variables:
        #         raise SignalError("%s is already associated with a "
        #                           "Variable" % signal)
        #
        #     if signal.is_view:
        #         if signal.base not in self.variables:
        #             assert signal.base not in self
        #
        #             self.create_variable(signal.base)
        #     else:
        #         name = utils.sanitize_name(signal.name)
        #
        #         x = tf.Variable(signal.initial_value, name=name,
        #                         dtype=utils.cast_dtype(signal.dtype, self.dtype))
        #
        #         if DEBUG:
        #             print("init base", x)
        #
        #         dict.__setitem__(self, signal, x)
        #
        #         self.variables[signal] = x
        #
        # def assign_view(self, dst, src, src_slice=Ellipsis, dst_slice=Ellipsis,
        #                 inc=False):
        #     if isinstance(src, Signal):
        #         src = self[src]
        #         if isinstance(src_slice, slice):
        #             src = src[src_slice]
        #         elif src_slice is not Ellipsis:
        #             # advanced indexing
        #             src = tf.gather(src, tf.constant(src_slice))
        #
        #     # get the base variable (e.g., if this is a slice)
        #     dst_var = self.get_variable(self[dst])
        #     if dst_var is None:
        #         raise BuildError("Attempted to assign to signal that is not based"
        #                          " on a Variable")
        #
        #     if DEBUG:
        #         print("assigning %s to %s" % (src, dst_var))
        #
        #     if not dst.is_view and dst_slice is Ellipsis:
        #         if inc:
        #             result = tf.assign_add(dst_var, src)
        #         else:
        #             result = tf.assign(dst_var, src)
        #     else:
        #         # TODO: make sliced assignment work for multidimensional arrays
        #         assert dst.ndim == 1
        #
        #         if dst_slice is Ellipsis:
        #             start = dst.elemoffset
        #             stride = dst.elemstrides[0]
        #             stop = dst.elemoffset + dst.size * dst.elemstrides[0]
        #
        #             indices = tf.range(start, stop, stride)
        #         elif isinstance(dst_slice, slice):
        #             start = dst.elemoffset + dst_slice.start * dst.elemstrides[0]
        #             stride = dst.elemstrides[0] * dst_slice.step
        #             stop = dst.elemoffset + dst_slice.stop * dst.elemstrides[0]
        #
        #             indices = tf.range(start, stop, stride)
        #         else:
        #             # advanced indexing
        #             indices = np.asarray(dst_slice)
        #
        #             indices *= dst.elemstrides[0]
        #             indices += dst.elemoffset
        #
        #             indices = tf.constant(indices)
        #
        #         if DEBUG:
        #             print("indices", indices)
        #
        #         if inc:
        #             result = tf.scatter_add(dst_var, indices, src)
        #         else:
        #             result = tf.scatter_update(dst_var, indices, src)
        #
        #         # we also need to update the base signal, so that future operations
        #         # on the base get the updated value
        #         dict.__setitem__(self, dst.base, result)
        #
        #     dict.__setitem__(self, dst, result)
        #
        #     if DEBUG:
        #         print("result", self[dst], dst)
        #
        #     return result
        #
        # def get_variable(self, tensor):
        #     """Trace a Tensor backwards to find the base Variable."""
        #
        #     if tensor.dtype._is_ref_dtype:
        #         return tensor
        #
        #     my_base = None
        #     for input in set(tensor.op.inputs):
        #         base = self.get_variable(input)
        #         if base is not None:
        #             if my_base is None:
        #                 my_base = base
        #             elif base is not my_base:
        #                 raise BuildError(
        #                     "Multiple base variables found for tensor")
        #
        #     return my_base
