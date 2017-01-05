from collections import defaultdict
from functools import partial

from nengo.builder.signal import Signal, SignalError
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

from nengo_deeplearning import utils, DEBUG


class TensorSignal(object):
    def __init__(self, indices, key, label="TensorSignal", display_shape=None):
        # indices into the base array corresponding to this signal
        if isinstance(indices, (tuple, list, np.ndarray)):
            # we allow the original indices creation to happen outside of
            # tensorflow (since the graph hasn't been created yet)
            self.in_tf = False
            self._np_indices = indices
        else:
            self.in_tf = True
            self._indices = indices

        # dtype of base array
        self.key = key

        # shape
        self.display_shape = display_shape

        self.label = label

    @property
    def indices(self):
        if self.in_tf:
            return self._indices
        else:
            return self._np_indices

    @indices.setter
    def indices(self, val):
        raise BuildError("Indices are read only")

    @property
    def dtype(self):
        return self.key[0]

    @property
    def base_shape(self):
        return self.key[1]

    @property
    def shape(self):
        if self.display_shape is None:
            if self.in_tf:
                length = self._indices.get_shape().as_list()[0]
            else:
                length = self._np_indices.shape[0]
            return (length,) + self.base_shape[1:]
        else:
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

        if not self.in_tf or isinstance(indices, (int, slice)):
            new_indices = self.indices[indices]
        else:
            # need to handle advanced indexing differently for tensor indices
            new_indices = tf.gather(self.indices, indices)

        return TensorSignal(new_indices, self.key, label=self.label + ".slice")

    def broadcast(self, axis, length):
        assert self.in_tf
        assert axis in (0, 1)

        indices = self.indices
        indices = tf.stack([indices] * length, axis=axis)
        indices = tf.reshape(indices, (-1,))

        if axis == 1:
            display_shape = self.shape + (length,)
        else:
            display_shape = (length,) + self.shape

        return TensorSignal(
            indices, self.key, display_shape=display_shape,
            label=self.label + ".broadcast(%d, %d)" % (axis, length))

    def reshape(self, shape):
        # note: we could support this in tensorflow if we want, but it hasn't
        # been used so far
        assert not self.in_tf

        if np.prod(shape) != np.prod(self.shape):
            raise BuildError("Number of elements don't match in reshape")

        return TensorSignal(
            self.indices, self.key, display_shape=shape,
            label=self.label + ".reshape(%s)" % (shape,))

    def tile(self, length):
        assert self.in_tf

        # repeat along the first axis the given number of times
        # note: we don't use tf.tile because it doesn't have a GPU kernel
        indices = tf.concat(0, [self.indices] * length)
        shape = (self.shape[0] * length,) + self.shape[1:]

        return TensorSignal(
            indices, self.key, display_shape=shape,
            label=self.label + ".tile(%d)" % length)

    def to_tf(self):
        self._indices = tf.constant(self._np_indices)
        self.in_tf = True


class SignalDict(dict):
    """Map from Signal -> Tensor

    Takes care of view/base logic
    """

    def __init__(self, sig_map, bases, dtype, dt, *args, **kwargs):
        super(SignalDict, self).__init__(*args, **kwargs)
        self.bases = bases
        self.dtype = dtype
        self.sig_map = sig_map
        self.reads_by_base = defaultdict(list)

        # create this constant once here so we don't end up creating a new
        # constant in each operator
        self.dt = tf.constant(dt, dtype)
        self.dt.dt_val = dt  # store the actual value as well

    def __getitem__(self, sigs):
        if isinstance(sigs, (list, tuple)):
            sigs = self.combine(sigs)
        elif isinstance(sigs, Signal):
            sigs = self.sig_map[sigs]

        return self.gather(sigs)

    def __setitem__(self, sigs, val):
        if isinstance(sigs, (list, tuple)):
            sigs = self.combine(sigs)

        self.scatter(sigs, val, mode="update")

    def _key_and_indices(self, sigs):
        assert isinstance(sigs, (list, tuple))
        assert isinstance(sigs[0], (Signal, TensorSignal))

        sigs = [self.sig_map[s] if isinstance(s, Signal) else s for s in sigs]

        key = sigs[0].key
        assert all([s.key == key for s in sigs])

        indices = tf.concat(0, [s.indices for s in sigs])

        assert all([s.shape[1:] == sigs[0].shape[1:] for s in sigs])
        display_shape = (np.sum([s.shape[0]
                                 for s in sigs]),) + sigs[0].shape[1:]

        return key, indices, display_shape

    def scatter(self, dst, val, mode="update"):
        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError("Tensor detected with wrong dtype (%s), should "
                             "be %s." % (val.dtype.base_dtype, self.dtype))

        if isinstance(val, tf.IndexedSlices):
            # # source is already an indexed slice, so we just need to change
            # # the indices
            #
            # assert dst.indices.get_shape()[0] == val.indices.get_shape()[0]
            # # val.indices = dst.indices
            # val = tf.IndexedSlices(val.val, dst.indices)

            # TODO: remove this after I'm sure it's not being used anywhere
            raise NotImplementedError

        # undo any reshaping that has happened relative to the base array
        dst_shape = (val.get_shape().as_list()[0],) + dst.key[1][1:]
        if val.get_shape().ndims != len(dst_shape):
            val = tf.reshape(val, dst_shape)

        if self.bases[dst.key].dtype._is_ref_dtype:
            # for variables we can use the tensorflow sparse updates
            if mode == "update":
                scatter_f = tf.scatter_update
            elif mode == "inc":
                scatter_f = tf.scatter_add
        else:
            # for tensors we have to use our own version
            scatter_f = partial(self._scatter_f, mode=mode)

        if DEBUG:
            print("scatter")
            print("dst", dst)
            print("values", val)
            print("dst base", self.bases[dst.key])

        with tf.control_dependencies(self.reads_by_base[self.bases[dst.key]]):
            self.bases[dst.key] = scatter_f(self.bases[dst.key], dst.indices,
                                            val)

        if DEBUG:
            print("new dst base", self.bases[dst.key])

    def _scatter_f(self, dst, idxs, src, mode="update"):
        idxs = tf.expand_dims(idxs, 1)
        scatter_src = tf.scatter_nd(idxs, src, dst.get_shape())
        if mode == "update":
            # TODO: is there a more efficient way to do this?
            mask = tf.scatter_nd(idxs, tf.ones_like(src, dtype=tf.int32),
                                 dst.get_shape())
            return tf.where(mask > 0, scatter_src, dst)
        elif mode == "inc":
            return dst + scatter_src

    def gather(self, src):
        result = tf.gather(self.bases[src.key], src.indices)

        if src.shape is not None:
            result = tf.reshape(result, src.shape)

        # whenever we read from an array we use this to mark it as "read"
        # (so that any future writes to the array will be scheduled after
        # the read)
        self.reads_by_base[self.bases[src.key]] += [result]

        return result

    def combine(self, sigs):
        """Combines several TensorSignals into one."""

        if len(sigs) == 0:
            return []

        key, indices, shape = self._key_and_indices(sigs)

        return TensorSignal(indices, key, display_shape=shape)

    def __str__(self):
        """Pretty-print the signals and current values."""

        return "\n".join(["%s: %s" % (repr(k), repr(self[k]))
                          for k in self])
