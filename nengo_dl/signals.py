"""
Represents and manages the internal simulation signals.
"""

import logging
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import tensorflow as tf
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError

logger = logging.getLogger(__name__)


class TensorSignal:
    """
    Represents a tensor as an indexed view into a base array.

    Parameters
    ----------
    slices : tuple of tuple of int
        Start/stop indices of slices along the first axis of the base array,
        corresponding to the data for this signal.
    key : object
        Key mapping to the base array that contains the data for this signal.
    dtype : str
        dtype of the values represented by this signal.
    shape : tuple of int
        View shape of this signal (may differ from shape of base array).
    minibatch_size : int
        If not None then this signal contains a minibatch dimension with the
        given size.
    label : str
        Name for this signal, used to make debugging easier.
    """

    def __init__(self, slices, key, dtype, shape, minibatch_size, label="TensorSignal"):
        # make sure slices are read-only
        slices = tuple(tuple(s) for s in slices)

        self._slices = slices
        self.key = key
        self.dtype = dtype
        self.shape = shape
        self.minibatch_size = minibatch_size
        self.label = label

        self.reset()

    def reset(self):
        """
        Reset cached Tensors.
        """
        self._tf_shape = None
        self._tf_indices = None
        self._tf_indices_nd = None
        self._tf_slice = -1

    @property
    def slices(self):
        """
        The slices containing the data for this signal in the base array.
        """
        return self._slices

    @slices.setter
    def slices(self, _):
        raise BuildError("Slices are read only")

    @property
    def ndim(self):
        """
        The rank of this signal.
        """
        return len(self.shape)

    def __repr__(self):
        return f"TensorSignal(key={self.key}, shape={self.shape}, label={self.label})"

    def __getitem__(self, indices):
        """
        Create a new TensorSignal representing a subset (slice or advanced
        indexing) of the indices of this TensorSignal.

        Parameters
        ----------
        indices : slice or list of int
            The desired subset of the indices in this TensorSignal

        Returns
        -------
        sig : `.signals.TensorSignal`
            A new TensorSignal representing the subset of this TensorSignal
        """

        if indices is Ellipsis or indices is None:
            return self

        # figure out indices in new view
        source_indices = np.concatenate(
            [np.arange(start, stop, dtype=np.int32) for start, stop in self.slices]
        )
        new_indices = source_indices[indices]

        # find all the entries that are not runs (not consecutive with the
        # previous entry)
        run_starts = np.empty(new_indices.shape[0], dtype=bool)
        run_starts[0] = True
        np.not_equal(new_indices[:-1] + 1, new_indices[1:], out=run_starts[1:])

        # find run start/stop indices
        run_breaks = np.nonzero(run_starts)[0]
        starts = new_indices[run_breaks]
        stops = np.append(new_indices[run_breaks - 1][1:], new_indices[-1]) + 1

        slices = tuple(zip(starts, stops))

        return TensorSignal(
            slices,
            self.key,
            self.dtype,
            (len(new_indices),) + self.shape[1:],
            self.minibatch_size,
            label=self.label + ".slice",
        )

    def reshape(self, shape):
        """
        Create a new TensorSignal representing a reshaped view of the
        same data in this TensorSignal (size of data must remain unchanged).

        Parameters
        ----------
        shape : tuple of int
            New shape for the signal (one dimension can be -1 to indicate
            an inferred dimension size, as in numpy)

        Returns
        -------
        sig : `.signals.TensorSignal`
            New TensorSignal representing the same data as this signal but
            with the given shape
        """

        # replace -1 with inferred dimension
        if shape.count(-1) > 1:
            raise BuildError("Only one inferred dimension allowed in reshape")
        elif shape.count(-1) == 1:
            n_elem = np.prod(self.shape)
            n_shape = int(np.prod([x for x in shape if x != -1]))
            if n_elem % n_shape != 0:
                raise BuildError("No valid length for inferred dimension")

            shape = tuple(x if x != -1 else n_elem // n_shape for x in shape)
        else:
            if np.prod(shape) != np.prod(self.shape):
                raise BuildError("Number of elements don't match in reshape")

        return TensorSignal(
            self.slices,
            self.key,
            self.dtype,
            shape,
            self.minibatch_size,
            label=self.label + f".reshape({shape})",
        )

    @property
    def tf_shape(self):
        """
        A ``tf.Tensor`` representing the shape of this signal.
        """
        if self._tf_shape is None:
            self._tf_shape = tf.constant(self.full_shape, dtype=tf.int32)

        return self._tf_shape

    @property
    def tf_indices(self):
        """
        A ``tf.Tensor`` representing the indices of this signal.
        """
        if self._tf_indices is None:
            self._tf_indices = tf.constant(
                np.concatenate(
                    [
                        np.arange(start, stop, dtype=np.int32)
                        for start, stop in self.slices
                    ],
                    axis=0,
                ),
                dtype=tf.int32,
            )

        return self._tf_indices

    @property
    def tf_indices_nd(self):
        """
        A ``tf.Tensor`` representing the indices of this signal for use with e.g.
        ``scatter_nd``.
        """

        if self._tf_indices_nd is None:
            if self.minibatched:
                self._tf_indices_nd = tf.stack(
                    tf.meshgrid(
                        tf.range(self.minibatch_size, dtype=tf.int32),
                        self.tf_indices,
                        indexing="ij",
                    ),
                    axis=-1,
                )
            else:
                self._tf_indices_nd = tf.expand_dims(self.tf_indices, -1)

        return self._tf_indices_nd

    @property
    def tf_slice(self):
        """
        A tuple of ``tf.Tensors`` representing the ``(start, stop, stride)``
        slice within the base array containing the data for this signal.

        This can be used as a more efficient representation of
        `.TensorSignal.tf_indices`.
        """
        if self._tf_slice == -1:
            if len(self.slices) == 1:
                start, stop = self.slices[0]
                if self.minibatched:
                    # add full slice along first (batch) dimension
                    start = [0, start]
                    stop = [self.minibatch_size, stop]
                else:
                    start = [start]
                    stop = [stop]

                self._tf_slice = (
                    tf.constant(start),
                    tf.constant(stop),
                )
            else:
                self._tf_slice = None

        return self._tf_slice

    @property
    def full_shape(self):
        """Shape of the signal including the minibatch dimension."""

        return ((self.minibatch_size,) + self.shape) if self.minibatched else self.shape

    @property
    def minibatched(self):
        """Whether or not this TensorSignal contains a minibatch dimension."""

        return self.minibatch_size is not None


class SignalDict(Mapping):
    """
    Handles the mapping from `~nengo.builder.Signal` to ``tf.Tensor``.

    Takes care of gather/scatter logic to read/write signals within the base
    arrays.

    Parameters
    ----------
    dtype : str
        Floating point precision used in signals (e.g. "float32")
    minibatch_size : int
        Number of items in each minibatch
    """

    def __init__(self, dtype, minibatch_size):
        self.dtype = tf.as_dtype(dtype)
        self.minibatch_size = minibatch_size
        self.sig_map = {}

        self.reset()

    def reset(self):
        """
        Reset build-specific data structures.

        These are data structures that are filled out during the TensorGraph build
        process (and therefore need to be re-initialized if we build the model again),
        as opposed to data that is constant for a given Nengo model.
        """
        # these values will be re-generated whenever the model is rebuilt
        self.bases = {}

        # reset TensorSignals
        for sig in self.sig_map.values():
            sig.reset()

        # logging
        self.read_types = defaultdict(int)
        self.write_types = defaultdict(int)

    def scatter(self, dst, val, mode="update"):
        """
        Updates the base data corresponding to ``dst``.

        Parameters
        ----------
        dst : `.TensorSignal`
            Signal indicating the data to be modified in base array
        val : ``tf.Tensor``
            Update data (same shape as ``dst``, i.e. a dense array <= the size
            of the base array)
        mode : "update" or "inc"
            Overwrite/add the data at ``dst`` with ``val``
        """

        logger.debug("scatter")
        logger.debug("values %s", val)
        logger.debug("dst %s", dst)
        logger.debug("slices %s", dst.slices)
        logger.debug(
            "dst base %s", self.bases[dst.key] if dst.key in self.bases else None
        )

        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError(
                f"Tensor detected with wrong dtype ({val.dtype.base_dtype}), should "
                f"be {self.dtype}."
            )

        # should never be writing to a variable
        if isinstance(self.bases[dst.key], tf.Variable):
            raise BuildError("Scatter target should not be a Variable")

        if isinstance(self.bases[dst.key], tuple):
            # this is the first set operation for this signal
            assert mode == "update"

            base_shape = self.bases[dst.key]
            var = None
        else:
            self.bases[dst.key].shape.assert_is_fully_defined()
            base_shape = self.bases[dst.key].shape
            var = self.bases[dst.key]

        # align val shape with dst base shape
        val.shape.assert_is_fully_defined()
        dst_shape = list(base_shape)
        dst_shape[dst.minibatched] = dst.shape[0]
        if val.shape != dst_shape:
            val = tf.reshape(val, dst.tf_shape)

        if len(dst.slices) == 1 and val.shape == base_shape:
            if mode == "inc":
                result = var + val
                self.write_types["assign_add"] += 1
            else:
                result = val
                self.write_types["assign"] += 1
        elif mode == "inc":
            result = tf.tensor_scatter_nd_add(var, dst.tf_indices_nd, val)
            self.write_types["scatter_add"] += 1
        else:
            if var is None:
                result = tf.scatter_nd(dst.tf_indices_nd, val, shape=base_shape)
            else:
                result = tf.tensor_scatter_nd_update(var, dst.tf_indices_nd, val)
            self.write_types["scatter_update"] += 1

        self.bases[dst.key] = result

        logger.debug("new dst base %s", self.bases[dst.key])

    def gather(self, src, force_copy=False):
        """
        Fetches the data corresponding to ``src`` from the base array.

        Parameters
        ----------
        src : `.TensorSignal`
            Signal indicating the data to be read from base array
        force_copy : bool
            If True, always perform a gather, not a slice (this forces a
            copy). Note that setting ``force_copy=False`` does not guarantee
            that a copy won't be performed.

        Returns
        -------
        gathered : ``tf.Tensor``
            Tensor object corresponding to a dense subset of data from the
            base array
        """

        logger.debug("gather")
        logger.debug("src %s", src)
        logger.debug("slices %s", src.slices)
        logger.debug("src base %s", self.bases[src.key])

        var = self.bases[src.key]

        assert isinstance(var, tf.Tensor)

        # we prefer to get the data via `strided_slice` or `identity` if
        # possible, as it is more efficient
        if force_copy or len(src.slices) > 1:
            result = tf.gather(var, src.tf_indices, axis=1 if src.minibatched else 0)
            self.read_types["gather"] += 1
        elif src.slices[0][0] == 0 and src.slices[0][1] == var.shape[src.minibatched]:
            result = var
            self.read_types["identity"] += 1
        else:
            result = tf.strided_slice(var, *src.tf_slice)
            self.read_types["strided_slice"] += 1

        # reshape the data according to the shape set in `src`, if there is
        # one, otherwise keep the shape of the base array
        if result.shape != src.full_shape:
            result = tf.reshape(result, src.tf_shape)

        return result

    def combine(self, sigs, label="Combine"):
        """
        Combines several TensorSignals into one by concatenating along
        the first axis.

        Parameters
        ----------
        sigs : list of `.TensorSignal` or `~nengo.builder.Signal`
            Signals to be combined
        label : str
            Name for combined signal (to help with debugging)

        Returns
        -------
        sig : `.TensorSignal`
            New TensorSignal representing the concatenation of the data in
            ``sigs``
        """

        if len(sigs) == 0:
            return []

        assert isinstance(sigs, (list, tuple))
        assert isinstance(sigs[0], (Signal, TensorSignal))

        sigs = [self[s] if isinstance(s, Signal) else s for s in sigs]

        # make sure all the signals have the same base
        # note: this also tells us that they have the same dtype and
        # minibatching
        key = sigs[0].key
        assert all(s.key == key for s in sigs)

        # make sure all signals have the same shape (except first axis,
        # which we're concatenating along); note, this can fail even if they
        # all have the same base, due to reshaping
        shape = (np.sum([s.shape[0] for s in sigs]),) + sigs[0].shape[1:]
        assert all(s.shape[1:] == shape[1:] for s in sigs)

        # combine slices from signals (possibly merging consecutive slices)
        combined_slices = []
        for sig in sigs:
            if len(combined_slices) > 0 and combined_slices[-1][1] == sig.slices[0][0]:
                combined_slices = combined_slices[:-1] + [
                    (combined_slices[-1][0], sig.slices[0][1])
                ]
                combined_slices.extend(sig.slices[1:])
            else:
                combined_slices.extend(sig.slices)

        output = self.get_tensor_signal(
            combined_slices,
            key,
            sigs[0].dtype,
            shape,
            sigs[0].minibatched,
            label=label,
        )

        return output

    def get_tensor_signal(
        self, slices, key, dtype, shape, minibatched, signal=None, label="TensorSignal"
    ):
        """
        Creates a new ``TensorSignal`` with the given properties.

        This should be used rather than instantiating a new TensorSignal
        directly, as it handles some extra book-keeping.

        Parameters
        ----------
        slices : tuple of tuple of int
            Start/stop indices of slices along the first axis of the base array,
            corresponding to the data for this signal.
        key : object
            Key mapping to the base array that contains the data for this
            signal
        dtype : str
            dtype of the values represented by this signal
        shape : tuple of int
            View shape of this signal (may differ from shape of base array)
        minibatched : bool
            Whether or not this signal contains a minibatch dimension
        signal : `~nengo.builder.Signal`
            If not None, associate the new ``TensorSignal`` with the given
            ``Signal`` in the ``sig_map``
        label : str
            Name for this signal, used to make debugging easier

        Returns
        -------
        sig : `.TensorSignal`
            A new ``TensorSignal`` with the given properties
        """

        tensor_sig = TensorSignal(
            slices,
            key,
            dtype,
            shape,
            self.minibatch_size if minibatched else None,
            label=label,
        )

        if signal is not None:
            if signal.sparse:
                assert sum(stop - start for start, stop in slices) == signal.size
                assert shape == (signal.size,)
            else:
                assert sum(stop - start for start, stop in slices) == (
                    1 if len(signal.shape) == 0 else signal.shape[0]
                )
                assert signal.size == np.prod(shape)
            assert signal.minibatched == minibatched
            self[signal] = tensor_sig

        return tensor_sig

    def op_constant(self, ops, op_sizes, attr, dtype, shape=(1, -1)):
        """
        Creates a tensor representing the constant parameters of an op group.

        Parameters
        ----------
        ops : list of object
            The operators for some merged group of ops
        op_sizes : list of int
            The number of constant elements in each op
        attr : str
            The attribute of the op that describes the constant parameter
        dtype : str
            Numeric type of the parameter
        shape : tuple of int
            Shape for returned constant (this will be ignored in the scalar case).
            The default adds an empty dimension for broadcasting along the batch axis.

        Returns
        -------
        constant : ``tf.Tensor``
            Tensor containing the values of ``attr`` for the given ops.  This
            will be a scalar if all the ops have the same parameter value, or
            an array giving the parameter value for each element in each op.
        """

        if not isinstance(dtype, tf.DType):
            dtype = tf.as_dtype(dtype)

        vals = [getattr(op, attr) for op in ops]
        if np.allclose(vals, vals[0]):
            return tf.constant(vals[0], dtype=dtype)

        assert len(op_sizes) == len(ops)
        v = np.zeros(sum(op_sizes), dtype=dtype.as_numpy_dtype)
        k = 0
        for val, size in zip(vals, op_sizes):
            v[k : k + size] = val
            k += size

        if shape is not None:
            v = np.reshape(v, shape)

        return tf.constant(v, dtype=dtype)

    def __getitem__(self, sig):
        return self.sig_map[sig]

    def __setitem__(self, sig, tensor_sig):
        self.sig_map[sig] = tensor_sig

    def __len__(self):
        return len(self.sig_map)

    def __iter__(self):
        return iter(self.sig_map)

    def __contains__(self, sig):
        return sig in self.sig_map
