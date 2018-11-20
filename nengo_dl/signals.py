"""
Represents and manages the internal simulation signals.
"""

from collections import defaultdict, OrderedDict, Mapping
import logging

from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class TensorSignal:
    """
    Represents a tensor as an indexed view into a base array.

    Parameters
    ----------
    indices : tuple or list or `~numpy.ndarray` of int
        Indices along the first axis of the base array corresponding to the
        data for this signal
    key : object
        Key mapping to the base array that contains the data for this signal
    dtype : `~numpy.dtype`
        dtype of the values represented by this signal
    shape : tuple of int
        View shape of this signal (may differ from shape of base array)
    minibatch_size : int
        If not None then this signal contains a minibatch dimension with the
        given size
    constant : callable
        A function that returns a TensorFlow constant (will be provided
        by `.signals.SignalDict.get_tensor_signal`)
    label : str
        Name for this signal, used to make debugging easier
    """

    def __init__(self, indices, key, dtype, shape, minibatch_size, constant,
                 label="TensorSignal"):
        # make indices read-only
        assert isinstance(indices, (tuple, list, np.ndarray))
        self._indices = np.asarray(indices)
        self._indices.flags.writeable = False
        self._tf_shape = None
        self._tf_indices = None
        self._tf_slice = -1

        self.key = key
        self.dtype = dtype
        self.shape = shape
        self.minibatch_size = minibatch_size
        self.constant = constant

        self.label = label

    @property
    def indices(self):
        """
        The indices containing the data for this signal in the base array.
        """
        return self._indices

    @indices.setter
    def indices(self, _):
        raise BuildError("Indices are read only")

    @property
    def ndim(self):
        """
        The rank of this signal.
        """
        return len(self.shape)

    def __repr__(self):
        return "TensorSignal(key=%s, shape=%s, label=%s)" % (
            self.key, self.shape, self.label)

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

        new_indices = self.indices[indices]
        return TensorSignal(
            new_indices, self.key, self.dtype,
            (len(new_indices),) + self.shape[1:], self.minibatch_size,
            self.constant, label=self.label + ".slice")

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
            self.indices, self.key, self.dtype, shape, self.minibatch_size,
            self.constant, label=self.label + ".reshape(%s)" % (shape,))

    def broadcast(self, axis, length):
        """
        Add a new dimension by broadcasting this signal along ``axis``
        for the given length.

        Parameters
        ----------
        axis : 0 or -1
            Where to insert the new dimension (currently only supports either
            the beginning or end of the array)
        length : int
            The number of times to duplicate signal along the broadcast
            dimension

        Returns
        -------
        sig : `.signals.TensorSignal`
            TensorSignal with new broadcasted shape
        """

        assert axis in (0, -1)
        # this only works on vectors
        assert self.ndim == 1 and not self.minibatched

        indices = self.indices
        indices = np.stack([indices] * length, axis=axis)
        indices = np.reshape(indices, (-1,))

        if axis == -1:
            display_shape = self.shape + (length,)
        else:
            display_shape = (length,) + self.shape

        return TensorSignal(
            indices, self.key, self.dtype, display_shape, self.minibatch_size,
            self.constant,
            label=self.label + ".broadcast(%d, %d)" % (axis, length))

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
            self._tf_indices = self.constant(self.indices, dtype=tf.int32)

        return self._tf_indices

    @property
    def tf_slice(self):
        """
        A tuple of ``tf.Tensors`` representing the ``(start, stop, stride)``
        slice within the base array containing the data for this signal.

        This can be used as a more efficient representation of
        `.TensorSignal.tf_indices`.
        """
        if self._tf_slice == -1:
            start = self.indices[0]
            stop = self.indices[-1] + 1
            step = (self.indices[1] - self.indices[0] if len(self.indices) > 1
                    else 1)
            if step != 0 and np.array_equal(self.indices,
                                            np.arange(start, stop, step)):
                self._tf_slice = (tf.constant([start]), tf.constant([stop]),
                                  tf.constant([step]))
            else:

                self._tf_slice = None

        return self._tf_slice

    @property
    def full_shape(self):
        """Shape of the signal including the minibatch dimension."""

        return (self.shape + (self.minibatch_size,) if self.minibatched else
                self.shape)

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
    dtype : ``tf.DType``
        Floating point precision used in signals
    minibatch_size : int
        Number of items in each minibatch
    """

    def __init__(self, dtype, minibatch_size):
        self.dtype = dtype
        self.minibatch_size = minibatch_size
        self.sig_map = {}
        self.bases = OrderedDict()  # will be filled in tensor_graph.build_loop
        self.reads_by_base = defaultdict(list)
        self.gather_bases = []
        self.internal_vars = OrderedDict()
        self.constant_phs = {}

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

        if val.dtype.is_floating and val.dtype.base_dtype != self.dtype:
            raise BuildError("Tensor detected with wrong dtype (%s), should "
                             "be %s." % (val.dtype.base_dtype, self.dtype))

        # align val shape with dst base shape
        self.bases[dst.key].get_shape().assert_is_fully_defined()
        val.get_shape().assert_is_fully_defined()
        dst_shape = ((dst.shape[0],) +
                     tuple(self.bases[dst.key].get_shape().as_list()[1:]))
        if val.get_shape() != dst_shape:
            val = tf.reshape(val, dst.tf_shape)

        logger.debug("scatter")
        logger.debug("values %s", val)
        logger.debug("dst %s", dst)
        logger.debug("indices %s", dst.indices)
        logger.debug("dst base %s", self.bases[dst.key])
        logger.debug("reads_by_base %s",
                     self.reads_by_base[self.bases[dst.key]])

        # make sure that any reads to the target signal happen before this
        # write (note: this is only any reads that have happened since the
        # last write, since each write changes the base array object)
        with tf.control_dependencies(self.reads_by_base[self.bases[dst.key]]):
            var = self.bases[dst.key]

            if (dst.tf_slice is not None and
                    var.get_shape().is_compatible_with(val.get_shape()) and
                    dst.indices[0] == 0 and
                    dst.indices[-1] == var.get_shape()[0].value - 1 and
                    len(dst.indices) == var.get_shape()[0]):
                if mode == "inc":
                    result = tf.assign_add(var, val, use_locking=False)
                    self.write_types["assign_add"] += 1
                else:
                    result = tf.assign(var, val, use_locking=False)
                    self.write_types["assign"] += 1
            elif mode == "inc":
                result = tf.scatter_add(var, dst.tf_indices, val,
                                        use_locking=False)
                self.write_types["scatter_add"] += 1
            else:
                result = tf.scatter_update(var, dst.tf_indices, val,
                                           use_locking=False)
                self.write_types["scatter_update"] += 1

            self.bases[dst.key] = result

        # update reads_by_base. the general workflow is
        # gather -> computation -> scatter
        # so when we get a scatter, we assume that that value indicates that
        # all the previous gathers are complete. so we block any writes to
        # those bases on the scatter value, to be sure that the
        # computation step is complete before the values can be overwritten
        for b in self.gather_bases:
            self.reads_by_base[b] += [self.bases[dst.key]]
        self.gather_bases = []

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
        logger.debug("indices %s", src.indices)
        logger.debug("src base %s", self.bases[src.key])

        var = self.bases[src.key]

        # we prefer to get the data via `strided_slice` or `identity` if
        # possible, as it is more efficient
        if force_copy or src.tf_slice is None:
            result = tf.gather(var, src.tf_indices)
            self.read_types["gather"] += 1
        elif (src.indices[0] == 0 and
              src.indices[-1] == var.get_shape()[0].value - 1 and
              len(src.indices) == var.get_shape()[0]):
            result = var
            self.read_types["identity"] += 1
        else:
            result = tf.strided_slice(var, *src.tf_slice)
            self.read_types["strided_slice"] += 1

        # reshape the data according to the shape set in `src`, if there is
        # one, otherwise keep the shape of the base array
        if result.get_shape() != src.full_shape:
            result = tf.reshape(result, src.tf_shape)

        # for some reason the shape inference doesn't work in some cases
        result.set_shape(src.full_shape)

        # whenever we read from an array we use this to mark it as "read"
        # (so that any future writes to the array will be scheduled after
        # the read)
        self.mark_gather(src)

        return result

    def mark_gather(self, src):
        """
        Marks ``src`` as being gathered, but doesn't actually perform a
        gather.  Used to indicate that some computation relies on ``src``.

        Parameters
        ----------
        src : `.TensorSignal`
            Signal indicating the data being read
        """

        self.gather_bases += [self.bases[src.key]]

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

        indices = np.concatenate([s.indices for s in sigs], axis=0)

        output = self.get_tensor_signal(indices, key, sigs[0].dtype, shape,
                                        sigs[0].minibatched, label=label)

        return output

    def make_internal(self, name, shape, minibatched=True):
        """
        Creates a variable to represent an internal simulation signal.

        This is to handle the case where we want to add a signal that is
        not represented as a `nengo.builder.Signal` in the Nengo op graph.

        Parameters
        ----------
        name : str
            Name for the signal/variable.
        shape : tuple of int
            Shape of the signal/variable.
        minibatched : bool
            Whether or not this signal contains a minibatch dimension.

        Returns
        -------
        sig : `.TensorSignal`
            A TensorSignal representing the newly created variable.
        """
        sig = self.get_tensor_signal(
            np.arange(shape[0]), object(), self.dtype, shape,
            minibatched, label=name)

        with tf.variable_scope(tf.get_default_graph().get_name_scope(),
                               reuse=False):
            var = tf.get_local_variable(
                name, shape=sig.full_shape, dtype=sig.dtype, trainable=False,
                initializer=tf.zeros_initializer())

        self.internal_vars[sig.key] = var

        return sig

    def get_tensor_signal(self, indices, key, dtype, shape, minibatched,
                          signal=None, label="TensorSignal"):
        """
        Creates a new ``TensorSignal`` with the given properties.

        This should be used rather than instantiating a new TensorSignal
        directly, as it handles some extra book-keeping (e.g., using the
        custom `.constant` function).

        Parameters
        ----------
        indices : tuple or list or `~numpy.ndarray` of int
            Indices along the first axis of the base array corresponding to the
            data for this signal
        key : object
            Key mapping to the base array that contains the data for this
            signal
        dtype : `~numpy.dtype`
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
            indices, key, dtype, shape,
            self.minibatch_size if minibatched else None,
            self.constant, label=label)

        if signal is not None:
            assert len(indices) == (1 if len(signal.shape) == 0 else
                                    signal.shape[0])
            assert signal.size == np.prod(shape)
            assert signal.minibatched == minibatched
            self[signal] = tensor_sig

        return tensor_sig

    def constant(self, value, dtype=None, cutoff=1 << 25):
        """
        Returns a constant Tensor containing the given value.

        The returned Tensor may be underpinned by a ``tf.constant`` op, or
        a ``tf.Variable`` that will be initialized to the constant value.  We
        use the latter in order to avoid storing large constant values in the
        TensorFlow GraphDef, which has a hard-coded limit of 2GB at the moment.

        Parameters
        ----------
        value : `~numpy.ndarray`
            Array containing the value of the constant
        dtype : ``tf.DType``
            The type for the constant (if ``None``, the dtype of ``value``
            will be used)
        cutoff : int
            The size of constant (in bytes) for which we will switch from
            ``tf.constant`` to ``tf.Variable``

        Returns
        -------
        constant : ``tf.Tensor``
            A tensor representing the given value
        """
        value = np.asarray(value)

        if dtype is None:
            dtype = value.dtype
        dtype = tf.as_dtype(dtype)

        if value.nbytes > cutoff:
            def make_ph(shape, dtype, **_):
                ph = tf.placeholder(dtype, shape)
                self.constant_phs[ph] = value
                return ph

            with tf.variable_scope("constant_vars", reuse=False):
                # tensorflow doesn't support int32 variables on the gpu, only
                # int64 (for some reason). we don't want to use int64 since
                # that would increase the size a lot, so we allow the variable
                # to be created on the CPU if necessary, and then move it to
                # the GPU with the identity
                # TODO: double check if this is still true in the future
                with tf.device(None):
                    const_var = tf.get_variable(
                        "constant_%d" % len(self.constant_phs),
                        initializer=make_ph, shape=value.shape, dtype=dtype,
                        collections=["constants"], trainable=False)

                return tf.identity(const_var)
        else:
            return tf.constant(value, dtype=dtype)

    def op_constant(self, ops, op_sizes, attr, dtype, ndims=2):
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
        dtype : ``tf.DType``
            Numeric type of the parameter
        ndims : int
            Empty dimensions will be added to the end of the returned tensor
            for all ndims > 1 (in the case that it is not a scalar).

        Returns
        -------
        constant : ``tf.Tensor``
            Tensor containing the values of ``attr`` for the given ops.  This
            will be a scalar if all the ops have the same parameter value, or
            an array giving the parameter value for each element in each op.
        """

        vals = [getattr(op, attr) for op in ops]
        if np.allclose(vals, vals[0]):
            return tf.constant(vals[0], dtype=dtype)

        assert len(op_sizes) == len(ops)
        v = np.zeros([sum(op_sizes)] + [1] * (ndims - 1),
                     dtype=dtype.as_numpy_dtype())
        k = 0
        for val, size in zip(vals, op_sizes):
            v[k:k + size] = val
            k += size
        return self.constant(v, dtype=dtype)

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
