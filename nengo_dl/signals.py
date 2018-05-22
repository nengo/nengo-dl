from collections import defaultdict, OrderedDict
import logging

from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class TensorSignal(object):
    """Represents a tensor as an indexed view into a base array.

    Parameters
    ----------
    indices : tuple or list or :class:`~numpy:numpy.ndarray` of int
        Indices along the first axis of the base array corresponding to the
        data for this signal
    key : object
        Key mapping to the base array that contains the data for this signal
    dtype : :class:`~numpy:numpy.dtype`
        dtype of the values represented by this signal
    shape : tuple of int
        View shape of this signal (may differ from shape of base array)
    minibatch_size : int
        If not None then this signal contains a minibatch dimension with the
        given size
    label : str, optional
        Name for this signal, used to make debugging easier
    """

    def __init__(self, indices, key, dtype, shape, minibatch_size,
                 label="TensorSignal"):
        # make indices read-only
        assert isinstance(indices, (tuple, list, np.ndarray))
        self._indices = np.asarray(indices)
        self._indices.flags.writeable = False
        self.tf_indices = None

        self.key = key
        self.dtype = dtype
        self.shape = shape
        self.minibatch_size = minibatch_size

        self.label = label

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, val):
        raise BuildError("Indices are read only")

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return "TensorSignal(key=%s, shape=%s, label=%s)" % (
            self.key, self.shape, self.label)

    def __getitem__(self, indices):
        """Create a new TensorSignal representing a subset (slice or advanced
        indexing) of the indices of this TensorSignal.

        Parameters
        ----------
        indices : slice or list of int
            The desired subset of the indices in this TensorSignal

        Returns
        -------
        :class:`.signals.TensorSignal`
            A new TensorSignal representing the subset of this TensorSignal
        """

        if indices is Ellipsis or indices is None:
            return self

        new_indices = self.indices[indices]
        return TensorSignal(
            new_indices, self.key, self.dtype,
            (len(new_indices),) + self.shape[1:], self.minibatch_size,
            label=self.label + ".slice")

    def reshape(self, shape):
        """Create a new TensorSignal representing a reshaped view of the
        same data in this TensorSignal (size of data must remain unchanged).

        Parameters
        ----------
        shape : tuple of int
            New shape for the signal (one dimension can be -1 to indicate
            an inferred dimension size, as in numpy)

        Returns
        -------
        :class:`.signals.TensorSignal`
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
            label=self.label + ".reshape(%s)" % (shape,))

    def broadcast(self, axis, length):
        """Add a new dimension by broadcasting this signal along ``axis``
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
        :class:`.signals.TensorSignal`
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
            label=self.label + ".broadcast(%d, %d)" % (axis, length))

    def load_indices(self, constant=None):
        """
        Loads the indices for this signal into TensorFlow.

        If the indices form a contiguous slice then also loads the
        start/stop/step of that slice.

        Parameters
        ----------
        constant : callable, optional
            A function that returns a TensorFlow constant (this is mainly
            used with :meth:`.SignalDict.constant`)
        """

        if constant is None:
            constant = tf.constant

        self.tf_indices = constant(self.indices, dtype=tf.int32)
        self.tf_shape = tf.constant(self.full_shape, dtype=tf.int32)

        start = self.indices[0]
        stop = self.indices[-1] + 1
        step = (self.indices[1] - self.indices[0] if len(self.indices) > 1
                else 1)
        if step != 0 and np.array_equal(self.indices,
                                        np.arange(start, stop, step)):
            self.as_slice = (tf.constant([start]), tf.constant([stop]),
                             tf.constant([step]))
        else:
            self.as_slice = None

    @property
    def full_shape(self):
        """Shape including the minibatch dimension."""

        return (self.shape + (self.minibatch_size,) if self.minibatched else
                self.shape)

    @property
    def minibatched(self):
        """Whether or not this TensorSignal contains a minibatch dimension."""

        return self.minibatch_size is not None


class SignalDict(object):
    """Handles the mapping from :class:`~nengo:nengo.builder.Signal`
    to ``tf.Tensor``.

    Takes care of gather/scatter logic to read/write signals within the base
    arrays.

    Parameters
    ----------
    sig_map : dict of {:class:`~nengo:nengo.builder.Signal`: \
                       :class:`.TensorSignal`}
        Mapping from ``nengo`` signals to ``nengo_dl`` signals
    dtype : ``tf.DType``
        Floating point precision used in signals
    minibatch_size : int
        Number of items in each minibatch
    """

    def __init__(self, sig_map, dtype, minibatch_size):
        self.dtype = dtype
        self.sig_map = sig_map
        self.minibatch_size = minibatch_size
        self.bases = None
        self.reads_by_base = defaultdict(list)
        self.gather_bases = []
        self.internal_vars = OrderedDict()
        self.constant_phs = {}

        # logging
        self.read_types = defaultdict(int)
        self.write_types = defaultdict(int)

    def scatter(self, dst, val, mode="update"):
        """Updates the base data corresponding to ``dst``.

        Parameters
        ----------
        dst : :class:`.TensorSignal`
            Signal indicating the data to be modified in base array
        val : ``tf.Tensor``
            Update data (same shape as ``dst``, i.e. a dense array <= the size
            of the base array)
        mode : "update" or "inc"
            Overwrite/add the data at ``dst`` with ``val``
        """

        if dst.tf_indices is None:
            raise BuildError("Indices for %s have not been loaded into "
                             "TensorFlow" % dst)
        # if not dst.minibatched:
        #     raise BuildError("Assigning to a trainable variable")
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
            self.bases[dst.key] = self._scatter_f_var(dst, val, mode=mode)

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

    def _scatter_f_var(self, dst, src, mode="update"):
        # create a temporary variable for dst so that we can use the sparse
        # variable updates. despite this looking incredibly inefficient, it is
        # actually faster than the scatter_nd approach
        # from tensorflow.python.ops import gen_state_ops
        # var = gen_state_ops._temporary_variable(
        #     self.bases[dst.key].get_shape(), self.bases[dst.key].dtype)
        # var_name = var.op.name
        # var = tf.assign(var, self.bases[dst.key])

        var = self.bases[dst.key]

        if (dst.as_slice is not None and
                var.get_shape().is_compatible_with(src.get_shape()) and
                dst.indices[0] == 0 and
                dst.indices[-1] == var.get_shape()[0].value - 1 and
                len(dst.indices) == var.get_shape()[0]):
            if mode == "inc":
                result = tf.assign_add(var, src, use_locking=False)
                self.write_types["assign_add"] += 1
            else:
                result = tf.assign(var, src, use_locking=False)
                self.write_types["assign"] += 1
        elif mode == "inc":
            result = tf.scatter_add(var, dst.tf_indices, src,
                                    use_locking=False)
            self.write_types["scatter_add"] += 1
        else:
            result = tf.scatter_update(var, dst.tf_indices, src,
                                       use_locking=False)
            self.write_types["scatter_update"] += 1

        # result = gen_state_ops._destroy_temporary_variable(var, var_name)

        return result

    def gather(self, src, force_copy=False):
        """Fetches the data corresponding to ``src`` from the base array.

        Parameters
        ----------
        src : :class:`.TensorSignal`
            Signal indicating the data to be read from base array
        force_copy : bool, optional
            If True, always perform a gather, not a slice (this forces a
            copy). Note that setting ``force_copy=False`` does not guarantee
            that a copy won't be performed.

        Returns
        -------
        ``tf.Tensor``
            Tensor object corresponding to a dense subset of data from the
            base array
        """

        if src.tf_indices is None:
            raise BuildError("Indices for %s have not been loaded into "
                             "TensorFlow" % src)

        logger.debug("gather")
        logger.debug("src %s", src)
        logger.debug("indices %s", src.indices)
        logger.debug("src base %s", self.bases[src.key])

        var = self.bases[src.key]

        # we prefer to get the data via `strided_slice` or `identity` if
        # possible, as it is more efficient
        if force_copy or src.as_slice is None:
            result = tf.gather(var, src.tf_indices)
            self.read_types["gather"] += 1
        elif (src.indices[0] == 0 and
              src.indices[-1] == var.get_shape()[0].value - 1 and
              len(src.indices) == var.get_shape()[0]):
            result = var
            self.read_types["identity"] += 1
        else:
            result = tf.strided_slice(var, *src.as_slice)
            self.read_types["strided_slice"] += 1

        # for some reason the shape inference doesn't work in some cases
        result.set_shape(src.tf_indices.get_shape()[:1].concatenate(
            var.get_shape()[1:]))

        # reshape the data according to the shape set in `src`, if there is
        # one, otherwise keep the shape of the base array
        if result.get_shape() != src.full_shape:
            result = tf.reshape(result, src.tf_shape)
            result.set_shape(src.full_shape)

        # whenever we read from an array we use this to mark it as "read"
        # (so that any future writes to the array will be scheduled after
        # the read)
        self.mark_gather(src)

        return result

    def mark_gather(self, src):
        """Marks ``src`` as being gathered, but doesn't actually perform a
        gather.  Used to indicate that some computation relies on ``src``.

        Parameters
        ----------
        src : :class:`.TensorSignal`
            Signal indicating the data being read
        """

        self.gather_bases += [self.bases[src.key]]

    def combine(self, sigs, load_indices=True, label="Combine"):
        """Combines several TensorSignals into one by concatenating along
        the first axis.

        Parameters
        ----------
        sigs : list of :class:`.TensorSignal` or \
                       :class:`~nengo:nengo.builder.Signal`
            Signals to be combined
        load_indices : bool, optional
            If True, load the indices for the new signal into TensorFlow right
            away (otherwise they will need to be manually loaded later)
        label : str, optional
            Name for combined signal (to help with debugging)

        Returns
        -------
        :class:`.TensorSignal`
            New TensorSignal representing the concatenation of the data in
            ``sigs``
        """

        if len(sigs) == 0:
            return []

        assert isinstance(sigs, (list, tuple))
        assert isinstance(sigs[0], (Signal, TensorSignal))

        sigs = [self.sig_map[s] if isinstance(s, Signal) else s for s in sigs]

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

        output = TensorSignal(indices, key, sigs[0].dtype, shape,
                              sigs[0].minibatch_size, label=label)

        if load_indices:
            output.load_indices(constant=self.constant)

        return output

    def make_internal(self, name, shape, minibatched=True):
        sig = TensorSignal(
            np.arange(shape[0]), object(), self.dtype, shape,
            self.minibatch_size if minibatched else None, label=name)
        sig.load_indices(constant=self.constant)

        with tf.variable_scope(tf.get_default_graph().get_name_scope(),
                               reuse=False):
            var = tf.get_local_variable(
                name, shape=sig.full_shape, dtype=sig.dtype, trainable=False,
                initializer=tf.zeros_initializer())

        self.internal_vars[sig.key] = var

        return sig

    def constant(self, value, dtype=None, cutoff=1 << 25):
        """
        Returns a constant Tensor containing the given value.

        The returned Tensor may be underpinned by a ``tf.constant`` op, or
        a ``tf.Variable`` that will be initialized to the constant value.  We
        use the latter in order to avoid storing large constant values in the
        TensorFlow GraphDef, which has a hard-coded limit of 2GB at the moment.

        Parameters
        ----------
        value : :class:`~numpy:numpy.ndarray`
            Array containing the value of the constant
        dtype : ``tf.DType``, optional
            The type for the constant (if ``None``, the dtype of ``value``
            will be used)
        cutoff : int, optional
            The size of constant (in bytes) for which we will switch from
            ``tf.constant`` to ``tf.Variable``

        Returns
        -------
        ``tf.Tensor``
            A tensor representing the given value
        """
        value = np.asarray(value)

        if dtype is None:
            dtype = value.dtype
        dtype = tf.as_dtype(dtype)

        if value.nbytes > cutoff:
            ph = tf.placeholder(dtype, value.shape)
            self.constant_phs[ph] = value

            with tf.variable_scope("constant_vars", reuse=False):
                # tensorflow doesn't support int32 variables on the gpu, only
                # int64 (for some reason). we don't want to use int64 since
                # that would increase the size a lot, so we allow the variable
                # to be created on the CPU if necessary, and then move it to
                # the GPU with the identity
                with tf.device(None):
                    const_var = tf.get_variable(
                        "constant_%d" % len(self.constant_phs), initializer=ph,
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
        ``tf.Tensor``
            Tensor containing the values of ``attr`` for the given ops.  This
            will be a scalar if all the ops have the same parameter value, or
            an array giving the parameter value for each element in each op.
        """

        val0 = getattr(ops[0], attr)
        if np.allclose([getattr(op, attr) for op in ops], val0):
            return tf.constant(val0, dtype=dtype)
        else:
            return self.constant(
                [np.reshape(getattr(op, attr), [1] * (ndims - 1))
                 for i, op in enumerate(ops) for _ in range(op_sizes[i])],
                dtype=dtype)
