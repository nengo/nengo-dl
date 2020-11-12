# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf
from nengo.builder.neurons import SimNeurons
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from nengo.neurons import LIF

from nengo_dl.signals import SignalDict, TensorSignal


def test_tensor_signal_basic():
    # check that indices are read-only
    sig = TensorSignal([(0, 2)], None, None, None, None)
    with pytest.raises(BuildError, match="Slices are read only"):
        sig.slices = ((2, 4),)
    with pytest.raises(TypeError, match="does not support item assignment"):
        sig.slices[0] = (2, 4)
    with pytest.raises(TypeError, match="does not support item assignment"):
        sig.slices[0][0] = 2

    # check ndim
    sig = TensorSignal([(0, 2)], None, None, (1, 2), 1)
    assert sig.ndim == 2


def test_tensor_signal_getitem():
    sig = TensorSignal([(1, 6)], object(), None, (4, 3), None)
    sig_slice = sig[:2]
    assert sig_slice.slices == ((1, 3),)
    assert sig_slice.key == sig.key
    assert sig_slice.shape == (2, 3)

    assert sig[...] is sig

    sig_adv = sig[[0, 1, 3, 4]]
    assert sig_adv.slices == ((1, 3), (4, 6))
    assert sig_adv.shape == (4, 3)


def test_tensor_signal_reshape():
    sig = TensorSignal([(1, 5)], object(), None, (4, 3), None)

    with pytest.raises(BuildError):
        sig.reshape((100,))

    sig_reshape = sig.reshape((6, 2))
    assert sig_reshape.slices == sig.slices
    assert sig_reshape.key == sig.key
    assert sig_reshape.shape == (6, 2)

    sig_reshape = sig.reshape((6, 2, 1))
    assert sig_reshape.slices == sig.slices
    assert sig_reshape.key == sig.key
    assert sig_reshape.shape == (6, 2, 1)

    sig_reshape = sig.reshape((-1, 2))
    assert sig_reshape.shape == (6, 2)

    sig_reshape = sig.reshape((-1,))
    assert sig_reshape.shape == (12,)

    with pytest.raises(BuildError):
        sig.reshape((-1, 5))

    with pytest.raises(BuildError):
        sig.reshape((-1, -1))

    with pytest.raises(BuildError):
        sig.reshape((4, 4))


@pytest.mark.eager_only
def test_tensor_signal_load_indices():
    sig = TensorSignal([(2, 6)], object(), None, (4,), None)
    assert np.all(sig.tf_indices == np.arange(*sig.slices[0]))
    start, stop = sig.tf_slice
    assert start == 2
    assert stop == 6

    sig = TensorSignal(((2, 3), (2, 3), (3, 4), (3, 4)), object(), None, (4,), None)
    assert np.all(sig.tf_indices == [2, 2, 3, 3])
    assert sig.tf_slice is None


@pytest.mark.eager_only
@pytest.mark.parametrize("minibatched", (True, False))
def test_signal_dict_scatter(minibatched):
    minibatch_size = 2
    var_size = 19
    signals = SignalDict(tf.float32, minibatch_size)

    key = object()
    var_key = object()
    val = np.random.random(
        (minibatch_size, var_size) if minibatched else (var_size,)
    ).astype(np.float32)
    update_shape = (minibatch_size, 4) if minibatched else (4,)
    pre_slice = np.index_exp[:, :4] if minibatched else np.index_exp[:4]
    post_slice = np.index_exp[:, 4:] if minibatched else np.index_exp[4:]

    signals.bases = {key: tf.constant(val), var_key: tf.Variable(val)}

    x = signals.get_tensor_signal([(0, 4)], key, tf.float32, (4,), minibatched)
    with pytest.raises(BuildError, match="wrong dtype"):
        signals.scatter(x, tf.ones(update_shape, dtype=tf.float64))

    x_var = signals.get_tensor_signal([(0, 4)], var_key, tf.float32, (4,), minibatched)
    with pytest.raises(BuildError, match="should not be a Variable"):
        signals.scatter(x_var, tf.ones(update_shape))

    # update
    signals.scatter(x, tf.ones(update_shape))
    y = signals.bases[key]
    assert np.allclose(y[pre_slice], 1)
    assert np.allclose(y[post_slice], val[post_slice])
    assert signals.write_types["scatter_update"] == 1

    # increment, and reshaping val
    signals.scatter(
        x, tf.ones((minibatch_size, 2, 2) if minibatched else (2, 2)), mode="inc"
    )
    y = signals.bases[key]
    assert np.allclose(y[pre_slice], 2)
    assert np.allclose(y[post_slice], val[post_slice])
    assert signals.write_types["scatter_add"] == 1

    # recognize assignment to full array
    x = signals.get_tensor_signal(
        [(0, var_size)], key, tf.float32, (var_size,), minibatched
    )
    y = tf.ones((minibatch_size, var_size) if minibatched else (var_size,))
    signals.scatter(x, y)
    assert signals.bases[key] is y
    assert signals.write_types["assign"] == 1


@pytest.mark.eager_only
@pytest.mark.parametrize("minibatched", (True, False))
def test_signal_dict_gather(minibatched):
    minibatch_size = 3
    var_size = 19
    signals = SignalDict(tf.float32, minibatch_size)

    key = object()
    val = np.random.random(
        (minibatch_size, var_size) if minibatched else (var_size,)
    ).astype(np.float32)
    gathered_val = val[:, :4] if minibatched else val[:4]
    signals.bases = {key: tf.constant(val, dtype=tf.float32)}

    x = signals.get_tensor_signal([(0, 4)], key, tf.float32, (4,), minibatched)

    # sliced read
    assert np.allclose(signals.gather(x), gathered_val)
    assert signals.read_types["strided_slice"] == 1

    # read with reshape
    x = signals.get_tensor_signal([(0, 4)], key, tf.float32, (2, 2), minibatched)
    y = signals.gather(x)
    shape = (minibatch_size, 2, 2) if minibatched else (2, 2)
    assert y.shape == shape
    assert np.allclose(y, gathered_val.reshape(shape))
    assert signals.read_types["strided_slice"] == 2

    # gather read
    x = signals.get_tensor_signal([(0, 4)], key, tf.float32, (4,), minibatched)
    y = signals.gather(x, force_copy=True)
    assert signals.read_types["gather"] == 1

    x = signals.get_tensor_signal(
        ((0, 1), (0, 1), (3, 4), (3, 4)), key, tf.float32, (4,), minibatched
    )
    assert np.allclose(
        signals.gather(x), val[:, [0, 0, 3, 3]] if minibatched else val[[0, 0, 3, 3]]
    )
    assert signals.read_types["gather"] == 2

    # reading from full array
    x = signals.get_tensor_signal(
        [(0, var_size)], key, tf.float32, (var_size,), minibatched
    )
    y = signals.gather(x)
    assert y is signals.bases[key]
    assert signals.read_types["identity"] == 1

    # reading from strided full array
    x = signals.get_tensor_signal(
        tuple((i * 2, i * 2 + 1) for i in range(var_size // 2 + 1)),
        key,
        tf.float32,
        (var_size // 2 + 1,),
        minibatched,
    )
    y = signals.gather(x)
    assert y is not signals.bases[key]
    assert signals.read_types["gather"] == 3


def test_signal_dict_combine():
    minibatch_size = 1
    signals = SignalDict(tf.float32, minibatch_size)

    key = object()

    assert signals.combine([]) == []

    y = signals.combine(
        [
            signals.get_tensor_signal([(0, 3)], key, None, (3, 2), False),
            signals.get_tensor_signal([(4, 7)], key, None, (3, 2), False),
        ]
    )
    assert y.key is key
    assert y._tf_indices is None
    assert y.shape == (6, 2)
    assert y.slices == ((0, 3), (4, 7))

    y = signals.combine(
        [
            signals.get_tensor_signal([(0, 3)], key, None, (3, 2), False),
            signals.get_tensor_signal([(3, 6)], key, None, (3, 2), False),
        ]
    )
    assert y.key is key
    assert y._tf_indices is None
    assert y.shape == (6, 2)
    assert y.slices == ((0, 6),)


@pytest.mark.eager_only
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("diff", (True, False))
def test_op_constant(dtype, diff):
    ops = (
        SimNeurons(LIF(tau_rc=1), Signal(np.zeros(10)), None),
        SimNeurons(LIF(tau_rc=2 if diff else 1), Signal(np.zeros(10)), None),
    )

    signals = SignalDict(tf.float32, 1)
    const = signals.op_constant(
        [op.neurons for op in ops], [op.J.shape[0] for op in ops], "tau_rc", dtype
    )
    const1 = signals.op_constant(
        [op.neurons for op in ops],
        [op.J.shape[0] for op in ops],
        "tau_rc",
        dtype,
        shape=(-1,),
    )
    const3 = signals.op_constant(
        [op.neurons for op in ops],
        [op.J.shape[0] for op in ops],
        "tau_rc",
        dtype,
        shape=(1, -1, 1),
    )

    assert const.dtype.base_dtype == dtype

    if diff:
        assert np.array_equal(const, [[1.0] * 10 + [2.0] * 10])
        assert np.array_equal(const[0], const1)
        assert np.array_equal(const, const3[..., 0])
    else:
        assert np.array_equal(const, 1.0)
        assert np.array_equal(const, const1)
        assert np.array_equal(const, const3)


def test_get_tensor_signal():
    signals = SignalDict(tf.float32, 3)

    # check that tensor_signal is created correctly
    key = object()
    tensor_signal = signals.get_tensor_signal([(0, 1)], key, np.float64, (3, 4), True)

    assert isinstance(tensor_signal, TensorSignal)
    assert tensor_signal.slices == ((0, 1),)
    assert tensor_signal.key == key
    assert tensor_signal.dtype == np.float64
    assert tensor_signal.shape == (3, 4)
    assert tensor_signal.minibatch_size == 3
    assert len(signals) == 0

    # check adding signal to sig_map
    sig = Signal(np.zeros(4))
    sig.minibatched = True
    tensor_signal = signals.get_tensor_signal(
        [(0, 4)], key, np.float64, (2, 2), True, signal=sig
    )
    assert len(signals) == 1
    assert signals[sig] is tensor_signal
    assert next(iter(signals)) is sig
    assert next(iter(signals.values())) is tensor_signal

    # error if sig shape doesn't match indices
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros((2, 2)))
        sig.minibatched = True
        signals.get_tensor_signal([(0, 4)], key, np.float64, (2, 2), True, signal=sig)

    # error if sig size doesn't match given shape
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros(4))
        sig.minibatched = True
        signals.get_tensor_signal([(0, 4)], key, np.float64, (2, 3), True, signal=sig)

    # error if minibatched doesn't match
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros(4))
        sig.minibatched = False
        signals.get_tensor_signal([(0, 4)], key, np.float64, (2, 2), True, signal=sig)


@pytest.mark.eager_only
@pytest.mark.parametrize("ndims", (1, 2, 3))
def test_tf_indices_nd(ndims):
    signals = SignalDict(tf.float32, 10)
    shape = (3, 4, 5)[:ndims]
    x = tf.ones(shape) * tf.reshape(
        tf.range(0, 3, dtype=tf.float32), (-1,) + (1,) * (ndims - 1)
    )
    assert x.shape == shape
    sig = signals.get_tensor_signal([(0, 1), (2, 3)], None, np.float32, shape, False)
    indices = sig.tf_indices_nd

    result = tf.gather_nd(x, indices)

    assert result.shape == (2,) + shape[1:]
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 2)
