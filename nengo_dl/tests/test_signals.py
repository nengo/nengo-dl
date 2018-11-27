# pylint: disable=missing-docstring

from nengo.builder.neurons import SimNeurons
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from nengo.neurons import LIF
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl.signals import TensorSignal, SignalDict


def test_tensor_signal_basic():
    # check that indices are read-only
    sig = TensorSignal([0, 1, 2], None, None, None, None, None)
    with pytest.raises(BuildError):
        sig.indices = [2, 3, 4]
    with pytest.raises(ValueError):
        sig.indices[0] = 1

    # check ndim
    sig = TensorSignal([0, 1, 2], None, None, (1, 2), 1, None)
    assert sig.ndim == 2


def test_tensor_signal_getitem():
    sig = TensorSignal([1, 2, 3, 4], object(), None, (4, 3), None, None)
    sig_slice = sig[:2]
    assert np.all(sig_slice.indices == (1, 2))
    assert sig_slice.key == sig.key
    assert sig_slice.shape == (2, 3)

    assert sig[...] is sig

    sig_adv = sig[[1, 3]]
    assert np.all(sig_adv.indices == (2, 4))
    assert sig_adv.shape == (2, 3)


def test_tensor_signal_reshape():
    sig = TensorSignal([1, 2, 3, 4], object(), None, (4, 3), None, None)

    with pytest.raises(BuildError):
        sig.reshape((100,))

    sig_reshape = sig.reshape((6, 2))
    assert np.all(sig_reshape.indices == sig.indices)
    assert sig_reshape.key == sig.key
    assert sig_reshape.shape == (6, 2)

    sig_reshape = sig.reshape((6, 2, 1))
    assert np.all(sig_reshape.indices == sig.indices)
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


def test_tensor_signal_broadcast():
    sig = TensorSignal([0, 1, 2, 3], object(), None, (4,), None, None)
    base = np.random.randn(4)

    sig_broad = sig.broadcast(-1, 2)
    assert sig_broad.shape == (4, 2)
    assert sig_broad.key == sig.key
    assert np.all(
        np.reshape(base[sig_broad.indices], sig_broad.shape) ==
        base[:, None])

    sig_broad = sig.broadcast(0, 2)
    assert sig_broad.shape == (2, 4)
    assert sig_broad.key == sig.key
    assert np.all(
        np.reshape(base[sig_broad.indices], sig_broad.shape) ==
        base[None, :])


def test_tensor_signal_load_indices():
    sess = tf.InteractiveSession()

    sig = TensorSignal([2, 3, 4, 5], object(), None, (4,), None, tf.constant)
    assert np.all(sig.tf_indices.eval() == sig.indices)
    start, stop, step = sess.run(sig.tf_slice)
    assert start == 2
    assert stop == 6
    assert step == 1

    sig = TensorSignal([2, 4, 6, 8], object(), None, (4,), None, tf.constant)
    assert np.all(sig.tf_indices.eval() == sig.indices)
    start, stop, step = sess.run(sig.tf_slice)
    assert start == 2
    assert stop == 9
    assert step == 2

    sig = TensorSignal([2, 2, 3, 3], object(), None, (4,), None, tf.constant)
    assert np.all(sig.tf_indices.eval() == sig.indices)
    assert sig.tf_slice is None

    sess.close()


def test_signal_dict_scatter():
    minibatch_size = 1
    var_size = 19
    signals = SignalDict(tf.float32, minibatch_size)

    sess = tf.InteractiveSession()

    key = object()
    val = np.random.randn(var_size, minibatch_size)
    signals.bases = {key: tf.assign(tf.Variable(val, dtype=tf.float32),
                                    val)}

    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (4,), True)
    with pytest.raises(BuildError):
        # wrong dtype
        signals.scatter(x, tf.ones((4,), dtype=tf.float64))

    # update
    signals.scatter(x, tf.ones((4,)))
    y = sess.run(signals.bases[key])
    assert np.allclose(y[:4], 1)
    assert np.allclose(y[4:], val[4:])

    # increment, and reshaping val
    signals.scatter(x, tf.ones((2, 2)), mode="inc")
    y = sess.run(signals.bases[key])
    assert np.allclose(y[:4], 2)
    assert np.allclose(y[4:], val[4:])

    # recognize assignment to full array
    x = signals.get_tensor_signal(np.arange(var_size), key, tf.float32,
                                  (var_size,), True)
    y = tf.ones((var_size, 1))
    signals.scatter(x, y)
    assert signals.bases[key].op.type == "Assign"

    # recognize assignment to strided full array
    x = signals.get_tensor_signal(np.arange(0, var_size, 2), key, tf.float32,
                                  (var_size // 2 + 1,), True)
    y = tf.ones((var_size // 2 + 1, 1))
    signals.scatter(x, y)
    assert signals.bases[key].op.type == "ScatterUpdate"

    sess.close()


def test_signal_dict_gather():
    minibatch_size = 1
    var_size = 19
    signals = SignalDict(tf.float32, minibatch_size)

    sess = tf.InteractiveSession()

    key = object()
    val = np.random.randn(var_size, minibatch_size)
    signals.bases = {key: tf.constant(val, dtype=tf.float32)}

    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (4,), True)

    # sliced read
    assert np.allclose(sess.run(signals.gather(x)), val[:4])

    # read with reshape
    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (2, 2), True)
    assert np.allclose(sess.run(signals.gather(x)),
                       val[:4].reshape((2, 2, minibatch_size)))

    # gather read
    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (4,), True)
    y = signals.gather(x, force_copy=True)
    assert "Gather" in y.op.type

    x = signals.get_tensor_signal([0, 0, 3, 3], key, tf.float32, (4,), True)
    assert np.allclose(sess.run(signals.gather(x)),
                       val[[0, 0, 3, 3]])
    assert "Gather" in y.op.type

    # reading from full array
    x = signals.get_tensor_signal(np.arange(var_size), key, tf.float32,
                                  (var_size,), True)
    y = signals.gather(x)
    assert y is signals.bases[key]

    # reading from strided full array
    x = signals.get_tensor_signal(np.arange(0, var_size, 2), key, tf.float32,
                                  (var_size // 2 + 1,), True)
    y = signals.gather(x)
    assert y.op.type == "StridedSlice"
    assert y.op.inputs[0] is signals.bases[key]

    # minibatch dimension
    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (4,), True)
    assert signals.gather(x).get_shape() == (4, 1)

    x = signals.get_tensor_signal([0, 1, 2, 3], key, tf.float32, (4,), False)
    assert signals.gather(x).get_shape() == (4,)

    sess.close()


def test_signal_dict_combine():
    minibatch_size = 1
    signals = SignalDict(tf.float32, minibatch_size)

    key = object()

    assert signals.combine([]) == []

    y = signals.combine(
        [signals.get_tensor_signal([0, 1, 2], key, None, (3, 2), False),
         signals.get_tensor_signal([4, 5, 6], key, None, (3, 2), False)])
    assert y.key is key
    assert y._tf_indices is None

    assert y.shape == (6, 2)

    assert np.all(y.indices == [0, 1, 2, 4, 5, 6])


@pytest.mark.parametrize("dtype", (None, tf.float32))
def test_constant(dtype, sess):
    val = np.random.randn(10, 10).astype(np.float64)

    signals = SignalDict(tf.float32, 1)
    const0 = signals.constant(val, dtype=dtype)
    const1 = signals.constant(val, dtype=dtype, cutoff=0)

    assert const0.op.type == "Const"
    assert const1.op.type == "Identity"

    assert const0.dtype == (dtype if dtype else tf.as_dtype(val.dtype))
    assert const1.dtype == (dtype if dtype else tf.as_dtype(val.dtype))

    sess.run(tf.variables_initializer(tf.get_collection("constants")),
             feed_dict=signals.constant_phs)
    c0, c1 = sess.run([const0, const1])

    assert np.allclose(c0, val)
    assert np.allclose(c1, val)


@pytest.mark.gpu
def test_constant_gpu(sess):
    val = np.random.randn(10, 10).astype(np.int32)

    with tf.device("/gpu:0"):
        signals = SignalDict(tf.float32, 1)
        const = signals.constant(val, cutoff=0)

        assert const.dtype == tf.int32
        assert "GPU" in const.device.upper()

    sess.run(tf.variables_initializer(tf.get_collection("constants")),
             feed_dict=signals.constant_phs)
    c = sess.run(const)

    assert np.allclose(val, c)


@pytest.mark.parametrize("dtype", (tf.float32, tf.float64))
@pytest.mark.parametrize("diff", (True, False))
def test_op_constant(dtype, diff, sess):
    ops = (SimNeurons(LIF(tau_rc=1), Signal(np.zeros(10)), None),
           SimNeurons(LIF(tau_rc=2 if diff else 1), Signal(np.zeros(10)),
                      None))

    signals = SignalDict(tf.float32, 1)
    const = signals.op_constant(
        [op.neurons for op in ops], [op.J.shape[0] for op in ops],
        "tau_rc", dtype)
    const1 = signals.op_constant(
        [op.neurons for op in ops], [op.J.shape[0] for op in ops],
        "tau_rc", dtype, ndims=1)
    const3 = signals.op_constant(
        [op.neurons for op in ops], [op.J.shape[0] for op in ops],
        "tau_rc", dtype, ndims=3)

    assert const.dtype.base_dtype == dtype

    sess.run(tf.variables_initializer(tf.get_collection("constants")),
             feed_dict=signals.constant_phs)
    x, x1, x3 = sess.run([const, const1, const3])

    if diff:
        assert np.array_equal(x, [[1]] * 10 + [[2]] * 10)
        assert np.array_equal(x[:, 0], x1)
        assert np.array_equal(x, x3[..., 0])
    else:
        assert np.array_equal(x, 1.0)
        assert np.array_equal(x, x1)
        assert np.array_equal(x, x3)


def test_get_tensor_signal():
    signals = SignalDict(tf.float32, 3)

    # check that tensor_signal is created correctly
    key = object()
    tensor_signal = signals.get_tensor_signal(
        (0,), key, np.float64, (3, 4), True)

    assert isinstance(tensor_signal, TensorSignal)
    assert np.array_equal(tensor_signal.indices, (0,))
    assert tensor_signal.key == key
    assert tensor_signal.dtype == np.float64
    assert tensor_signal.shape == (3, 4)
    assert tensor_signal.minibatch_size == 3
    assert tensor_signal.constant == signals.constant
    assert len(signals) == 0

    # check adding signal to sig_map
    sig = Signal(np.zeros(4))
    sig.minibatched = True
    tensor_signal = signals.get_tensor_signal(
        np.arange(4), key, np.float64, (2, 2), True, signal=sig)
    assert len(signals) == 1
    assert signals[sig] is tensor_signal
    assert next(iter(signals)) is sig
    assert next(iter(signals.values())) is tensor_signal

    # error if sig shape doesn't match indices
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros((2, 2)))
        sig.minibatched = True
        signals.get_tensor_signal(
            np.arange(4), key, np.float64, (2, 2), True, signal=sig)

    # error if sig size doesn't match given shape
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros(4))
        sig.minibatched = True
        signals.get_tensor_signal(
            np.arange(4), key, np.float64, (2, 3), True, signal=sig)

    # error if minibatched doesn't match
    with pytest.raises(AssertionError):
        sig = Signal(np.zeros(4))
        sig.minibatched = False
        signals.get_tensor_signal(
            np.arange(4), key, np.float64, (2, 2), True, signal=sig)
