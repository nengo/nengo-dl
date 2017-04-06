import nengo
from nengo.exceptions import BuildError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl.signals import TensorSignal, SignalDict, mark_signals


def test_tensor_signal_basic():
    # check that indices are read-only
    sig = TensorSignal([0, 1, 2], None, None, None, None)
    with pytest.raises(BuildError):
        sig.indices = [2, 3, 4]
    with pytest.raises(ValueError):
        sig.indices[0] = 1

    # check ndim
    sig = TensorSignal([0, 1, 2], None, None, (1, 2), True)
    assert sig.ndim == 2


def test_tensor_signal_getitem():
    sig = TensorSignal([1, 2, 3, 4], object(), None, (4, 3), None)
    sig_slice = sig[:2]
    assert np.all(sig_slice.indices == (1, 2))
    assert sig_slice.key == sig.key
    assert sig_slice.shape == (2, 3)

    assert sig[...] is sig

    sig_adv = sig[[1, 3]]
    assert np.all(sig_adv.indices == (2, 4))
    assert sig_adv.shape == (2, 3)


def test_tensor_signal_reshape():
    sig = TensorSignal([1, 2, 3, 4], object(), None, (4, 3), None)

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


def test_tensor_signal_broadcast():
    sig = TensorSignal([0, 1, 2, 3], object(), None, (4,), None)
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
    with tf.Session() as sess:
        sig = TensorSignal([2, 3, 4, 5], object(), None, (4,), None)
        sig.load_indices()
        assert np.all(sig.tf_indices.eval() == sig.indices)
        start, stop, step = sess.run(sig.as_slice)
        assert start == 2
        assert stop == 6
        assert step == 1

        sig = TensorSignal([2, 4, 6, 8], object(), None, (4,), None)
        sig.load_indices()
        assert np.all(sig.tf_indices.eval() == sig.indices)
        start, stop, step = sess.run(sig.as_slice)
        assert start == 2
        assert stop == 9
        assert step == 2

        sig = TensorSignal([2, 2, 3, 3], object(), None, (4,), None)
        sig.load_indices()
        assert np.all(sig.tf_indices.eval() == sig.indices)
        assert sig.as_slice is None


def test_signal_dict_scatter():
    minibatch_size = 1
    signals = SignalDict(None, tf.float32, minibatch_size)

    with tf.Session() as sess:
        key = object()
        val = np.random.randn(20, minibatch_size)
        signals.bases = {key: tf.assign(tf.Variable(val, dtype=tf.float32),
                                        val)}

        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (4,), False)
        with pytest.raises(BuildError):
            # assigning to trainable variable
            signals.scatter(x, None)

        x.minibatched = True
        with pytest.raises(BuildError):
            # indices not loaded
            signals.scatter(x, None)

        x.load_indices()
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
        x = TensorSignal(np.arange(20), key, tf.float32, (20,), True)
        x.load_indices()
        y = tf.ones((20, 1))
        signals.scatter(x, y)
        assert signals.bases[key].op.type == "Assign"


def test_signal_dict_gather():
    minibatch_size = 1
    signals = SignalDict(None, tf.float32, minibatch_size)

    with tf.Session() as sess:
        key = object()
        val = np.random.randn(20, minibatch_size)
        signals.bases = {key: tf.constant(val, dtype=tf.float32)}

        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (4,), True)
        with pytest.raises(BuildError):
            # indices not loaded
            signals.gather(x)

        # sliced read
        x.load_indices()
        assert np.allclose(sess.run(signals.gather(x)), val[:4])

        # read with reshape
        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (2, 2), True)
        x.load_indices()
        assert np.allclose(sess.run(signals.gather(x)),
                           val[:4].reshape((2, 2, minibatch_size)))

        # gather read
        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (4,), True)
        x.load_indices()
        y = signals.gather(x, force_copy=True)
        assert y.op.type == "Gather"

        x = TensorSignal([0, 0, 3, 3], key, tf.float32, (4,), True)
        x.load_indices()
        assert np.allclose(sess.run(signals.gather(x)),
                           val[[0, 0, 3, 3]])
        assert y.op.type == "Gather"

        # reading from full array
        x = TensorSignal(np.arange(20), key, tf.float32, (20,), True)
        x.load_indices()
        y = signals.gather(x)
        assert y.op.type == "Identity"
        assert y.op.inputs[0] is signals.bases[key]

        # minibatch dimension
        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (4,), True)
        x.load_indices()
        assert signals.gather(x).get_shape() == (4, 1)

        x = TensorSignal([0, 1, 2, 3], key, tf.float32, (4,), False)
        x.load_indices()
        assert signals.gather(x).get_shape() == (4,)


def test_signal_dict_combine():
    minibatch_size = 1
    signals = SignalDict(None, tf.float32, minibatch_size)

    key = object()

    assert signals.combine([]) == []

    y = signals.combine([TensorSignal([0, 1, 2], key, None, (3, 2), None),
                         TensorSignal([4, 5, 6], key, None, (3, 2), None)],
                        False)
    assert y.key is key
    assert y.tf_indices is None

    assert y.shape == (6, 2)

    assert np.all(y.indices == [0, 1, 2, 4, 5, 6])


def test_mark_signals():
    with nengo.Network() as net:
        ens0 = nengo.Ensemble(10, 1, neuron_type=nengo.LIF())
        ens1 = nengo.Ensemble(20, 1, neuron_type=nengo.Direct())
        ens2 = nengo.Ensemble(30, 1)
        conn0 = nengo.Connection(ens0, ens1)
        conn1 = nengo.Connection(ens0, ens1, learning_rule_type=nengo.PES())
        conn2 = nengo.Connection(ens0, ens2, learning_rule_type=nengo.Voja())
        nengo.Probe(ens2)

    model = nengo.builder.Model()
    model.build(net)

    mark_signals(model)

    assert model.sig[ens0]["encoders"].trainable
    assert model.sig[ens1]["encoders"].trainable
    assert not model.sig[ens2]["encoders"].trainable
    assert model.sig[ens0.neurons]["bias"].trainable
    assert model.sig[ens2.neurons]["bias"].trainable
    assert model.sig[conn0]["weights"].trainable
    assert not model.sig[conn1]["weights"].trainable
    assert model.sig[conn2]["weights"].trainable

    trainables = (
        model.sig[ens0]["encoders"], model.sig[ens1]["encoders"],
        model.sig[ens0.neurons]["bias"], model.sig[ens2.neurons]["bias"],
        model.sig[conn0]["weights"], model.sig[conn2]["weights"])

    for op in model.operators:
        for sig in op.all_signals:
            if sig in trainables:
                assert sig.trainable
            else:
                assert not sig.trainable
