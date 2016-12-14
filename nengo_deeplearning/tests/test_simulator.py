import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import ElementwiseInc, DotInc
from nengo.utils.testing import warns
import numpy as np

import nengo_deeplearning


def test_warn_on_opensim_del():
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)

    sim = nengo_deeplearning.Simulator(net)
    with warns(RuntimeWarning):
        sim.__del__()
    sim.close()


def test_args():
    class Fn(object):
        def __init__(self):
            self.last_x = None

        def __call__(self, t, x):
            assert isinstance(t, np.float32)
            assert t.shape == ()
            assert isinstance(x, np.ndarray)
            assert self.last_x is not x  # x should be a new copy on each call
            self.last_x = x
            assert x[0] == t

    with nengo.Network() as model:
        u = nengo.Node(lambda t: t)
        v = nengo.Node(Fn(), size_in=1, size_out=0)
        nengo.Connection(u, v, synapse=None)

    with nengo_deeplearning.Simulator(model) as sim:
        sim.run(0.01)


def test_signal_init_values():
    """Tests that initial values are not overwritten."""
    zero = Signal([0.0])
    one = Signal([1.0])
    five = Signal([5.0])
    zeroarray = Signal([[0.0], [0.0], [0.0]])
    array = Signal([1.0, 2.0, 3.0])

    m = nengo.builder.Model(dt=0)
    m.operators += [ElementwiseInc(zero, zero, five),
                    DotInc(zeroarray, one, array)]

    with nengo_deeplearning.Simulator(None, model=m) as sim:
        tensors = [sim.signals[s] for s in (zero, one, five, array)]
        output = sim.sess.run(tensors)
        assert output[0][0] == 0.0
        assert output[1][0] == 1.0
        assert output[2][0] == 5.0
        assert np.all(output[3] == np.array([1, 2, 3]))
        output = sim.sess.run(tensors)
        assert output[0][0] == 0.0
        assert output[1][0] == 1.0
        assert output[2][0] == 5.0
        assert np.all(output[3] == np.array([1, 2, 3]))
