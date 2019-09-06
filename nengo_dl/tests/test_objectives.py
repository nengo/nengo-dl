# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest

from nengo_dl.compat import tf_compat


@pytest.mark.xfail(reason="TODO: support train")
@pytest.mark.training
@pytest.mark.parametrize("mode", ("activity", "weights"))
def test_regularize_train(Simulator, mode, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([1])
        b = nengo.Ensemble(
            30,
            1,
            neuron_type=nengo.Sigmoid(tau_ref=1),
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )
        c = nengo.Connection(
            a, b.neurons, synapse=None, transform=nengo.dists.Uniform(-0.1, 0.1)
        )

        if mode == "weights":
            p = nengo.Probe(c, "weights")
        else:
            p = nengo.Probe(b.neurons)

    with Simulator(net) as sim:
        sim.train(
            5,
            tf_compat.train.RMSPropOptimizer(0.01 if mode == "weights" else 0.1),
            objective={p: objectives.Regularize()},
            n_epochs=100,
        )

        sim.step()
        assert np.allclose(sim.data[p], 0, atol=1e-2)
