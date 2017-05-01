import nengo
import numpy as np


def test_lif_deterministic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(
            100, 1, noise=nengo.processes.WhiteNoise(seed=seed))
        p = nengo.Probe(ens.neurons)

    with nengo.Simulator(net) as sim:
        sim.run_steps(50)

    canonical = sim.data[p]

    for _ in range(5):
        with Simulator(net) as sim:
            sim.run_steps(50)

        assert np.allclose(sim.data[p], canonical)
