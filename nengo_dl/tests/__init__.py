import logging
import os

import nengo
from nengo.tests import test_synapses, test_learning_rules
import numpy as np
import tensorflow as tf

from nengo_dl import simulator


# set looser tolerances on synapse tests
def allclose(*args, **kwargs):
    kwargs.setdefault('atol', 5e-7)
    return nengo.utils.testing.allclose(*args, **kwargs)


test_synapses.allclose = allclose

# cast output of run_synapse to float64. this is necessary because
# Synapse.filt bases its internal dtypes on the dtype of its inputs, and
# we don't want to downcast everything there to float32.
nengo_run_synapse = test_synapses.run_synapse


def run_synapse(*args, **kwargs):
    output = nengo_run_synapse(*args, **kwargs)
    return tuple(x.astype(np.float64) for x in output)


test_synapses.run_synapse = run_synapse


# remove the correction probes from _test_pes
def _test_pes(NengoSimulator, nl, plt, seed,
              pre_neurons=False, post_neurons=False, weight_solver=False,
              vin=np.array([0.5, -0.5]), vout=None, n=200,
              function=None, transform=np.array(1.), rate=1e-3):
    vout = np.array(vin) if vout is None else vout

    model = nengo.Network(seed=seed)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl()

        u = nengo.Node(output=vin)
        v = nengo.Node(output=vout)
        a = nengo.Ensemble(n, dimensions=u.size_out)
        b = nengo.Ensemble(n, dimensions=u.size_out)
        e = nengo.Ensemble(n, dimensions=v.size_out)

        nengo.Connection(u, a)

        bslice = b[:v.size_out] if v.size_out < u.size_out else b
        pre = a.neurons if pre_neurons else a
        post = b.neurons if post_neurons else bslice

        conn = nengo.Connection(pre, post,
                                function=function, transform=transform,
                                learning_rule_type=nengo.PES(rate))
        if weight_solver:
            conn.solver = nengo.solvers.LstsqL2(weights=True)

        nengo.Connection(v, e, transform=-1)
        nengo.Connection(bslice, e)
        nengo.Connection(e, conn.learning_rule)

        b_p = nengo.Probe(bslice, synapse=0.03)
        e_p = nengo.Probe(e, synapse=0.03)

        weights_p = nengo.Probe(conn, 'weights', sample_every=0.01)

    with NengoSimulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()
    weights = sim.data[weights_p]

    tend = t > 0.4
    assert np.allclose(sim.data[b_p][tend], vout, atol=0.05)
    assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)
    assert not np.allclose(weights[0], weights[-1], atol=1e-5)


test_learning_rules._test_pes = _test_pes


class Simulator(simulator.Simulator):
    def __init__(self, *args, **kwargs):
        logging.basicConfig(level=logging.WARNING)

        if "NENGO_DL_TEST_PRECISION" in os.environ:
            if os.environ["NENGO_DL_TEST_PRECISION"] == "32":
                kwargs.setdefault("dtype", tf.float32)
            else:
                kwargs.setdefault("dtype", tf.float64)

        if "NENGO_DL_TEST_UNROLL" in os.environ:
            kwargs.setdefault("unroll_simulation",
                              int(os.environ["NENGO_DL_TEST_UNROLL"]))

        if "NENGO_DL_TEST_DEVICE" in os.environ:
            device = os.environ["NENGO_DL_TEST_DEVICE"]
            if device == "None":
                kwargs.setdefault("device", None)
            else:
                kwargs.setdefault("device", os.environ["NENGO_DL_TEST_DEVICE"])

        super(Simulator, self).__init__(*args, **kwargs)
