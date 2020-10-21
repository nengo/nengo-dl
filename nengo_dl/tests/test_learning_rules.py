# pylint: disable=missing-docstring

from functools import partial

import nengo
import numpy as np
import pytest
from nengo.builder.learning_rules import SimBCM, SimOja, SimVoja

from nengo_dl import configure_settings, graph_optimizer
from nengo_dl.learning_rule_builders import SimPES


@pytest.mark.parametrize(
    "rule, weights",
    [
        (nengo.Voja, False),
        (nengo.Oja, True),
        (nengo.BCM, True),
        (nengo.PES, True),
        (nengo.PES, False),
    ],
)
def test_merged_learning(Simulator, rule, weights, seed):
    # a slightly more complicated network with mergeable learning rules, to
    # make sure that works OK
    dimensions = 2
    with nengo.Network(seed=seed) as net:
        configure_settings(
            planner=partial(graph_optimizer.tree_planner, max_depth=10), dtype="float64"
        )

        a = nengo.Ensemble(3, dimensions, label="a")
        b = nengo.Ensemble(3, dimensions, label="b")
        c = nengo.Ensemble(5, dimensions, label="c")

        # for PES rules the post (error) shape also has to match for the rules
        # to be mergeable
        d = nengo.Ensemble(5 if rule == nengo.PES else 10, dimensions, label="d")

        conn0 = nengo.Connection(
            a,
            c,
            learning_rule_type=rule(learning_rate=0.1),
            solver=nengo.solvers.LstsqL2(weights=weights),
        )
        conn1 = nengo.Connection(
            b,
            d,
            learning_rule_type=rule(learning_rate=0.2),
            solver=nengo.solvers.LstsqL2(weights=weights),
        )

        p0 = nengo.Probe(conn0.learning_rule, "delta")
        p1 = nengo.Probe(conn1.learning_rule, "delta")

    with nengo.Simulator(net) as sim:
        sim.run_steps(10)

        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net, minibatch_size=2) as sim:
        build_type = {
            nengo.Voja: SimVoja,
            nengo.Oja: SimOja,
            nengo.BCM: SimBCM,
            nengo.PES: SimPES,
        }

        assert (
            len([x for x in sim.tensor_graph.plan if type(x[0]) == build_type[rule]])
            == 1
        )

        sim.run_steps(10)

        for i in range(sim.minibatch_size):
            assert np.allclose(sim.data[p0][i], canonical[0])
            assert np.allclose(sim.data[p1][i], canonical[1])


def test_online_learning_reset(Simulator, tmp_path, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Ensemble(10, 1)
        out = nengo.Node(size_in=1)
        conn = nengo.Connection(inp, out, learning_rule_type=nengo.PES(1))
        nengo.Connection(nengo.Node([1]), conn.learning_rule)

    with Simulator(net) as sim:
        w0 = np.array(sim.data[conn].weights)

        sim.run(0.1, stateful=False)

        w1 = np.array(sim.data[conn].weights)

        sim.save_params(tmp_path / "tmp")

        # test that learning has changed weights
        assert not np.allclose(w0, w1)

        # test that include_trainable=False does NOT reset the online learning weights
        sim.reset(include_trainable=False)
        assert np.allclose(w1, sim.data[conn].weights)

        # test that full reset DOES reset the online learning weights
        sim.reset(include_trainable=True)
        assert np.allclose(w0, sim.data[conn].weights)

    # test that weights load correctly
    with Simulator(net) as sim:
        assert not np.allclose(w1, sim.data[conn].weights)

        sim.load_params(tmp_path / "tmp")

        assert np.allclose(w1, sim.data[conn].weights)
