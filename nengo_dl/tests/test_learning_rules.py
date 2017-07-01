from functools import partial

import nengo
from nengo.builder.learning_rules import SimVoja, SimOja, SimBCM
import numpy as np
import pytest

from nengo_dl import configure_settings, graph_optimizer


@pytest.mark.parametrize("rule", (nengo.Voja, nengo.Oja, nengo.BCM))
def test_merged_learning(Simulator, rule, seed):
    # a slightly more complicated network with mergeable learning rules, to
    # make sure that works OK
    dimensions = 2
    with nengo.Network(seed=seed) as net:
        configure_settings(
            planner=partial(graph_optimizer.tree_planner, max_depth=10))

        a = nengo.Ensemble(3, dimensions, label="a")
        b = nengo.Ensemble(3, dimensions, label="b")
        c = nengo.Ensemble(5, dimensions, label="c")
        d = nengo.Ensemble(10, dimensions, label="d")

        conn0 = nengo.Connection(
            a, c, learning_rule_type=rule(),
            solver=nengo.solvers.LstsqL2(weights=rule != nengo.Voja))
        conn1 = nengo.Connection(
            b, d, learning_rule_type=rule(),
            solver=nengo.solvers.LstsqL2(weights=rule != nengo.Voja))

        p0 = nengo.Probe(conn0.learning_rule, "delta")
        p1 = nengo.Probe(conn1.learning_rule, "delta")

    with nengo.Simulator(net) as sim:
        sim.run_steps(10)

        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net) as sim:
        build_type = {nengo.Voja: SimVoja, nengo.Oja: SimOja,
                      nengo.BCM: SimBCM}

        assert len([x for x in sim.tensor_graph.plan
                    if type(x[0]) == build_type[rule]]) == 1

        sim.run_steps(10)

        assert np.allclose(sim.data[p0], canonical[0])
        assert np.allclose(sim.data[p1], canonical[1])
