# pylint: disable=missing-docstring

from functools import partial

import nengo
from nengo.builder.learning_rules import SimVoja, SimOja, SimBCM
import numpy as np
import pytest

from nengo_dl import configure_settings, graph_optimizer
from nengo_dl.learning_rule_builders import SimPES


@pytest.mark.parametrize("rule, weights", [(nengo.Voja, False),
                                           (nengo.Oja, True),
                                           (nengo.BCM, True),
                                           (nengo.PES, True),
                                           (nengo.PES, False)])
def test_merged_learning(Simulator, rule, weights, seed):
    # a slightly more complicated network with mergeable learning rules, to
    # make sure that works OK
    dimensions = 2
    with nengo.Network(seed=seed) as net:
        configure_settings(
            planner=partial(graph_optimizer.tree_planner, max_depth=10))

        a = nengo.Ensemble(3, dimensions, label="a")
        b = nengo.Ensemble(3, dimensions, label="b")
        c = nengo.Ensemble(5, dimensions, label="c")

        # for PES rules the post (error) shape also has to match for the rules
        # to be mergeable
        d = nengo.Ensemble(5 if rule == nengo.PES else 10, dimensions,
                           label="d")

        conn0 = nengo.Connection(
            a, c, learning_rule_type=rule(),
            solver=nengo.solvers.LstsqL2(weights=weights))
        conn1 = nengo.Connection(
            b, d, learning_rule_type=rule(),
            solver=nengo.solvers.LstsqL2(weights=weights))

        p0 = nengo.Probe(conn0.learning_rule, "delta")
        p1 = nengo.Probe(conn1.learning_rule, "delta")

    with nengo.Simulator(net) as sim:
        sim.run_steps(10)

        canonical = (sim.data[p0], sim.data[p1])

    with Simulator(net) as sim:
        build_type = {nengo.Voja: SimVoja, nengo.Oja: SimOja,
                      nengo.BCM: SimBCM, nengo.PES: SimPES}

        assert len([x for x in sim.tensor_graph.plan
                    if type(x[0]) == build_type[rule]]) == 1

        sim.run_steps(10)

        assert np.allclose(sim.data[p0], canonical[0])
        assert np.allclose(sim.data[p1], canonical[1])
