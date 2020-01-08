# pylint: disable=missing-docstring

from collections import defaultdict
import sys

import pytest
import nengo
import numpy as np

from nengo_dl import benchmarks, SoftLIFRate


@pytest.mark.parametrize(
    "benchmark",
    (benchmarks.cconv, benchmarks.integrator, benchmarks.pes, benchmarks.basal_ganglia),
)
def test_networks(benchmark):
    dimensions = 16
    neurons_per_d = 10
    neuron_type = nengo.RectifiedLinear()

    net = benchmark(dimensions, neurons_per_d, neuron_type)

    try:
        assert net.inp.size_out == dimensions
    except AttributeError:
        assert net.inp_a.size_out == dimensions
        assert net.inp_b.size_out == dimensions

    assert net.p.size_in == dimensions

    for ens in net.all_ensembles:
        assert ens.neuron_type == neuron_type
        if benchmark == benchmarks.cconv:
            # the cconv network divides the neurons between two ensemble
            # arrays
            assert ens.n_neurons == ens.dimensions * (neurons_per_d // 2)
        else:
            assert ens.n_neurons == ens.dimensions * neurons_per_d


@pytest.mark.parametrize("tensor_layer", (True, False))
def test_mnist(tensor_layer):
    net = benchmarks.mnist(use_tensor_layer=tensor_layer)

    if tensor_layer:
        assert len(net.all_nodes) == 7
        assert len(net.all_ensembles) == 3
    else:
        assert len(net.all_nodes) == 2
        assert len(net.all_ensembles) == 0

    assert net.inp.size_out == 28 * 28
    assert net.p.size_in == 10


def test_spaun():
    pytest.importorskip("_spaun")

    dimensions = 2

    net = benchmarks.spaun(dimensions=dimensions)
    assert net.mem.mb1_net.output.size_in == dimensions


@pytest.mark.parametrize(
    "dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections",
    ((1, 10, nengo.RectifiedLinear(), 5, 3), (2, 4, nengo.LIF(), 10, 2)),
)
def test_random_network(
    dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections
):
    net = benchmarks.random_network(
        dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections
    )
    _test_random(
        net, dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections
    )


def _test_random(
    net, dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections
):
    assert net.inp.size_out == dimensions
    assert net.out.size_in == dimensions
    assert len(net.all_ensembles) == n_ensembles
    assert all(ens.neuron_type == neuron_type for ens in net.all_ensembles)
    assert all(ens.n_neurons == dimensions * neurons_per_d for ens in net.all_ensembles)

    pre_conns = defaultdict(list)
    post_conns = defaultdict(list)
    for conn in net.all_connections:
        if isinstance(conn.pre, nengo.Ensemble):
            pre_conns[conn.pre].append(conn.post)
        if isinstance(conn.post, nengo.Ensemble):
            post_conns[conn.post].append(conn.pre)

    assert len(pre_conns) == n_ensembles
    assert all(len(x) == n_connections + 1 for x in pre_conns.values())
    assert all(net.out in x for x in pre_conns.values())
    assert all(net.inp in x for x in post_conns.values())


@pytest.mark.parametrize("network, train", [("integrator", True), ("cconv", False)])
def test_run_profile(network, train, pytestconfig, monkeypatch, tmpdir):
    monkeypatch.chdir(tmpdir)

    if network == "integrator":
        net = benchmarks.integrator(3, 2, nengo.SpikingRectifiedLinear())
    elif network == "cconv":
        net = benchmarks.cconv(3, 10, nengo.LIF())

    benchmarks.run_profile(
        net,
        train=train,
        n_steps=10,
        do_profile=True,
        device=pytestconfig.getoption("--device"),
        unroll_simulation=pytestconfig.getoption("--unroll-simulation"),
        dtype=pytestconfig.getoption("dtype"),
    )

    assert net.config[net].inference_only == (not train)


def test_cli():
    dimensions = 2
    neurons_per_d = 1
    n_ensembles = 4
    n_connections = 3

    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + (
        "build --benchmark random_network --dimensions %d "
        "--neurons_per_d %d --neuron_type SoftLIFRate "
        "--kwarg n_ensembles=%d --kwarg connections_per_ensemble=%d "
        "profile --no-train --n_steps 10 --batch_size 2 --device /cpu:0 "
        "--unroll 5 --time-only"
        % (dimensions, neurons_per_d, n_ensembles, n_connections)
    ).split()
    obj = {}
    with pytest.raises(SystemExit):
        benchmarks.main(obj=obj)

    _test_random(
        obj["net"], dimensions, neurons_per_d, SoftLIFRate(), n_ensembles, n_connections
    )

    assert "time" in obj

    with pytest.raises(ValueError):
        sys.argv = [sys.argv[0], "profile"]
        benchmarks.main(obj={})

    sys.argv = old_argv


@pytest.mark.training
@pytest.mark.parametrize("native_nengo", (True, False))
def test_lmu(Simulator, native_nengo, pytestconfig):
    n_steps = 1000
    net = benchmarks.lmu(
        n_steps, 1, native_nengo=native_nengo, dtype=pytestconfig.getoption("--dtype")
    )

    # TODO: It would be good to optimize LMU performance as the NengoDL implementation
    # is a bit slower than the original TensorFlow implementation.

    # benchmarks.run_profile(
    #     net,
    #     train=True,
    #     n_steps=n_steps if native_nengo else 1,
    #     do_profile=False,
    #     minibatch_size=100,
    #     unroll_simulation=25 if native_nengo else 1,
    #     reps=5,
    # )

    with Simulator(net) as sim:
        n_trainable = sum(
            np.prod(w.shape.as_list()) for w in sim.keras_model.trainable_weights
        )
        assert n_trainable == 102017

    assert net.inp.size_out == 1 if native_nengo else n_steps
    assert net.p.size_in == 10


@pytest.mark.performance
@pytest.mark.parametrize(
    "net, train, minibatch_size, min, max",
    [
        (benchmarks.cconv(128, 64, nengo.RectifiedLinear()), False, 64, 0.65, 0.8),
        (benchmarks.cconv(128, 64, nengo.LIF()), False, 64, 1.45, 1.65),
        (benchmarks.integrator(128, 32, nengo.RectifiedLinear()), True, 64, 0.6, 1.0),
        (benchmarks.integrator(128, 32, nengo.LIF()), True, 64, 1.1, 1.4),
        (
            benchmarks.random_network(
                64,
                32,
                nengo.RectifiedLinear(),
                n_ensembles=20,
                connections_per_ensemble=5,
                seed=0,
            ),
            False,
            None,
            0.35,
            0.55,
        ),
        (benchmarks.lmu(1000, 1, native_nengo=True), True, 100, 0.85, 1.15),
    ],
)
def test_performance(net, train, minibatch_size, min, max):
    # performance is based on ABR GPU server
    # CPU: Intel Xeon E5-1650 v3 @ 3.50GHz
    # GPU: GeForce GTX Titan X
    # Python version: 3.6.8
    # TensorFlow GPU version: 2.0.0
    # Nengo version: 3.1.0
    # NengoDL version: 3.1.0

    time = benchmarks.run_profile(
        net,
        minibatch_size=minibatch_size,
        train=train,
        n_steps=1000,
        unroll_simulation=25,
        progress_bar=False,
        do_profile=False,
        reps=15,
    )
    assert time > min
    assert time < max
