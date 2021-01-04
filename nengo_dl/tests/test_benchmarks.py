# pylint: disable=missing-docstring

import sys
from collections import defaultdict

import nengo
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import SoftLIFRate, benchmarks


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


@pytest.mark.parametrize(
    "network, train", [("integrator", True), ("cconv", False), ("test", True)]
)
def test_run_profile(network, train, pytestconfig, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    if network == "integrator":
        net = benchmarks.integrator(3, 2, nengo.SpikingRectifiedLinear())
    elif network == "cconv":
        net = benchmarks.cconv(3, 10, nengo.LIF())
    elif network == "test":
        with nengo.Network() as net:
            ens = nengo.Ensemble(10, 1)
            net.p = nengo.Probe(ens)

    benchmarks.run_profile(
        net,
        train=train,
        n_steps=10,
        do_profile=True,
        device=pytestconfig.getoption("--device"),
        unroll_simulation=pytestconfig.getoption("--unroll-simulation"),
        dtype=pytestconfig.getoption("dtype"),
        progress_bar=False,
    )

    assert net.config[net].inference_only == (not train)


def test_cli():
    dimensions = 2
    neurons_per_d = 1
    n_ensembles = 4
    n_connections = 3

    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + (
        f"build "
        f"--benchmark random_network "
        f"--dimensions {dimensions} "
        f"--neurons_per_d {neurons_per_d} "
        f"--neuron_type SoftLIFRate "
        f"--kwarg n_ensembles={n_ensembles} "
        f"--kwarg connections_per_ensemble={n_connections} "
        f"profile "
        f"--no-train --n_steps 10 --batch_size 2 --device /cpu:0 --unroll 5 --time-only"
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
    "net, train, minibatch_size, eager, min, max",
    [
        (
            benchmarks.cconv(128, 64, nengo.RectifiedLinear()),
            False,
            64,
            True,
            1.0,
            1.15,
        ),
        (benchmarks.cconv(128, 64, nengo.LIF()), False, 64, True, 2.25, 2.55),
        (
            benchmarks.integrator(128, 32, nengo.RectifiedLinear()),
            True,
            64,
            True,
            0.6,
            0.9,
        ),
        (benchmarks.integrator(128, 32, nengo.LIF()), True, 64, True, 0.95, 1.15),
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
            True,
            0.5,
            0.7,
        ),
        (benchmarks.lmu(1000, 1, native_nengo=True), True, 100, True, 1.25, 1.45),
        (benchmarks.lmu(1000, 1, native_nengo=True), True, 100, False, 1.05, 1.25),
    ],
)
def test_performance(net, train, minibatch_size, eager, min, max):
    # performance is based on Azure NC6 VM
    # CPU: Intel Xeon E5-2690 v3 @ 2.60Ghz
    # GPU: Nvidia Tesla K80
    # Python version: 3.7.1
    # TensorFlow GPU version: 2.2.1
    # Nengo version: 3.1.0
    # NengoDL version: 3.4.1

    if not eager:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_control_flow_v2()

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
