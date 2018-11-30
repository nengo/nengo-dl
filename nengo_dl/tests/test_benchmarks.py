# pylint: disable=missing-docstring

from collections import defaultdict
import sys

import pytest
import nengo
import tensorflow as tf

from nengo_dl import benchmarks, SoftLIFRate


@pytest.mark.parametrize(
    "benchmark", (benchmarks.cconv, benchmarks.integrator, benchmarks.pes,
                  benchmarks.basal_ganglia))
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
    ((1, 10, nengo.RectifiedLinear(), 5, 3),
     (2, 4, nengo.LIF(), 10, 2)))
def test_random_network(dimensions, neurons_per_d, neuron_type, n_ensembles,
                        n_connections):
    net = benchmarks.random_network(dimensions, neurons_per_d, neuron_type,
                                    n_ensembles, n_connections)
    _test_random(net, dimensions, neurons_per_d, neuron_type, n_ensembles,
                 n_connections)


def _test_random(net, dimensions, neurons_per_d, neuron_type, n_ensembles,
                 n_connections):
    assert net.inp.size_out == dimensions
    assert net.out.size_in == dimensions
    assert len(net.all_ensembles) == n_ensembles
    assert all(ens.neuron_type == neuron_type for ens in net.all_ensembles)
    assert all(ens.n_neurons == dimensions * neurons_per_d
               for ens in net.all_ensembles)

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


@pytest.mark.parametrize("train", (True, False))
def test_run_profile(train, pytestconfig):
    net = benchmarks.integrator(3, 2, nengo.RectifiedLinear())

    benchmarks.run_profile(
        net, train=train, n_steps=10, do_profile=False,
        device=pytestconfig.getvalue("--device"),
        unroll_simulation=pytest.config.getvalue("--unroll_simulation"),
        dtype=(tf.float32 if pytest.config.getvalue("dtype") == "float32" else
               tf.float64))

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
        "--unroll 5 --time-only" % (
            dimensions, neurons_per_d, n_ensembles, n_connections)).split()
    obj = {}
    with pytest.raises(SystemExit):
        benchmarks.main(obj=obj)

    _test_random(obj["net"], dimensions, neurons_per_d, SoftLIFRate(),
                 n_ensembles, n_connections)

    assert 0 < obj["time"] < 1

    with pytest.raises(ValueError):
        sys.argv = [sys.argv[0], "profile"]
        benchmarks.main(obj={})

    sys.argv = old_argv
