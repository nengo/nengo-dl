# pylint: disable=missing-docstring

import logging
import os
import subprocess
import sys
import textwrap

import nengo
import numpy as np
import pytest
import tensorflow as tf
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from packaging import version

from nengo_dl import config, graph_optimizer, tensor_graph, utils
from nengo_dl.tests import dummies


@pytest.mark.parametrize("unroll", (1, 2))
@pytest.mark.training
def test_gradients(Simulator, unroll, seed):
    minibatch_size = 4

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        inp = nengo.Node([0], label="inp")

        # sigmoid neurons
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())

        # normal decoded connection
        nengo.Connection(inp, ens)

        # recurrent connection
        nengo.Connection(ens, ens, transform=0.1)

        # rectified neurons
        ens2 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear())

        # neuron--neuron connection
        nengo.Connection(
            ens, ens2, transform=[[1], [1]], solver=nengo.solvers.LstsqL2(weights=True)
        )

        # sliced output, no synapse
        nengo.Connection(inp, ens2[0], synapse=None, transform=0.5)

        # sliced input, sliced output
        inp2 = nengo.Node([0, 0], label="inp2")
        nengo.Connection(inp2[0], ens2[1])

        nengo.Probe(ens)
        nengo.Probe(ens2)

    with Simulator(net, unroll_simulation=unroll, minibatch_size=minibatch_size) as sim:
        sim.check_gradients(atol=1e-4)


def test_mark_signals():
    with nengo.Network() as net:
        ens0 = nengo.Ensemble(10, 1, neuron_type=nengo.LIF())
        ens1 = nengo.Ensemble(20, 1, neuron_type=nengo.Direct())
        ens2 = nengo.Ensemble(30, 1)
        conn0 = nengo.Connection(ens0, ens1)
        conn1 = nengo.Connection(ens0, ens1, learning_rule_type=nengo.PES())
        conn2 = nengo.Connection(ens0, ens2, learning_rule_type=nengo.Voja())
        nengo.Probe(ens2)

    model = nengo.builder.Model()
    model.build(net)

    tg = tensor_graph.TensorGraph(
        model, None, None, 1, None, utils.NullProgressBar(), None
    )
    tg.mark_signals()

    assert model.sig[ens0]["encoders"].trainable
    assert model.sig[ens1]["encoders"].trainable
    assert not model.sig[ens2]["encoders"].trainable
    assert model.sig[ens0.neurons]["bias"].trainable
    assert model.sig[ens2.neurons]["bias"].trainable
    assert model.sig[conn0]["weights"].trainable
    assert not model.sig[conn1]["weights"].trainable
    assert model.sig[conn2]["weights"].trainable

    trainables = (
        model.sig[ens0]["encoders"],
        model.sig[ens1]["encoders"],
        model.sig[ens0.neurons]["bias"],
        model.sig[ens2.neurons]["bias"],
        model.sig[conn0]["weights"],
        model.sig[conn2]["weights"],
    )

    for op in model.operators:
        for sig in op.all_signals:
            if sig in trainables:
                assert sig.trainable
            else:
                assert not sig.trainable


def test_mark_signals_config():
    with nengo.Network() as net:
        config.configure_settings(trainable=None)
        net.config[nengo.Ensemble].trainable = False

        with nengo.Network():
            # check that object in subnetwork inherits config from parent
            ens0 = nengo.Ensemble(10, 1, label="ens0")

            # check that ens.neurons can be set independent of ens
            net.config[ens0.neurons].trainable = True

            with nengo.Network():
                with nengo.Network():
                    # check that subnetworks can override parent configs
                    config.configure_settings(trainable=True)
                    ens1 = nengo.Ensemble(10, 1, label="ens1")

                    with nengo.Network():
                        # check that subnetworks inherit the trainable settings
                        # from parent networks
                        ens3 = nengo.Ensemble(10, 1, label="ens3")

            # check that instances can be set independent of class
            ens2 = nengo.Ensemble(10, 1, label="ens2")
            net.config[ens2].trainable = True

    model = nengo.builder.Model()
    model.build(net)

    progress = utils.NullProgressBar()

    tg = tensor_graph.TensorGraph(model, None, None, 1, None, progress, None)
    tg.mark_signals()

    assert not model.sig[ens0]["encoders"].trainable
    assert model.sig[ens0.neurons]["bias"].trainable

    assert model.sig[ens1]["encoders"].trainable

    assert model.sig[ens2]["encoders"].trainable

    assert model.sig[ens3]["encoders"].trainable

    # check that learning rule connections can be manually set to True
    with nengo.Network() as net:
        config.configure_settings(trainable=None)

        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        conn0 = nengo.Connection(a, b, learning_rule_type=nengo.PES())
        net.config[conn0].trainable = True

    model = nengo.builder.Model()
    model.build(net)

    tg = tensor_graph.TensorGraph(model, None, None, 1, None, progress, None)
    with pytest.warns(UserWarning):
        tg.mark_signals()

    assert model.sig[conn0]["weights"].trainable

    with nengo.Network() as net:
        config.configure_settings(trainable=None)

        a = nengo.Node([0])
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(a, ens, learning_rule_type=nengo.Voja())
        net.config[nengo.Ensemble].trainable = True

    model = nengo.builder.Model()
    model.build(net)

    tg = tensor_graph.TensorGraph(model, None, None, 1, None, progress, None)
    with pytest.warns(UserWarning):
        tg.mark_signals()

    assert model.sig[ens]["encoders"].trainable

    # check that models with no toplevel work
    sig = nengo.builder.signal.Signal([0])
    op = nengo.builder.operator.Reset(sig, 1)
    model = nengo.builder.Model()
    model.add_op(op)

    tg = tensor_graph.TensorGraph(model, None, None, 1, None, progress, None)
    with pytest.warns(UserWarning):
        tg.mark_signals()

    assert not sig.trainable


@pytest.mark.parametrize("config_planner", (True, False, None))
def test_planner_config(config_planner):
    with nengo.Network() as net:
        if config_planner is not None:
            net.config.configures(nengo.Network)
            if config_planner:
                net.config[nengo.Network].set_param(
                    "planner",
                    nengo.params.Parameter("planner", graph_optimizer.noop_planner),
                )

    model = nengo.builder.Model()
    model.build(net)
    sig = nengo.builder.signal.Signal([1])
    sig2 = nengo.builder.signal.Signal([1])
    sig3 = nengo.builder.signal.Signal([1])
    model.add_op(nengo.builder.operator.DotInc(sig, sig2, sig3))
    model.add_op(nengo.builder.operator.DotInc(sig, sig2, sig3))

    tg = tensor_graph.TensorGraph(
        model, None, None, 1, None, utils.NullProgressBar(), None
    )

    assert len(tg.plan) == (3 if config_planner else 2)


def test_signal_order_deterministic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0])
        ens0 = nengo.Ensemble(10, 1, label="ens")
        ens1 = nengo.Ensemble(10, 1, label="ens")
        nengo.Connection(inp, ens0, synapse=None)
        nengo.Connection(inp, ens1, synapse=None)

    with Simulator(net, seed=seed) as sim1:
        with Simulator(net, seed=seed) as sim2:
            for trainable in ("trainable", "non_trainable", "state"):
                for v, v2 in zip(
                    sim1.tensor_graph.base_arrays_init[trainable].values(),
                    sim2.tensor_graph.base_arrays_init[trainable].values(),
                ):
                    assert all(
                        (x is None and y is None) or np.allclose(x, y)
                        for x, y in zip(v[0], v2[0])
                    )


def test_create_signals():
    # check that floats/ints get split into different arrays

    sigs = [
        dummies.Signal(dtype=np.float32),
        dummies.Signal(dtype=np.float32),
        dummies.Signal(dtype=np.int32),
        dummies.Signal(dtype=np.int32),
    ]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)

    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that floats all get converted to same precision and combined
    sigs = [
        dummies.Signal(dtype=np.float32),
        dummies.Signal(dtype=np.float32),
        dummies.Signal(dtype=np.float64),
        dummies.Signal(dtype=np.float64),
    ]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert np.all([graph.signals[x].dtype == "float32" for x in sigs])
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that ints all get converted to same precision and combined
    sigs = [
        dummies.Signal(dtype=np.int32),
        dummies.Signal(dtype=np.int32),
        dummies.Signal(dtype=np.int64),
        dummies.Signal(dtype=np.int64),
    ]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert np.all([graph.signals[x].dtype == "int32" for x in sigs])
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that different shapes go in different groups
    sigs = [
        dummies.Signal(shape=(10,)),
        dummies.Signal(shape=(5,)),
        dummies.Signal(shape=(10, 1)),
        dummies.Signal(shape=(5, 1)),
    ]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.base_arrays_init["non_trainable"][graph.signals[sigs[0]].key][1] == [
        (10, 10),
        (10, 5),
    ]
    assert graph.base_arrays_init["non_trainable"][graph.signals[sigs[2]].key][1] == [
        (10, 10, 1),
        (10, 5, 1),
    ]
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check trainable
    sigs = [
        dummies.Signal(trainable=True),
        dummies.Signal(trainable=True),
        dummies.Signal(trainable=False),
        dummies.Signal(trainable=False),
    ]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.base_arrays_init["trainable"][graph.signals[sigs[0]].key][1] == [
        (1,),
        (1,),
    ]
    assert graph.base_arrays_init["non_trainable"][graph.signals[sigs[2]].key][1] == [
        (10, 1),
        (10, 1),
    ]
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that scalars get upsized
    sigs = [dummies.Signal(shape=()), dummies.Signal(shape=(4,))]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert list(graph.base_arrays_init["non_trainable"].values())[0][1] == [
        (10, 1),
        (10, 4),
    ]

    # check that boolean signals are handled correctly
    sigs = [dummies.Signal(dtype=np.bool, shape=())]
    plan = [(dummies.Op(reads=sigs),)]
    graph = dummies.TensorGraph(plan, tf.float32, 1)
    graph.create_signals(sigs)
    assert list(graph.base_arrays_init["non_trainable"].values())[0][2] == "bool"


def test_create_signals_views():
    sigs = [
        dummies.Signal(shape=(2, 2), base_shape=(4,)),
        dummies.Signal(shape=(2, 2), base_shape=(4,)),
    ]
    sigs += [sigs[0].base, sigs[1].base]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs[2:])
    assert list(graph.base_arrays_init["non_trainable"].values())[0][1] == [
        (10, 4),
        (10, 4),
    ]
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key
    assert graph.signals[sigs[0]].slices == ((0, 4),)
    assert graph.signals[sigs[1]].slices == ((4, 8),)
    assert graph.signals[sigs[0]].slices == graph.signals[sigs[2]].slices
    assert graph.signals[sigs[1]].slices == graph.signals[sigs[3]].slices

    slice_sig = Signal(shape=(10, 4, 3, 2))[1:8:2]
    slice_sig.trainable = False
    slice_sig.minibatched = True
    slice_sig.base.trainable = False
    slice_sig.base.minibatched = True
    plan = [(dummies.Op(reads=[slice_sig]),)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals([slice_sig.base])
    assert graph.signals[slice_sig].key == graph.signals[slice_sig.base].key
    assert graph.signals[slice_sig].slices == ((1, 2), (3, 4), (5, 6), (7, 8))


def test_create_signals_partition():
    # check that signals are partitioned based on plan
    sigs = [dummies.Signal(), dummies.Signal(), dummies.Signal(), dummies.Signal()]
    plan = [
        tuple(dummies.Op(reads=[x]) for x in sigs[:2]),
        tuple(dummies.Op(reads=[x]) for x in sigs[2:]),
    ]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that signals are partitioned for different read blocks
    plan = [tuple(dummies.Op(reads=[sigs[i], sigs[2 + i]]) for i in range(2))]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that signals are partitioned for different sig types
    plan = [tuple(dummies.Op(reads=[sigs[i]], sets=[sigs[2 + i]]) for i in range(2))]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that resets are ignored
    sigs = [dummies.Signal(), dummies.Signal(), dummies.Signal(), dummies.Signal()]
    plan = [tuple(Reset(x) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert len(graph.base_arrays_init["non_trainable"]) == 4


@pytest.mark.parametrize("use_loop", (True, False))
def test_get_tensor(Simulator, use_loop):
    with nengo.Network() as net:
        config.configure_settings(use_loop=use_loop)

        a = nengo.Node([1])
        b = nengo.Ensemble(10, 1)
        c = nengo.Connection(
            a, b.neurons, transform=np.arange(10)[:, None], synapse=None
        )
        p = nengo.Probe(c)

        # build a signal probe so that the indices get loaded into the sim
        # (checks that the indices reloading works properly)
        nengo.Probe(c, "weights")

    kwargs = dict() if use_loop else dict(unroll_simulation=10)
    with Simulator(net, **kwargs) as sim:
        tensor = sim.tensor_graph.get_tensor(sim.model.sig[c]["weights"])

        assert np.allclose(tf.keras.backend.get_value(tensor), np.arange(10)[:, None])

        sim.run_steps(10)
        assert np.allclose(sim.data[p], np.arange(10)[None, :])


@pytest.mark.eager_only
@pytest.mark.parametrize("trainable", (True, False))
def test_build(trainable, rng):
    sigs = [
        dummies.Signal(
            shape=(2, 1), dtype="float32", initial_value=0, trainable=trainable
        ),
        dummies.Signal(
            shape=(3, 1),
            dtype="float32",
            initial_value=np.zeros((3, 1)),
            trainable=trainable,
        ),
        dummies.Signal(
            shape=(4, 1), dtype="float32", initial_value=1, trainable=trainable
        ),
        dummies.Signal(
            shape=(5, 1),
            dtype="float32",
            initial_value=np.ones((5, 1)),
            trainable=trainable,
        ),
        dummies.Signal(
            shape=(6, 1),
            dtype="float32",
            initial_value=rng.uniform(size=(6, 1)),
            trainable=trainable,
        ),
        dummies.Signal(
            shape=(7, 1),
            dtype="float32",
            initial_value=rng.uniform(size=(7, 1)),
            trainable=trainable,
        ),
    ]

    plan = [
        tuple(dummies.Op(reads=[x]) for x in sigs[:2]),
        tuple(dummies.Op(reads=[x]) for x in sigs[2:4]),
        tuple(dummies.Op(reads=[x]) for x in sigs[4:]),
    ]

    graph = dummies.TensorGraph(plan=plan, dtype="float32", minibatch_size=16)
    graph.create_signals(sigs)
    graph.build()

    if trainable:
        assert len(graph.trainable_weights) == 3
        assert len(graph.non_trainable_weights) == 0
    else:
        assert len(graph.trainable_weights) == 0
        assert len(graph.non_trainable_weights) == 3

    init0 = graph.weights[0].numpy()
    assert init0.shape == (5, 1) if trainable else (16, 5, 1)
    assert np.allclose(init0, 0)

    init1 = graph.weights[1].numpy()
    assert init1.shape == (9, 1) if trainable else (16, 9, 1)
    assert np.allclose(init1, 1)

    init2 = graph.weights[2].numpy()
    if trainable:
        assert init2.shape == (13, 1)
        assert np.allclose(init2[:6], sigs[4].initial_value)
        assert np.allclose(init2[6:], sigs[5].initial_value)
    else:
        assert init2.shape == (16, 13, 1)
        assert np.allclose(init2[:, :6], sigs[4].initial_value)
        assert np.allclose(init2[:, 6:], sigs[5].initial_value)


@pytest.mark.parametrize("use_loop", (True, False))
def test_conditional_update(Simulator, use_loop, caplog):
    caplog.set_level(logging.INFO)

    with nengo.Network() as net:
        config.configure_settings(stateful=False, use_loop=use_loop)

        a = nengo.Ensemble(10, 1)
        b = nengo.Node(size_in=1)
        conn = nengo.Connection(a, b)

    with Simulator(net):
        pass

    assert "Number of state updates: 0" in caplog.text
    caplog.clear()

    conn.learning_rule_type = nengo.PES()

    with Simulator(net):
        pass

    assert "Number of state updates: 1" in caplog.text
    caplog.clear()

    with net:
        config.configure_settings(trainable=True)

    with Simulator(net):
        pass

    assert "Number of state updates: 1" in caplog.text


def test_unsupported_op_error():
    class MyOp(dummies.Op):  # pylint: disable=abstract-method
        pass

    model = nengo.builder.Model()
    model.add_op(MyOp())
    with pytest.raises(BuildError, match="No registered builder"):
        tensor_graph.TensorGraph(model, None, None, None, None, None, None)


@pytest.mark.skipif(
    sys.version_info < (3, 6, 0), reason="order is not deterministic in python<3.6"
)
@pytest.mark.parametrize(
    "planner", ("greedy_planner", "tree_planner", "transitive_planner", "noop_planner")
)
def test_deterministic_order(planner, tmp_path):
    if version.parse(nengo.__version__) <= version.parse("3.1.0") and planner in (
        "greedy_planner",
        "noop_planner",
    ):
        # we could make this work by backporting the deterministic toposort from
        # nengo>3.1.0, but that doesn't seem worth it to get these two niche planners
        # to be deterministic
        pytest.skip(f"'{planner}' is nondeterministic in Nengo<=3.1.0")

    code = textwrap.dedent(
        f"""
    import nengo
    import nengo_dl

    with nengo.Network(seed=0) as net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.{planner})

        # use ensemblearrays as they create a lot of parallel ops
        ens0 = nengo.networks.EnsembleArray(1, 10)
        ens1 = nengo.networks.EnsembleArray(1, 10)
        nengo.Connection(ens0.output, ens1.input)
        nengo.Probe(ens1.output)

    model = nengo_dl.builder.NengoModel(
        dt=0.001,
        builder=nengo_dl.builder.NengoBuilder(),
        fail_fast=False,
    )
    model.build(net)

    tensor_graph = nengo_dl.tensor_graph.TensorGraph(
        model=model,
        dt=0.001,
        unroll_simulation=1,
        minibatch_size=1,
        device=None,
        progress=nengo_dl.utils.NullProgressBar(),
        seed=0,
    )

    plan = tensor_graph.plan
    sigs = tensor_graph.signals.sig_map

    for ops in plan:
        print(len(ops))
        for op in ops:
            print(type(op))
            for s in op.all_signals:
                print(s._name)
                print(s.shape)
                print(s.initial_value)

    for s in sigs:
        print(s._name)
        print(s.shape)
        print(s.initial_value)
    """
    )
    tmp_path = tmp_path / "test.py"
    tmp_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()

    env["PYTHONHASHSEED"] = "0"
    output0 = subprocess.run(
        [sys.executable, str(tmp_path)],
        stdout=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        check=True,
    )

    env["PYTHONHASHSEED"] = "1"
    output1 = subprocess.run(
        [sys.executable, str(tmp_path)],
        stdout=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        check=True,
    )

    assert output0.stdout == output1.stdout
