# pylint: disable=missing-docstring

import nengo
from nengo.builder.operator import Reset
from nengo.exceptions import ValidationError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import tensor_graph, utils, graph_optimizer, config, objectives
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
        nengo.Connection(ens, ens2, transform=[[1], [1]],
                         solver=nengo.solvers.LstsqL2(weights=True))

        # sliced output, no synapse
        nengo.Connection(inp, ens2[0], synapse=None, transform=0.5)

        # sliced input, sliced output
        inp2 = nengo.Node([0, 0], label="inp2")
        nengo.Connection(inp2[0], ens2[1])

        nengo.Probe(ens)
        nengo.Probe(ens2)

    with Simulator(net, unroll_simulation=unroll,
                   minibatch_size=minibatch_size) as sim:
        sim.check_gradients(atol=1e-4)


def test_build_outputs(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        p = nengo.Probe(inp)

    with Simulator(net) as sim:
        # check that the output caching works
        assert (sim.tensor_graph.build_outputs({p: objectives.mse})[0] is
                sim.tensor_graph.build_outputs({p: objectives.mse})[0])

        def loss(x):
            return x

        assert (sim.tensor_graph.build_outputs({p: loss})[0] is
                sim.tensor_graph.build_outputs({p: loss})[0])

        # check function argument counting
        def loss3(x, y, z):
            return x

        with pytest.raises(ValidationError):
            sim.tensor_graph.build_outputs({p: loss3})

        def loss3b(x, y, z=None):
            return x

        # no error for extra keyword arguments
        sim.tensor_graph.build_outputs({p: loss3b})

        # arg parsing works with callable class
        class CallableLoss:
            def __call__(self, outputs, targets):
                return outputs
        sim.tensor_graph.build_outputs({p: CallableLoss()})

        # arg parsing works with class methods
        class MethodLoss:
            def loss(self, outputs, targets):
                return outputs
        sim.tensor_graph.build_outputs({p: MethodLoss().loss})

        # validation error for invalid output type
        with pytest.raises(ValidationError):
            sim.tensor_graph.build_outputs({p: 1})


@pytest.mark.training
def test_build_optimizer(Simulator):
    with nengo.Network() as net:
        inp = nengo.Node([0])
        ens = nengo.Ensemble(10, 1, neuron_type=nengo.Sigmoid())
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens)

    # check optimizer caching
    with Simulator(net) as sim:
        opt = tf.train.GradientDescentOptimizer(0)
        assert (
            sim.tensor_graph.build_optimizer_func(opt, {p: objectives.mse}) is
            sim.tensor_graph.build_optimizer_func(opt, {p: objectives.mse}))

    # error when no trainable elements
    with nengo.Network() as net:
        inp = nengo.Node([0])
        p = nengo.Probe(inp)

    with Simulator(net) as sim:
        with pytest.raises(ValueError):
            sim.tensor_graph.build_outputs(
                {p: sim.tensor_graph.build_optimizer_func(
                    opt, {p: objectives.mse})})

    # capturing variables from nested loss function
    def loss(x):
        return abs(
            tf.get_variable("two", initializer=tf.constant_initializer(2.0),
                            shape=(), dtype=x.dtype) - x)

    net, _, p = dummies.linear_net()

    with Simulator(net) as sim:
        sim.train(5, tf.train.GradientDescentOptimizer(0.1),
                  objective={p: loss}, n_epochs=10)
        sim.step()
        assert np.allclose(sim.data[p], 2)


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

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  utils.NullProgressBar())
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
        model.sig[ens0]["encoders"], model.sig[ens1]["encoders"],
        model.sig[ens0.neurons]["bias"], model.sig[ens2.neurons]["bias"],
        model.sig[conn0]["weights"], model.sig[conn2]["weights"])

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
                with nengo.Network() as subnet:
                    # check that subnetworks can override parent configs
                    # net.config[nengo.Ensemble].trainable = True
                    net.config[subnet].trainable = True
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

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  progress)
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

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  progress)
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

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  progress)
    with pytest.warns(UserWarning):
        tg.mark_signals()

    assert model.sig[ens]["encoders"].trainable

    # check that models with no toplevel work
    sig = nengo.builder.signal.Signal([0])
    op = nengo.builder.operator.Reset(sig, 1)
    model = nengo.builder.Model()
    model.add_op(op)

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  progress)
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
                    "planner", nengo.params.Parameter(
                        "planner", graph_optimizer.noop_planner))

    model = nengo.builder.Model()
    model.build(net)
    sig = nengo.builder.signal.Signal([1])
    sig2 = nengo.builder.signal.Signal([1])
    sig3 = nengo.builder.signal.Signal([1])
    model.add_op(nengo.builder.operator.DotInc(sig, sig2, sig3))
    model.add_op(nengo.builder.operator.DotInc(sig, sig2, sig3))

    tg = tensor_graph.TensorGraph(model, None, None, tf.float32, 1, None,
                                  utils.NullProgressBar())

    assert len(tg.plan) == (2 if config_planner else 1)


def test_signal_order_deterministic(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([0])
        ens0 = nengo.Ensemble(10, 1, label="ens")
        ens1 = nengo.Ensemble(10, 1, label="ens")
        nengo.Connection(inp, ens0, synapse=None)
        nengo.Connection(inp, ens1, synapse=None)

    with Simulator(net, seed=seed) as sim1:
        pass

    with Simulator(net, seed=seed) as sim2:
        for v, v2 in zip(
                sim1.tensor_graph.base_arrays_init.values(),
                sim2.tensor_graph.base_arrays_init.values()):
            assert np.allclose(v[0], v2[0])


def test_create_signals():
    # check that floats/ints get split into different arrays

    sigs = [dummies.Signal(dtype=np.float32), dummies.Signal(dtype=np.float32),
            dummies.Signal(dtype=np.int32), dummies.Signal(dtype=np.int32)]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)

    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that floats all get converted to same precision and combined
    sigs = [dummies.Signal(dtype=np.float32), dummies.Signal(dtype=np.float32),
            dummies.Signal(dtype=np.float64), dummies.Signal(dtype=np.float64)]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert np.all([graph.signals[x].dtype == np.float32 for x in sigs])
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that ints all get converted to same precision and combined
    sigs = [dummies.Signal(dtype=np.int32), dummies.Signal(dtype=np.int32),
            dummies.Signal(dtype=np.int64), dummies.Signal(dtype=np.int64)]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert np.all([graph.signals[x].dtype == np.int32 for x in sigs])
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that different shapes go in different groups
    sigs = [dummies.Signal(shape=(10,)), dummies.Signal(shape=(5,)),
            dummies.Signal(shape=(10, 1)), dummies.Signal(shape=(5, 1))]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.base_arrays_init[graph.signals[sigs[0]].key][0].shape == (
        15, 10)
    assert graph.base_arrays_init[graph.signals[sigs[2]].key][0].shape == (
        15, 1, 10)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check trainable
    sigs = [dummies.Signal(trainable=True), dummies.Signal(trainable=True),
            dummies.Signal(trainable=False), dummies.Signal(trainable=False)]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.base_arrays_init[graph.signals[sigs[0]].key][0].shape == (2,)
    assert graph.base_arrays_init[graph.signals[sigs[2]].key][0].shape == (
        2, 10)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that scalars get upsized
    sigs = [dummies.Signal(shape=()), dummies.Signal(shape=(4,))]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert list(graph.base_arrays_init.values())[0][0].shape == (5, 10)

    # check that boolean signals are handled correctly
    sigs = [dummies.Signal(dtype=np.bool, shape=())]
    plan = [(dummies.Op(reads=sigs),)]
    graph = dummies.TensorGraph(plan, tf.float32, 1)
    graph.create_signals(sigs)
    assert list(graph.base_arrays_init.values())[0][0].dtype == np.bool


def test_create_signals_views():
    sigs = [dummies.Signal(shape=(2, 2), base_shape=(4,)),
            dummies.Signal(shape=(2, 2), base_shape=(4,))]
    sigs += [sigs[0].base, sigs[1].base]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs[2:])
    assert list(graph.base_arrays_init.values())[0][0].shape == (8, 10)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key == graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key
    assert np.all(graph.signals[sigs[0]].indices == (0, 1, 2, 3))
    assert np.all(graph.signals[sigs[1]].indices == (4, 5, 6, 7))
    assert np.all(graph.signals[sigs[0]].indices ==
                  graph.signals[sigs[2]].indices)
    assert np.all(graph.signals[sigs[1]].indices ==
                  graph.signals[sigs[3]].indices)


def test_create_signals_partition():
    # check that signals are partitioned based on plan
    sigs = [dummies.Signal(), dummies.Signal(),
            dummies.Signal(), dummies.Signal()]
    plan = [tuple(dummies.Op(reads=[x]) for x in sigs[:2]),
            tuple(dummies.Op(reads=[x]) for x in sigs[2:])]
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
    plan = [tuple(dummies.Op(reads=[sigs[i]], sets=[sigs[2 + i]])
                  for i in range(2))]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert graph.signals[sigs[0]].key == graph.signals[sigs[1]].key
    assert graph.signals[sigs[1]].key != graph.signals[sigs[2]].key
    assert graph.signals[sigs[2]].key == graph.signals[sigs[3]].key

    # check that resets are ignored
    sigs = [dummies.Signal(), dummies.Signal(), dummies.Signal(),
            dummies.Signal()]
    plan = [tuple(Reset(x) for x in sigs)]
    graph = dummies.TensorGraph(plan, tf.float32, 10)
    graph.create_signals(sigs)
    assert len(graph.base_arrays_init) == 4


def test_get_tensor(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Ensemble(10, 1)
        c = nengo.Connection(a, b.neurons, transform=np.arange(10)[:, None],
                             synapse=None)
        p = nengo.Probe(c)

        # build a signal probe so that the indices get loaded into the sim
        # (checks that the indices reloading works properly)
        nengo.Probe(c, "weights")

    with Simulator(net) as sim:
        tensor = sim.tensor_graph.get_tensor(sim.model.sig[c]["weights"])

        assert np.allclose(sim.sess.run(tensor), np.arange(10)[:, None])

        sim.run_steps(10)
        assert np.allclose(sim.data[p], np.arange(10)[None, :])
