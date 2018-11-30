"""
Benchmark networks and utilities for evaluating NengoDL's performance.
"""

import inspect
import itertools
import os
import random
import time

import click
import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow as tf

import nengo_dl


def cconv(dimensions, neurons_per_d, neuron_type):
    """
    Circular convolution (EnsembleArray) benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="cconv", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.cconv = nengo.networks.CircularConvolution(
            neurons_per_d, dimensions)

        net.inp_a = nengo.Node([0] * dimensions)
        net.inp_b = nengo.Node([1] * dimensions)
        nengo.Connection(net.inp_a, net.cconv.A)
        nengo.Connection(net.inp_b, net.cconv.B)

        net.p = nengo.Probe(net.cconv.output)

    return net


def integrator(dimensions, neurons_per_d, neuron_type):
    """
    Single integrator ensemble benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="integrator", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.integ = nengo.networks.EnsembleArray(neurons_per_d, dimensions)
        nengo.Connection(net.integ.output, net.integ.input, synapse=0.01)

        net.inp = nengo.Node([0] * dimensions)
        nengo.Connection(net.inp, net.integ.input, transform=0.01)

        net.p = nengo.Probe(net.integ.output)

    return net


def pes(dimensions, neurons_per_d, neuron_type):
    """
    PES learning rule benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="pes", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.inp = nengo.Node([1] * dimensions)
        net.pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        net.post = nengo.Node(size_in=dimensions)

        nengo.Connection(net.inp, net.pre)

        conn = nengo.Connection(
            net.pre, net.post, learning_rule_type=nengo.PES())

        nengo.Connection(net.post, conn.learning_rule, transform=-1)
        nengo.Connection(net.inp, conn.learning_rule)

        net.p = nengo.Probe(net.post)

    return net


def basal_ganglia(dimensions, neurons_per_d, neuron_type):
    """
    Basal ganglia network benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="basal_ganglia", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type

        net.inp = nengo.Node([1] * dimensions)
        net.bg = nengo.networks.BasalGanglia(dimensions, neurons_per_d)
        nengo.Connection(net.inp, net.bg.input)
        net.p = nengo.Probe(net.bg.output)

    return net


def mnist(use_tensor_layer=True):
    """
    A network designed to stress-test tensor layers (based on mnist net).

    Parameters
    ----------
    use_tensor_layer : bool
        If True, use individual tensor_layers to build the network, as opposed
        to a single TensorNode containing all layers.

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network() as net:
        # create node to feed in images
        net.inp = nengo.Node(np.ones(28 * 28))

        if use_tensor_layer:
            nengo_nl = nengo.RectifiedLinear()

            ensemble_params = dict(max_rates=nengo.dists.Choice([100]),
                                   intercepts=nengo.dists.Choice([0]))
            amplitude = 1
            synapse = None

            x = nengo_dl.tensor_layer(
                net.inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
                kernel_size=3
            )
            x = nengo_dl.tensor_layer(x, nengo_nl,
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(26, 26, 32),
                transform=amplitude, filters=32, kernel_size=3
            )
            x = nengo_dl.tensor_layer(x, nengo_nl,
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(24, 24, 32),
                synapse=synapse, transform=amplitude, pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(
                x, tf.layers.dense, units=128
            )
            x = nengo_dl.tensor_layer(x, nengo_nl,
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.4,
                                      transform=amplitude)

            x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)
        else:
            nl = tf.nn.relu

            # def softlif_layer(x, sigma=1, tau_ref=0.002, tau_rc=0.02,
            #                   amplitude=1):
            #     # x -= 1
            #     z = tf.nn.softplus(x / sigma) * sigma
            #     z += 1e-10
            #     rates = amplitude / (tau_ref + tau_rc * tf.log1p(1 / z))
            #     return rates

            @nengo_dl.reshaped((28, 28, 1))
            def mnist_node(_, x):  # pragma: no cover
                x = tf.layers.conv2d(x, filters=32, kernel_size=3,
                                     activation=nl)
                x = tf.layers.conv2d(x, filters=32, kernel_size=3,
                                     activation=nl)
                x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, 128, activation=nl)
                x = tf.layers.dropout(x, rate=0.4)
                x = tf.layers.dense(x, 10)

                return x

            node = nengo_dl.TensorNode(mnist_node, size_in=28 * 28,
                                       size_out=10)
            x = node
            nengo.Connection(net.inp, node, synapse=None)

        net.p = nengo.Probe(x)

    return net


def spaun(dimensions):
    """
    Builds the Spaun network from [1]_

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values

    Returns
    -------
    net : `nengo.Network`
        benchmark network

    References
    ----------
    .. [1] Chris Eliasmith, Terrence C. Stewart, Xuan Choo, Trevor Bekolay,
       Travis DeWolf, Yichuan Tang, and Daniel Rasmussen (2012). A large-scale
       model of the functioning brain. Science, 338:1202-1205.

    Notes
    -----
    This network needs to be installed via

    .. code-block:: bash

        pip install git+https://github.com/drasmuss/spaun2.0.git
    """

    from _spaun.configurator import cfg
    from _spaun.vocabulator import vocab
    from _spaun.experimenter import experiment
    from _spaun.modules.stim import stim_data
    from _spaun.modules.vision import vis_data
    from _spaun.modules.motor import mtr_data
    from _spaun.spaun_main import Spaun

    vocab.sp_dim = dimensions
    cfg.mtr_arm_type = None

    cfg.set_seed(1)
    experiment.initialize(
        "A", stim_data.get_image_ind, stim_data.get_image_label,
        cfg.mtr_est_digit_response_time, "", cfg.rng)
    vocab.initialize(
        stim_data.stim_SP_labels, experiment.num_learn_actions, cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    return Spaun()


def random_network(dimensions, neurons_per_d, neuron_type, n_ensembles,
                   connections_per_ensemble, seed=0):
    """
    Basal ganglia network benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type
    n_ensembles : int
        Number of ensembles in the network
    connections_per_ensemble : int
        Outgoing connections from each ensemble

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    random.seed(seed)
    with nengo.Network(label="random", seed=seed) as net:
        net.inp = nengo.Node([0] * dimensions)
        net.out = nengo.Node(size_in=dimensions)
        net.p = nengo.Probe(net.out)
        ensembles = [
            nengo.Ensemble(neurons_per_d * dimensions, dimensions,
                           neuron_type=neuron_type)
            for _ in range(n_ensembles)]
        dec = np.ones((neurons_per_d * dimensions, dimensions))
        for ens in net.ensembles:
            # add a connection to input and output node, so we never have
            # any "orphan" ensembles
            nengo.Connection(net.inp, ens)
            nengo.Connection(ens, net.out, solver=nengo.solvers.NoSolver(dec))

            posts = random.sample(ensembles, connections_per_ensemble)
            for post in posts:
                nengo.Connection(ens, post, solver=nengo.solvers.NoSolver(dec))

    return net


def run_profile(net, train=False, n_steps=150, do_profile=True, **kwargs):
    """
    Run profiler on a benchmark network.

    Parameters
    ----------
    net : `~nengo.Network`
        The nengo Network to be profiled.
    train : bool
        If True, profile the ``sim.train`` function. Otherwise, profile the
        ``sim.run`` function.
    n_steps : int
        The number of timesteps to run the simulation.
    do_profile : bool
        Whether or not to run profiling

    Notes
    -----
    kwargs will be passed on to `.Simulator`
    """

    with net:
        nengo_dl.configure_settings(inference_only=not train)

    with nengo_dl.Simulator(net, **kwargs) as sim:
        # note: we run a few times to try to eliminate startup overhead (only
        # the data from the last run will be kept)
        if train:
            opt = tf.train.GradientDescentOptimizer(0.001)
            x = np.random.randn(sim.minibatch_size, n_steps, net.inp.size_out)
            y = np.random.randn(sim.minibatch_size, n_steps, net.p.size_in)

            for _ in range(2):
                sim.train({net.inp: x, net.p: y}, optimizer=opt, n_epochs=1,
                          profile=do_profile)

            start = time.time()
            sim.train({net.inp: x, net.p: y}, optimizer=opt, n_epochs=1,
                      profile=do_profile)
            exec_time = time.time() - start
            print("Execution time:", exec_time)

        else:
            for _ in range(2):
                sim.run_steps(n_steps, profile=do_profile)

            start = time.time()
            sim.run_steps(n_steps, profile=do_profile)
            exec_time = time.time() - start
            print("Execution time:", exec_time)

    return exec_time


@click.group(chain=True)
def main():
    """Command-line interface for benchmarks."""


@main.command()
@click.pass_obj
@click.option("--benchmark", default="cconv", help="Name of benchmark network")
@click.option("--dimensions", default=128, help="Number of dimensions")
@click.option("--neurons_per_d", default=64, help="Neurons per dimension")
@click.option("--neuron_type", default="RectifiedLinear",
              help="Nengo neuron model")
@click.option("--kwarg", type=str, multiple=True,
              help="Arbitrary kwarg to pass to benchmark network (key=value)")
def build(obj, benchmark, dimensions, neurons_per_d, neuron_type,
          kwarg):
    """Builds one of the benchmark networks"""

    # get benchmark network by name
    benchmark = globals()[benchmark]

    # get the neuron type object from string class name
    try:
        neuron_type = getattr(nengo, neuron_type)()
    except AttributeError:
        neuron_type = getattr(nengo_dl, neuron_type)()

    # set up kwargs
    kwargs = dict((k, int(v)) for k, v in [a.split("=") for a in kwarg])

    # add the special cli kwargs if applicable; note we could just do
    # everything through --kwarg, but it is convenient to have a
    # direct option for the common arguments
    params = inspect.signature(benchmark).parameters
    for kw in ("benchmark", "dimensions", "neurons_per_d", "neuron_type"):
        if kw in params:
            kwargs[kw] = locals()[kw]

    # build benchmark and add to context for chaining
    print("Building %s with %s" % (
        nengo_dl.utils.function_name(benchmark, sanitize=False), kwargs))

    obj["net"] = benchmark(**kwargs)


@main.command()
@click.pass_obj
@click.option("--train/--no-train", default=False,
              help="Whether to profile training (as opposed to running) "
                   "the network")
@click.option("--n_steps", default=150,
              help="Number of steps for which to run the simulation")
@click.option("--batch_size", default=1, help="Number of inputs to the model")
@click.option("--device", default="/gpu:0",
              help="TensorFlow device on which to run the simulation")
@click.option("--unroll", default=25,
              help="Number of steps for which to unroll the simulation")
@click.option("--time-only", is_flag=True, default=False,
              help="Only count total time, rather than profiling internals")
def profile(obj, train, n_steps, batch_size, device, unroll, time_only):
    """Runs profiling on a network (call after 'build')"""

    if "net" not in obj:
        raise ValueError("Must call `build` before `profile`")

    obj["time"] = run_profile(
        obj["net"], do_profile=not time_only, train=train, n_steps=n_steps,
        minibatch_size=batch_size, device=device, unroll_simulation=unroll)


@main.command()
def matmul_vs_reduce():  # pragma: no cover
    """
    Compares two different approaches to batched matrix multiplication
    (tf.matmul vs tf.multiply+tf.reduce_sum).

    This is relevant for figuring out which approach is more efficient
    on a given system for different matrix shapes (determining which method
    we use in DotIncBuilder).
    """

    # a_shape = (n_ops, s0, s1, 1)
    # x_shape = (n_ops, 1, s1, mini)
    # for matmul we omit the 1 dimensions

    a_c = tf.placeholder(tf.float64, shape=(None, None, None), name="a_c")
    x_c = tf.placeholder(tf.float64, shape=(None, None, None), name="b_c")
    a_d = tf.placeholder(tf.float64, shape=(None, None, None, 1), name="a_d")
    x_d = tf.placeholder(tf.float64, shape=(None, 1, None, None), name="b_d")
    c = tf.matmul(a_c, x_c)
    d = tf.reduce_sum(tf.multiply(a_d, x_d), axis=-2)

    reps = 100
    n_ops_range = [1, 4, 8, 16, 32, 64]
    mini_range = [1, 16, 32, 64, 128]
    s0_range = [1, 64, 128, 192, 256]
    s1_range = [1, 64, 128, 192, 256]

    matmul_times = np.zeros((len(n_ops_range), len(mini_range), len(s0_range),
                             len(s1_range)))
    reduce_times = np.zeros_like(matmul_times)

    params = itertools.product(
        enumerate(n_ops_range), enumerate(mini_range), enumerate(s0_range),
        enumerate(s1_range))

    with tf.Session() as sess:
        for (i, n_ops), (j, mini), (k, s0), (l, s1) in params:
            print(n_ops, mini, s0, s1)

            a_val = np.random.randn(n_ops, s0, s1, 1)
            x_val = np.random.randn(n_ops, 1, s1, mini)

            for r in range(reps + 3):
                if r == 3:
                    start = time.time()
                c_val = sess.run(c, feed_dict={a_c: a_val[..., 0],
                                               x_c: x_val[:, 0]})
            matmul_times[i, j, k, l] = (time.time() - start) / reps

            for r in range(reps + 3):
                if r == 3:
                    start = time.time()
                d_val = sess.run(d, feed_dict={a_d: a_val, x_d: x_val})
            reduce_times[i, j, k, l] = (time.time() - start) / reps

            assert np.allclose(c_val, d_val)

    fig, ax = plt.subplots(len(n_ops_range), len(mini_range), sharex=True,
                           sharey=True)

    X, Y = np.meshgrid(s0_range, s1_range)
    Z = matmul_times - reduce_times
    v = np.sort(np.concatenate((np.linspace(np.min(Z), np.max(Z), 10), [0])))
    for i, n_ops in enumerate(n_ops_range):
        for j, mini in enumerate(mini_range):
            cs = ax[i][j].contourf(X, Y, Z[i, j], v)
            if i == 0:
                ax[i][j].set_title("mini %d" % mini)
            if j == 0:
                ax[i][j].set_ylabel("ops %d" % n_ops)

    DATA_DIR = os.path.join(os.path.dirname(nengo_dl.__file__), "..", "data")
    np.savez(os.path.join(DATA_DIR, "matmul_benchmarks"), n_ops_range,
             mini_range, s0_range, s1_range, Z)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax)

    plt.show()


@main.command()
@click.option("--device", default="/gpu:0",
              help="TensorFlow device on which to run benchmarks")
def performance_samples(device):  # pragma: no cover
    """
    Run a brief sample of the benchmarks to check overall performance.

    This is mainly used to quickly check that there haven't been any unexpected
    performance regressions.
    """

    # TODO: automatically run some basic performance tests during CI

    default_kwargs = {"n_steps": 1000, "device": device,
                      "unroll_simulation": 25,
                      "progress_bar": False, "do_profile": False}

    print("cconv + relu")
    net = cconv(128, 64, nengo.RectifiedLinear())
    run_profile(net, minibatch_size=64, **default_kwargs)

    print("cconv + lif")
    net = cconv(128, 64, nengo.LIF())
    run_profile(net, minibatch_size=64, **default_kwargs)

    print("integrator training + relu")
    net = integrator(128, 32, nengo.RectifiedLinear())
    run_profile(net, minibatch_size=64, train=True, **default_kwargs)

    print("integrator training + lif")
    net = integrator(128, 32, nengo.LIF())
    run_profile(net, minibatch_size=64, train=True, **default_kwargs)

    print("random")
    net = random_network(128, 64, nengo.RectifiedLinear(), n_ensembles=50,
                         connections_per_ensemble=5, seed=0)
    run_profile(net, **default_kwargs)

    print("spaun")
    net = spaun(1)
    run_profile(net, **default_kwargs)

    # example benchmark data
    # CPU: 4.00GHz Intel Core i7-6700K
    # GPU: NVIDIA GeForce GTX 980 Ti
    # TensorFlow version: 1.10.0
    # Nengo version: 2.8.0
    # NengoDL version: 1.2.0

    # cconv + relu
    # Execution time: 1.0098507404327393

    # cconv + lif
    # Execution time: 2.074916362762451

    # integrator training + relu
    # Execution time: 1.8205187320709229

    # integrator training + lif
    # Execution time: 2.669060707092285

    # random
    # Execution time: 21.686023235321045

    # spaun
    # Execution time: 9.540623426437378


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
