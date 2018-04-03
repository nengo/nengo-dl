import os
import time

import matplotlib.pyplot as plt
import nengo
import nengo_ocl
import numpy as np
import tensorflow as tf

import nengo_dl
from nengo_dl import DATA_DIR


def cconv(dimensions, neurons_per_d, neuron_type):
    """Circular convolution (EnsembleArray) benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    nengo.Network
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
    """Single integrator ensemble benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="integrator", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.integ = nengo.networks.Integrator(0.1, neurons_per_d * dimensions,
                                              dimensions)

        net.inp = nengo.Node([0] * dimensions)
        nengo.Connection(net.inp, net.integ.input)

        net.p = nengo.Probe(net.integ.ensemble)

    return net


def pes(dimensions, neurons_per_d, neuron_type):
    """PES learning rule benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="pes", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.inp = nengo.Node([1] * dimensions)
        net.pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        net.post = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        net.err = nengo.Node(size_in=dimensions)
        nengo.Connection(net.inp, net.pre)
        nengo.Connection(net.post, net.err, transform=-1)
        nengo.Connection(net.inp, net.err)

        conn = nengo.Connection(
            net.pre, net.post, learning_rule_type=nengo.PES())
        nengo.Connection(net.err, conn.learning_rule)

        net.p = nengo.Probe(net.post)

    return net


def compare_backends(raw=False):
    """Compare the run time of different backends across benchmarks and
    a range of parameters.

    Parameters
    ----------
    raw : bool
        If True, run the benchmarks to collect data, otherwise load data from
        file
    """

    benchmarks = [pes, integrator, cconv]
    n_range = [32]
    d_range = [64, 128, 256]
    neuron_types = [nengo.RectifiedLinear, nengo.LIF]
    backends = [nengo_dl, nengo_ocl, nengo]

    if raw:
        data = np.zeros((len(benchmarks), len(n_range), len(d_range),
                         len(neuron_types), len(backends)))

        for i, bench in enumerate(benchmarks):
            for j, neurons in enumerate(n_range):
                for k, dimensions in enumerate(d_range):
                    for l, neuron_type in enumerate(neuron_types):
                        print("-" * 30)
                        print(bench, neurons, dimensions, neuron_type)

                        net = bench(dimensions, neurons, neuron_type())
                        model = nengo.builder.Model()
                        model.build(net)

                        for m, backend in enumerate(backends):
                            print(backend)

                            if backend is None:
                                continue
                            elif backend == nengo_dl:
                                kwargs = {"unroll_simulation": 25,
                                          "minibatch_size": None,
                                          "device": "/gpu:0",
                                          "dtype": tf.float32,
                                          }
                            elif backend == nengo:
                                kwargs = {"progress_bar": None,
                                          "optimize": True}
                            elif backend == nengo_ocl:
                                kwargs = {"progress_bar": None}

                            try:
                                # with backend.Simulator(net, **kwargs) as sim:
                                with backend.Simulator(None, model=model,
                                                       **kwargs) as sim:
                                    # run once to eliminate startup overhead
                                    sim.run(0.1)

                                    start = time.time()
                                    sim.run(5)
                                    # reps = 1 if backend == nengo_dl else 50
                                    # for r in range(reps):
                                    #     sim.run(1.0)
                                    data[i, j, k, l, m] = time.time() - start
                                    print("time", data[i, j, k, l, m])
                            except Exception as e:
                                print(backend, "CRASHED")
                                print(e)
                                data[i, j, k, l, m] = np.nan

                            # if backend == nengo:
                            #     canonical = sim.data[net.p]
                            # else:
                            #     assert np.allclose(
                            #         canonical, sim.data[net.p], atol=1e-3)

        np.savez("%s/benchmark_data.npz" % DATA_DIR, data)
    else:
        data = np.load("%s/benchmark_data.npz" % DATA_DIR)["arr_0"]

    bench_names = ["pes", "integrator", "cconv"]
    neuron_names = ["relu", "lif"]

    for j in range(len(neuron_types)):
        f, axes = plt.subplots(1, 3)
        for i in range(len(benchmarks)):
            plt.figure()
            plt.title("%s (%s)" % (bench_names[i], neuron_names[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 2])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo")

            plt.figure()
            plt.title("%s (%s)" % (bench_names[i], neuron_names[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 1])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo_ocl")

            axes[i].set_title("%s (%s)" % (bench_names[i], neuron_names[j]))
            axes[i].plot(d_range, data[i, 0, :, j, :])
            axes[i].set_xlabel("dimensions")
            axes[i].set_ylabel("seconds")
            axes[i].legend(["nengo_dl", "nengo_ocl", "nengo"])
            axes[i].set_ylim([0, 100])

    plt.show()


def profile(train=False):
    """Run profiler on one of the benchmarks."""

    net = pes(128, 32, nengo.LIFRate())
    with nengo_dl.Simulator(net, tensorboard=None, unroll_simulation=50,
                            device="/gpu:0") as sim:

        # note: we run a few times to try to eliminate startup overhead (only
        # the data from the last run will be kept)
        if train:
            opt = tf.train.GradientDescentOptimizer(0.001)
            x = np.random.randn(1, 100, net.inp.size_out)
            y = np.random.randn(1, 100, net.p.size_in)
            for _ in range(3):
                sim.train({net.inp: x}, {net.p: y}, optimizer=opt, n_epochs=1,
                          profile=False)

        else:
            for _ in range(3):
                sim.run_steps(150, profile=True)


def profile_tensor_node(use_tensor_layer):
    """Run profiler on a training benchmark.

    Parameters
    ----------
    use_tensor_layer : bool
        If True, use individual tensor_layers to build the network, as opposed
        to a single TensorNode containing all layers.
    """


    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=False)

        # create node to feed in images
        inp = nengo.Node(np.ones(28 * 28))

        if use_tensor_layer:
            nengo_nl = nengo.RectifiedLinear()

            ensemble_params = dict(max_rates=nengo.dists.Choice([100]),
                                   intercepts=nengo.dists.Choice([0]))
            amplitude = 1
            synapse = None

            x = nengo_dl.tensor_layer(
                inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
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
            def mnist_node(_, x):
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
            nengo.Connection(inp, node, synapse=None)

        out_p = nengo.Probe(x)

    with nengo_dl.Simulator(net, minibatch_size=128, dtype=tf.float32,
                            device="/gpu:0") as sim:
        inputs = {inp: np.ones((128, 2, 28 * 28))}
        targets = {
            out_p: np.zeros((128, 2, 10)) + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

        # def obj(x, y):
        #     return tf.nn.softmax_cross_entropy_with_logits_v2(
        #         logits=x, labels=y)
        obj = "mse"

        opt = tf.train.GradientDescentOptimizer(learning_rate=1)

        # run a few times to try to eliminate startup overhead (only the data
        # from the last run will be kept)
        for _ in range(3):
            sim.train(inputs, targets, opt, n_epochs=10, objective=obj,
                      profile=True)

            # sim.run_steps(2, input_feeds=inputs, profile=True)


def matmul_vs_reduce():
    """Compares two different approaches to batched matrix multiplication
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

    with tf.Session() as sess:
        for i, n_ops in enumerate(n_ops_range):
            for j, mini in enumerate(mini_range):
                print(n_ops, mini)

                for k, s0 in enumerate(s0_range):
                    for l, s1 in enumerate(s1_range):
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
                            d_val = sess.run(d, feed_dict={a_d: a_val,
                                                           x_d: x_val})
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

    np.savez(os.path.join(DATA_DIR, "matmul_benchmarks"), n_ops_range,
             mini_range, s0_range, s1_range, Z)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax)

    plt.show()


if __name__ == "__main__":
    compare_backends(raw=True)
    # profile(train=False)
    # profile_tensor_node(use_tensor_layer=True)
    # matmul_vs_reduce()
