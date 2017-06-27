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
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="cconv", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        cconv = nengo.networks.CircularConvolution(neurons_per_d, dimensions)

        inp_a = nengo.Node([0] * dimensions)
        inp_b = nengo.Node([1] * dimensions)
        nengo.Connection(inp_a, cconv.A)
        nengo.Connection(inp_b, cconv.B)

        p = nengo.Probe(cconv.output)

    return net, p


def integrator(dimensions, neurons_per_d, neuron_type):
    """Single integrator ensemble benchmark.

    Parameters
    ----------
    dimensions : int
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="integrator", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        integ = nengo.networks.Integrator(0.1, neurons_per_d * dimensions,
                                          dimensions)

        inp = nengo.Node([0] * dimensions)
        nengo.Connection(inp, integ.input)

        p = nengo.Probe(integ.ensemble)

    return net, p


def pes(dimensions, neurons_per_d, neuron_type):
    """PES learning rule benchmark.

    Parameters
    ----------
    dimensions : int
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : :class:`~nengo:nengo.neurons.NeuronType`
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="pes", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        inp = nengo.Node([1] * dimensions)
        pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        post = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        err = nengo.Node(size_in=dimensions)
        nengo.Connection(inp, pre)
        nengo.Connection(post, err, transform=-1)
        nengo.Connection(inp, err)

        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        nengo.Connection(err, conn.learning_rule)

        p = nengo.Probe(post)

    return net, p


def compare_backends(raw=False):
    """Compare the run time of different backends across benchmarks and
    a range of parameters.

    Parameters
    ----------
    raw : bool
        if True, run the benchmarks to collect data, otherwise load data from
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

                        net, p = bench(dimensions, neurons, neuron_type())
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
                            #     canonical = sim.data[p]
                            # else:
                            #     assert np.allclose(canonical, sim.data[p],
                            #                        atol=1e-3)

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


def profile_run():
    """Run profiler on one of the benchmarks."""

    # note: in order for GPU profiling to work, you have to manually add
    # ...\CUDA\v8.0\extras\CUPTI\libx64 to your path
    net, p = pes(128, 32, nengo.RectifiedLinear())
    with nengo_dl.Simulator(net, tensorboard=False, unroll_simulation=50,
                            device="/gpu:0") as sim:
        # run a few times to try to eliminate startup overhead (only the data
        # from the last run will be kept)
        for _ in range(3):
            sim.run_steps(150, profile=True)


def profile_train(use_tensor_layer):
    nl = tf.nn.relu

    # nl = functools.partial(softlif_layer, sigma=0.002, tau_rc=0.022,
    #                        tau_ref=0.002, amplitude=0.063)
    # nl = tf.nn.sigmoid

    def softlif_layer(x, sigma=1, tau_ref=0.002, tau_rc=0.02, amplitude=1):
        # x -= 1
        z = tf.nn.softplus(x / sigma) * sigma
        z += 1e-10
        rates = amplitude / (tau_ref + tau_rc * tf.log1p(1 / z))
        return rates

    @nengo_dl.reshaped((28, 28, 1))
    def mnist_node(_, x):
        # init = init_ops.variance_scaling_initializer(scale=1, mode="fan_avg",
        #                                              distribution="uniform")
        # bias_init = init_ops.zeros_initializer()

        x = tf.layers.conv2d(x, filters=32, kernel_size=3,
                             activation=nl,
                             # kernel_initializer=init,
                             # bias_initializer=bias_init,
                             )
        x = tf.layers.conv2d(x, filters=32, kernel_size=3,
                             activation=nl,
                             # kernel_initializer=init,
                             # bias_initializer=bias_init,
                             )
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 128, activation=nl,
                            # kernel_initializer=init,
                            # bias_initializer=bias_init,
                            )
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.dense(x, 10)

        return x

    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=False)

        # create node to feed in images
        inp = nengo.Node(np.ones(28 * 28))

        if use_tensor_layer:
            ensemble_params = dict(max_rates=nengo.dists.Choice([100]),
                                   intercepts=nengo.dists.Choice([0]))
            amplitude = 1
            synapse = None

            x = nengo_dl.tensor_layer(
                inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
                kernel_size=3,
                # activation=nl
            )
            x = nengo_dl.tensor_layer(x, nengo.RectifiedLinear(),
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(26, 26, 32),
                transform=amplitude, filters=32, kernel_size=3,
                # activation=nl
            )
            x = nengo_dl.tensor_layer(x, nengo.RectifiedLinear(),
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(24, 24, 32),
                synapse=synapse, transform=amplitude, pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(
                x, tf.layers.dense, units=128,
                # activation=nl
            )
            x = nengo_dl.tensor_layer(x, nengo.RectifiedLinear(),
                                      **ensemble_params)

            x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.4,
                                      transform=amplitude)

            x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)
        else:
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

        def obj(x, y):
            return tf.nn.softmax_cross_entropy_with_logits(
                logits=x, labels=y)

        opt = tf.train.AdadeltaOptimizer(learning_rate=1)

        # run a few times to try to eliminate startup overhead (only the data
        # from the last run will be kept)
        for _ in range(3):
            sim.train(inputs, targets, opt, n_epochs=1000, objective=obj,
                      profile=False)

            # sim.run_steps(2, input_feeds=inputs, profile=True)


if __name__ == "__main__":
    # compare_backends(raw=True)
    # profile_run()
    profile_train(use_tensor_layer=True)
