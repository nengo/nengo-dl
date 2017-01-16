import time

import matplotlib.pyplot as plt
import nengo
import nengo_ocl
import numpy as np

import nengo_deeplearning as nengo_dl


def cconv(dimensions, neurons_per_d, neuron_type):
    """Circular convolution (EnsembleArray) benchmark.

    Parameters
    ----------
    dimensions : int
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : nengo.neurons.NeuronType
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="cconv") as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])

        cconv = nengo.networks.CircularConvolution(neurons_per_d, dimensions)

        inp_a = nengo.Node([0] * dimensions)
        inp_b = nengo.Node([1] * dimensions)
        nengo.Connection(inp_a, cconv.A)
        nengo.Connection(inp_b, cconv.B)

        nengo.Probe(cconv.output)

    return net


def integrator(dimensions, neurons_per_d, neuron_type):
    """Single integrator ensemble benchmark.

    Parameters
    ----------
    dimensions : int
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : nengo.neurons.NeuronType
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="integrator") as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])

        integ = nengo.networks.Integrator(0.1, neurons_per_d * dimensions,
                                          dimensions)

        inp = nengo.Node([0] * dimensions)
        nengo.Connection(inp, integ.input)

        nengo.Probe(integ.ensemble)

    return net


def pes(dimensions, neurons_per_d, neuron_type):
    """PES learning rule benchmark.

    Parameters
    ----------
    dimensions : int
        number of dimensions for vector values
    neurons_per_d : int
        number of neurons to use per vector dimension
    neuron_type : nengo.neurons.NeuronType
        simulation neuron type

    Returns
    -------
    nengo.Network
        benchmark network
    """

    with nengo.Network(label="pes") as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])

        inp = nengo.Node([1] * dimensions)
        pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        post = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        err = nengo.Node(size_in=dimensions)
        nengo.Connection(inp, pre)
        nengo.Connection(post, err, transform=-1)
        nengo.Connection(inp, err)

        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        nengo.Connection(err, conn.learning_rule)

        nengo.Probe(post)

    return net


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
    neuron_types = [nengo.RectifiedLinear]
    backends = [nengo_dl, nengo, nengo_ocl]

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
                            if backend == nengo_dl:
                                kwargs = {"step_blocks": 50,
                                          "unroll_simulation": True,
                                          # "device": "/cpu:0"
                                          }
                            else:
                                kwargs = {}
                            try:
                                with backend.Simulator(None, model=model,
                                                       **kwargs) as sim:
                                    start = time.time()
                                    sim.run(5.0)
                                    data[i, j, k, l, m] = time.time() - start
                                    print("time", data[i, j, k, l, m])
                            except Exception as e:
                                print(backend, "CRASHED")
                                print(e)
                                data[i, j, k, l, m] = np.nan

        np.savez("benchmark_data.npz", data)
    else:
        data = np.load("benchmark_data.npz")["arr_0"]

    bench_names = ["pes", "integrator", "cconv"]
    neuron_names = ["relu"]

    f, axes = plt.subplots(1, 3)

    for i in range(len(benchmarks)):
        for j in range(len(neuron_types)):
            plt.figure()
            plt.title("%s (%s)" % (bench_names[i], neuron_names[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 1])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo")

            plt.figure()
            plt.title("%s (%s)" % (bench_names[i], neuron_names[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 2])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo_ocl")

            axes[i].set_title("%s (%s)" % (bench_names[i], neuron_names[j]))
            axes[i].plot(d_range, data[i, 0, :, j, :])
            axes[i].set_xlabel("dimensions")
            axes[i].set_ylabel("seconds")
            axes[i].legend(["nengo_dl", "nengo", "nengo_ocl"])
            axes[i].set_ylim([0, 50])

    plt.show()


def profiling():
    """Run profiler on one of the benchmarks."""

    # note: in order for GPU profiling to work, you have to manually add
    # ...\CUDA\v8.0\extras\CUPTI\libx64 to your path
    net = pes(128, 32, nengo.RectifiedLinear())
    with nengo_dl.Simulator(net, tensorboard=True, step_blocks=50,
                            device="/gpu:0", unroll_simulation=True) as sim:
        sim.run_steps(50, profile=True)


if __name__ == "__main__":
    compare_backends(raw=True)
    # profiling()
