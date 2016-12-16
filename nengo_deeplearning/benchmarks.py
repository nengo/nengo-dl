import time

import matplotlib.pyplot as plt
import nengo
import nengo_ocl
import numpy as np

import nengo_deeplearning as nengo_dl


def cconv(dimensions, neurons_per_d, neuron_type):
    with nengo.Network() as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type

        cconv = nengo.networks.CircularConvolution(neurons_per_d, dimensions)

        inp_a = nengo.Node([0] * dimensions)
        inp_b = nengo.Node([1] * dimensions)
        nengo.Connection(inp_a, cconv.A)
        nengo.Connection(inp_b, cconv.B)

    return net


def integrator(dimensions, neurons_per_d, neuron_type):
    with nengo.Network() as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type

        integ = nengo.networks.Integrator(0.1, neurons_per_d * dimensions,
                                          dimensions)

        inp = nengo.Node([0] * dimensions)
        nengo.Connection(inp, integ.input)

    return net


def pes(dimensions, neurons_per_d, neuron_type):
    with nengo.Network() as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type

        inp = nengo.Node([1] * dimensions)
        pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        post = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        err = nengo.Node(size_in=dimensions)
        nengo.Connection(inp, pre)
        nengo.Connection(post, err, transform=-1)
        nengo.Connection(inp, err)

        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        nengo.Connection(err, conn.learning_rule)

    return net


# TODO: add a probing benchmark

if __name__ == "__main__":
    benchmarks = [pes, cconv, integrator]
    n_range = [32]
    d_range = [128, 256, 512]
    neuron_types = [nengo.LIFRate]  # , nengo.LIF]
    backends = [nengo_dl, nengo, nengo_ocl]

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

                        with backend.Simulator(None, model=model) as sim:
                            start = time.time()
                            sim.run(5.0)
                            data[i, j, k, l, m] = time.time() - start
    np.savez("benchmark_data.npz", data)

    # data = np.load("benchmark_data.npz")["arr_0"]

    print(data[..., 1] / data[..., 0])

    for i in range(len(benchmarks)):
        for j in range(len(neuron_types)):
            plt.figure()
            plt.title("%s %s" % (benchmarks[i], neuron_types[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 1])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo")

            plt.figure()
            plt.title("%s %s" % (benchmarks[i], neuron_types[j]))
            plt.plot(d_range, data[i, 0, :, j, 0] / data[i, 0, :, j, 2])
            plt.xlabel("dimensions")
            plt.ylabel("nengo_dl / nengo_ocl")

    plt.show()
