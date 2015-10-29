import hessianfree as hf
import matplotlib.pyplot as plt
import nengo
from nengo.utils.functions import piecewise
import numpy as np


def convert_layer(nonlinearity, n_neurons):
    # TODO: let this map in both directions

    ens = nengo.Ensemble(n_neurons, 1, gain=np.ones(n_neurons),
                         bias=np.zeros(n_neurons))

    if isinstance(nonlinearity, hf.nl.ReLU):
        ens.neuron_type = nengo.neurons.RectifiedLinear()
    elif isinstance(nonlinearity, hf.nl.Logistic):
        ens.neuron_type = nengo.neurons.Sigmoid(tau_ref=1)
    elif isinstance(nonlinearity, hf.nl.Linear):
        ens = nengo.Node(size_in=n_neurons)
        ens.neurons = ens
    else:
        raise TypeError("Cannot convert nonlinearity %s" % nonlinearity)

    return ens


def hf_to_nengo(hfnet, dt=1):
    with nengo.Network() as net:
        # convert each layer to an ensemble
        layers = [convert_layer(l, hfnet.shape[i])
                  for i, l in enumerate(hfnet.layers)]

        # add in feedforward connections
        for pre in hfnet.conns:
            for post in hfnet.conns[pre]:
                W, b = hfnet.get_weights(hfnet.W, (pre, post))
                nengo.Connection(layers[pre].neurons, layers[post].neurons,
                                 transform=W.T, synapse=None)
                layers[post].bias += b

        if isinstance(hfnet, hf.RNNet):
            # bias on first timestep
            bias = nengo.Node(piecewise({dt: [1], 2 * dt: [0]}), size_out=1)

            for l in range(hfnet.n_layers):
                if hfnet.rec_layers[l]:
                    W, b = hfnet.get_weights(hfnet.W, (l, l))
                    nengo.Connection(layers[l].neurons, layers[l].neurons,
                                     transform=W.T, synapse=0)

                    nengo.Connection(bias, layers[l].neurons,
                                     transform=b[:, None], synapse=None)

    net.layers = layers
    return net


class ArrayInput(nengo.processes.Process):
    def __init__(self, data):
        super(ArrayInput, self).__init__()
        self.data = data

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == self.data.shape[1]

        def step(_):
            step.count += 1
            return self.data[step.count % self.data.shape[0]]
        step.count = -1

        return step


def test_ff():
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    hfnet = hf.FFNet([2, 10, 1], layers=[hf.nl.Linear(), hf.nl.Logistic(),
                                         hf.nl.Logistic()])
    hfnet.run_batches(inputs, targets, max_epochs=40,
                      optimizer=hf.opt.HessianFree(CG_iter=10))

    outputs = hfnet.forward(inputs, hfnet.W)[-1]
    print "hf net"
    print inputs
    print outputs

    net = hf_to_nengo(hfnet)

    with net:
        inpt = nengo.Node(ArrayInput(inputs), size_out=2)
        nengo.Connection(inpt, net.layers[0].neurons, synapse=None)
        in_p = nengo.Probe(net.layers[0].neurons)
        out_p = nengo.Probe(net.layers[-1].neurons)

    sim = nengo.Simulator(net, dt=1)
    sim.run(4)

    print "nengo net"
    print sim.data[in_p]
    print sim.data[out_p]


def test_rnn():
    n_inputs = 10
    sig_len = 10
    inputs = np.outer(np.linspace(0.1, 0.9, n_inputs),
                      np.ones(sig_len))[:, :, None]
    targets = np.outer(np.linspace(0.1, 0.9, n_inputs),
                       np.linspace(0, 1, sig_len))[:, :, None]
    inputs = inputs.astype(np.float32)
    targets = targets.astype(np.float32)

    hfnet = hf.RNNet(shape=[1, 10, 1], layers=hf.nl.Logistic())
    hfnet.run_batches(inputs, targets, max_epochs=30,
                      optimizer=hf.opt.HessianFree(CG_iter=100))

    plt.figure()
    plt.plot(inputs.squeeze().T)
    plt.title("hf inputs")

    outputs = hfnet.forward(inputs, hfnet.W)[-1]
    plt.figure()
    plt.plot(outputs.squeeze().T)
    plt.title("hf outputs")

    net = hf_to_nengo(hfnet)
    with net:
        array_in = ArrayInput(inputs[0])
        inp = nengo.Node(array_in, size_out=1)
        nengo.Connection(inp, net.layers[0], synapse=None)
        in_p = nengo.Probe(net.layers[0].neurons)
        out_p = nengo.Probe(net.layers[-1].neurons)

    plt.figure()
    plt_in = plt.gca()
    plt.title("nengo inputs")
    plt.figure()
    plt_out = plt.gca()
    plt.title("nengo outputs")

    sim = nengo.Simulator(net, dt=1)
    for i in range(n_inputs):
        array_in.data = inputs[i]
        sim.reset()
        sim.run(sig_len)

        plt_in.plot(sim.data[in_p])
        plt_out.plot(sim.data[out_p])

    plt.show()

# test_ff()
test_rnn()
