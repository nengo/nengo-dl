import nengo
from nengo.params import Parameter
from nengo.processes import PresentInput
import numpy as np

import nengo_lasagne


def test_xor():
    inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    with nengo.Network() as net, nengo_lasagne.default_config():
        N = 10
        d = 2
        inp = nengo.Node(output=PresentInput(inputs, 0.001),
                         label="input")
        ens = nengo.Ensemble(N, d, label="ens")
        output = nengo.Node(size_in=1, label="output")

        #         nengo.Connection(inp, ens.neurons, transform=np.zeros((N, 2)))
        #         nengo.Connection(ens.neurons, output, transform=np.zeros((1, N)))
        nengo.Connection(inp, ens, transform=np.zeros((d, 2)))
        nengo.Connection(ens, output, transform=np.zeros((1, d)))

        p = nengo.Probe(output)

    nengo_lasagne.settings(net, {inp: inputs}, {output: targets}, batch_size=4,
                           n_epochs=1000, l_rate=0.1)

    sim = nengo_lasagne.Simulator(net)
    sim.run_steps(4)

    print(sim.data[p])


def test_fancy():
    inputs = [np.random.uniform(-1, 1, size=(1000, 1)).astype(np.float32)
              for _ in range(3)]
    outputs = (inputs[0] + 1) * inputs[1] ** 2 + np.sin(inputs[2])

    with nengo.Network() as net, nengo_lasagne.default_config() as conf:
        input_nodes = [nengo.Node(output=PresentInput(x, 0.001))
                       for x in inputs]

        ens0 = nengo.Ensemble(100, 2)
        ens1 = nengo.Ensemble(50, 1)
        nengo.Connection(input_nodes[0], ens0[0])
        nengo.Connection(input_nodes[1], ens0[1])
        nengo.Connection(input_nodes[2], ens1)

        output_node = nengo.Node(size_in=1)
        nengo.Connection(ens0, output_node,
                         function=lambda x: (x[0] + 1) * x[1] ** 2)
        nengo.Connection(ens1, output_node, function=np.sin)

        p = nengo.Probe(output_node)

    nengo_lasagne.settings(net, dict(zip(input_nodes, inputs)),
                           {output_node: outputs}, n_epochs=1000, l_rate=1e-2,
                           batch_size=100, nef_init=True)
    sim = nengo_lasagne.Simulator(net)
    # sim = nengo.Simulator(net)
    sim.run_steps(1000)

    print(np.sqrt(np.mean(outputs - sim.data[p]) ** 2))
    print(inputs[0][:3])
    print(inputs[1][:3])
    print(inputs[2][:3])
    print(outputs[:3])
    print(sim.data[p][:3])


# test_xor()
test_fancy()
