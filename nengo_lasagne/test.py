import pickle

import lasagne
import lasagne.nonlinearities as nl
import matplotlib.pyplot as plt
import nengo
from nengo.processes import PresentInput
import numpy as np

import nengo_lasagne as nlg


def test_xor():
    inputs = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]],
                        dtype=np.float32)
    targets = np.asarray([[[0]], [[1]], [[1]], [[0]]], dtype=np.float32)

    with nengo.Network() as net, nlg.default_config():
        N = 10
        d = 2
        inp = nengo.Node(output=PresentInput(inputs, 0.001),
                         label="input")
        ens = nengo.Ensemble(N, d, label="ens")
        output = nengo.Node(size_in=1, label="output")

        #         nengo.Connection(inp, ens.neurons, transform=np.zeros((N, 2)))
        #         nengo.Connection(ens.neurons, output, transform=np.zeros((1, N)))
        nengo.Connection(inp, ens, transform=nlg.init.GlorotUniform())
        nengo.Connection(ens, output, transform=nlg.init.GlorotUniform())

        p = nengo.Probe(output)

    sim = nlg.Simulator(net)
    sim.model.train({inp: inputs}, {output: targets}, minibatch_size=4,
                    n_epochs=1000, l_rate=0.1)
    sim.run_steps(4)

    print(sim.data[p])


def test_fancy():
    inputs = np.asarray([np.random.uniform(-1, 1,
                                           size=(1000, 1)).astype(np.float32)
                         for _ in range(3)])
    outputs = (inputs[0] + 1) * inputs[1] ** 2 + np.sin(inputs[2])
    inputs = inputs[:, :, None]
    outputs = outputs[:, None]

    with nengo.Network() as net, nlg.default_config():
        input_nodes = [nengo.Node(output=PresentInput(x.squeeze(axis=1),
                                                      0.001),
                                  label="node_%d" % i)
                       for i, x in enumerate(inputs)]

        ens0 = nengo.Ensemble(100, 2, label="ens0")
        ens1 = nengo.Ensemble(50, 1, label="ens1")
        nengo.Connection(input_nodes[0], ens0[0])
        nengo.Connection(input_nodes[1], ens0[1])
        nengo.Connection(input_nodes[2], ens1)

        output_node = nengo.Node(size_in=1)
        nengo.Connection(ens0, output_node,
                         function=lambda x: (x[0] + 1) * x[1] ** 2)
        nengo.Connection(ens1, output_node, function=np.sin)

        p = nengo.Probe(output_node)

    sim = nlg.Simulator(net)
    sim.model.train(dict(zip(input_nodes, inputs)), {output_node: outputs},
                    n_epochs=1000, minibatch_size=100,
                    optimizer_kwargs={"learning_rate": 1e-2})
    # sim = nengo.Simulator(net)
    sim.run_steps(1000)

    print(np.sqrt(np.mean(outputs - sim.data[p]) ** 2))
    print(inputs[0][:3])
    print(inputs[1][:3])
    print(inputs[2][:3])
    print(outputs[:3])
    print(sim.data[p][:, :3])


def test_recurrent():
    batch_size = 1000
    sig_len = 10
    dt = 1e-3
    inputs = np.random.randn(batch_size, sig_len, 1).astype(np.float32)
    targets = np.cumsum(inputs, axis=1) * dt

    with nengo.Network() as net, nlg.default_config():
        input_node = nengo.Node(output=nengo.processes.WhiteNoise(scale=False))
        ens = nengo.Ensemble(50, 1)
        output_node = nengo.Node(size_in=1)

        nengo.Connection(input_node, ens)
        nengo.Connection(ens.neurons, ens.neurons,
                         transform=nlg.init.GlorotUniform())
        nengo.Connection(ens, output_node)

        input_p = nengo.Probe(input_node)
        output_p = nengo.Probe(output_node)

    sim = nlg.Simulator(net, dt=dt)

    print("layers")
    print(lasagne.layers.get_all_layers(sim.model.params[output_node].output))

    sim.model.train({input_node: inputs}, {output_node: targets},
                    n_epochs=1000, minibatch_size=100,
                    optimizer_kwargs={"learning_rate": 1e-2})

    sim.run_steps(sig_len,
                  {input_node: np.random.randn(batch_size,
                                               sig_len, 1).astype(np.float32)})

    truth = np.cumsum(sim.data[input_p], axis=1) * dt

    print(sim.data[output_p].shape)
    print(np.sqrt(np.mean((truth - sim.data[output_p]) ** 2)))


def test_lasagnenode():
    with open("mnist.pkl", "rb") as f:
        train, _, test = pickle.load(f, encoding="bytes")

    n_conv = 2

    # input layer
    l = lasagne.layers.InputLayer(shape=(None,))
    l = lasagne.layers.ReshapeLayer(l, shape=(-1, 1, 28, 28))

    # conv layers
    for _ in range(n_conv):
        l = lasagne.layers.Conv2DLayer(l, num_filters=32, filter_size=(5, 5),
                                       nonlinearity=nl.rectify,
                                       W=lasagne.init.HeNormal(gain="relu"))
        l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))

    # dense layer
    l = lasagne.layers.DenseLayer(l, num_units=256, nonlinearity=nl.rectify,
                                  W=lasagne.init.HeNormal(gain='relu'))

    # dropout
    l = lasagne.layers.DropoutLayer(l, p=0.5)

    # output layer
    l = lasagne.layers.DenseLayer(l, num_units=10, nonlinearity=nl.softmax)

    with nengo.Network() as net, nlg.default_config():
        net.config[nengo.Connection].set_param("insert_weights",
                                               nengo.params.BoolParam(True))

        input_node = nengo.Node(output=PresentInput(test[0], 0.001))
        conv_layers = nlg.layers.LasagneNode(output=l, size_in=784)
        output_node = nengo.Node(size_in=10)

        conn1 = nengo.Connection(input_node, conv_layers)
        conn2 = nengo.Connection(conv_layers, output_node)

        net.config[conn1].insert_weights = False
        net.config[conn2].insert_weights = False

        input_p = nengo.Probe(input_node)
        p = nengo.Probe(output_node)
        p2 = nengo.Probe(conv_layers)

    sim = nlg.Simulator(net)

    # print("layers")
    # for l in lasagne.layers.get_all_layers(
    #         sim.model.params[output_node].output):
    #     print("=" * 30)
    #     print(str(l), l.output_shape)
    #     print("incoming:", end=" ")
    #     if hasattr(l, "input_layer"):
    #         print(l.input_layer)
    #     elif hasattr(l, "input_layers"):
    #         print(l.input_layers)
    #     print()

    targets = np.zeros((train[1].shape[0], 1, 10), dtype=np.float32)
    targets[np.arange(train[1].shape[0]), :, train[1]] = 1.0

    sim.train({input_node: train[0][:, None]}, {output_node: targets},
              n_epochs=1, minibatch_size=500,
              optimizer=lasagne.updates.nesterov_momentum,
              optimizer_kwargs={"learning_rate": 0.01, "momentum": 0.9},
              objective=lasagne.objectives.categorical_crossentropy)

    sim.run_steps(test[0].shape[0])
    # sim.run_steps(1, inputs={input_node: test[0][:, None]})

    output = sim.data[p].squeeze()

    print(np.argmax(output[:10], axis=1))
    print(test[1][:10])
    print(np.mean(np.argmax(output, axis=1) == test[1]))

    # **** nengo simulator ****

    sim2 = nengo.Simulator(net)
    sim2.reset()
    sim2.run_steps(test[0].shape[0])

    output = sim2.data[p]

    print(np.argmax(output[:10], axis=1))
    print(test[1][1:11])
    print(np.mean(np.argmax(output[:-1], axis=1) == test[1][1:]))

    plt.show()


# test_xor()
# test_fancy()
# test_recurrent()
test_lasagnenode()
