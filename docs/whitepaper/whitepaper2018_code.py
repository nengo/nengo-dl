# snippet 1 (section 3.1)
import nengo
import nengo_dl

with nengo.Network() as net:
    a = nengo.Node(output=[1])
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(a, b)
    p = nengo.Probe(b)

with nengo_dl.Simulator(net) as sim:
    sim.run(1.0)
    print(sim.data[p])

# snippet 2 (section 3.1)
with nengo_dl.Simulator(net, minibatch_size=10) as sim:
    sim.run(1.0)
    print(sim.data[p])

# snippet 3 (section 3.1)
with nengo_dl.Simulator(net) as sim:
    for i in range(10):
        sim.run(1.0)
        print(sim.data[p])
        sim.reset()

# snippet 4 (section 3.1)
import numpy as np

with nengo_dl.Simulator(net, minibatch_size=10) as sim:
    sim.run(1.0, input_feeds={a: np.random.randn(10, 1000, 1)})
    print(sim.data[p])

# snippet 5 (section 3.2)
import tensorflow as tf

inputs = np.random.randn(50, 1000, 1)
targets = inputs**2

with nengo_dl.Simulator(net, minibatch_size=10) as sim:
    sim.train(
        data={a: inputs,
              p: targets},
        optimizer=tf.train.AdamOptimizer(),
        n_epochs=2,
        objective={p: "mse"})

# snippet 6 (section 3.3)
with net:
    def tensor_func(t, x):
        return tf.layers.dense(x, 100, activation=tf.nn.relu)
    t = nengo_dl.TensorNode(tensor_func, size_in=1)
    nengo.Connection(a, t)
    nengo.Connection(t, b.neurons)

# snippet 7 (section 3.3)
with net:
    t = nengo_dl.tensor_layer(a, tf.layers.dense, units=100,
                              activation=tf.nn.relu)
    nengo.Connection(t, b.neurons)
