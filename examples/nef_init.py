"""
This example shows how to train a network to compute a more complex feedforward
function :math:`f(x, y, z) = (x + 1)*y^2 + \sin(z)^2`.  It demonstrates how
the standard NEF optimization can be used to initialize the network, and then
how further applying NengoDL training can improve performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import nengo
import tensorflow as tf

import nengo_dl


# the function we want to approximate
def target_func(x, y, z):
    return (x + 1) * y ** 2 + np.sin(z) ** 2


with nengo.Network() as net:
    # default parameters for the network
    net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    net.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(0, 1)
    net.config[nengo.Connection].synapse = None

    # input to the network (could be anything, we'll use band-limited white
    # noise signals
    x, y, z = [nengo.Node(output=nengo.processes.WhiteSignal(1, 5))
               for i in range(3)]

    # output node
    f = nengo.Node(size_in=1)

    # neural ensembles
    ens0 = nengo.Ensemble(100, 2)
    ens1 = nengo.Ensemble(50, 1)
    ens2 = nengo.Ensemble(50, 1)

    # connect the input signals to ensemble inputs
    nengo.Connection(x, ens0[0])
    nengo.Connection(y, ens0[1])
    nengo.Connection(z, ens1)

    # create connections between the neural ensembles, using the standard
    # Nengo/NEF optimization to compute connection weights on a layer-by-layer
    # basis so that the overall output will approximate the target function
    nengo.Connection(ens1, ens2, function=np.sin)
    nengo.Connection(ens2, f, function=np.square)
    nengo.Connection(ens0, f, function=lambda x: (x[0] + 1) * x[1] ** 2)

    # collect data on the inputs/outputs
    x_p = nengo.Probe(x)
    y_p = nengo.Probe(y)
    z_p = nengo.Probe(z)
    f_p = nengo.Probe(f)

with nengo_dl.Simulator(net, minibatch_size=32) as sim:
    # create training data, using random inputs from -1 to 1
    inputs = {x: np.random.uniform(-1, 1, size=(1024, 1, 1)),
              y: np.random.uniform(-1, 1, size=(1024, 1, 1)),
              z: np.random.uniform(-1, 1, size=(1024, 1, 1))}

    targets = {f_p: target_func(inputs[x], inputs[y], inputs[z])}

    # run the network before training
    sim.run(1.0)
    plt.figure()
    plt.plot(sim.trange(), sim.data[f_p][0], label="output")
    plt.plot(sim.trange(), target_func(sim.data[x_p][0], sim.data[y_p][0],
                                       sim.data[z_p][0]), label="target")
    plt.legend()
    plt.title("pre training")
    plt.xlabel("time")

    # compute mean-squared error
    print("pre-training mse:", sim.loss(inputs, targets, "mse"))

    # optimize the parameters of the network (in this case we're using
    # gradient descent with nesterov momentum)
    sim.train(inputs, targets,
              tf.train.MomentumOptimizer(0.002, 0.9, use_nesterov=True),
              n_epochs=100)

    # check mean-squared error after training (should improve by several
    # orders of magnitude)
    print("post-training mse:", sim.loss(inputs, targets, "mse"))

    # run the network after training, to visualize improvement
    sim.soft_reset(include_probes=True)
    sim.run(1.0)
    plt.figure()
    plt.plot(sim.trange(), sim.data[f_p][0], label="output")
    plt.plot(sim.trange(), target_func(sim.data[x_p][0], sim.data[y_p][0],
                                       sim.data[z_p][0]), label="target")
    plt.legend()
    plt.title("post training")
    plt.xlabel("time")

plt.show()
