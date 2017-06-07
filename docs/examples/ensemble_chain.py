"""
This example illustrates how a chain of ensembles can be trained end-to-end
using NengoDL.  The function is not particularly challenging
(:math:`f(x) = 2x`), this is just to illustrate how to apply
:meth:`.Simulator.train`.
"""

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

with nengo.Network(seed=0) as net:
    # these parameter settings aren't necessary, but they set things up in
    # a more standard machine learning way, for familiarity
    net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    net.config[nengo.Connection].synapse = None

    # connect up our input node, and 3 ensembles in series
    a = nengo.Node([0.5])
    b = nengo.Ensemble(30, 1)
    c = nengo.Ensemble(30, 1)
    d = nengo.Ensemble(30, 1)
    nengo.Connection(a, b)
    nengo.Connection(b, c)
    nengo.Connection(c, d)

    # define our outputs with a probe on the last ensemble in the chain
    p = nengo.Probe(d)

n_steps = 5  # the number of simulation steps we want to run our model for
mini_size = 100  # minibatch size

with nengo_dl.Simulator(net, minibatch_size=mini_size,
                        device="/cpu:0") as sim:
    # create input/target data. this could be whatever we want, but here
    # we'll train the network to output 2x its input
    input_data = np.random.uniform(-1, 1, size=(10000, n_steps, 1))
    target_data = input_data * 2

    # train the model, passing `input_data` to our input node `a` and
    # `target_data` to our output probe `p`. we can use whatever TensorFlow
    # optimizer we want here.
    sim.train({a: input_data}, {p: target_data},
              tf.train.MomentumOptimizer(5e-2, 0.9), n_epochs=30)

    # run the model to see the results of the training. note that this will
    # use the input values specified in our `nengo.Node` definition
    # above (0.5)
    sim.run_steps(n_steps)

    # so the output should be 1
    assert np.allclose(sim.data[p], 1, atol=1e-2)

    sim.soft_reset(include_probes=True)

    # or if we wanted to see the performance on a test dataset, we could do
    test_data = np.random.uniform(-1, 1, size=(mini_size, n_steps, 1))
    sim.run_steps(n_steps, input_feeds={a: test_data})

    assert np.allclose(test_data * 2, sim.data[p], atol=1e-2)
