"""
Benchmark networks and utilities for evaluating NengoDL's performance.
"""

import inspect
import random
import timeit

import click
import nengo
import numpy as np
import tensorflow as tf
from nengo.utils.filter_design import cont2discrete

import nengo_dl


def cconv(dimensions, neurons_per_d, neuron_type):
    """
    Circular convolution (EnsembleArray) benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="cconv", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.cconv = nengo.networks.CircularConvolution(neurons_per_d, dimensions)

        net.inp_a = nengo.Node([0] * dimensions)
        net.inp_b = nengo.Node([1] * dimensions)
        nengo.Connection(net.inp_a, net.cconv.input_a)
        nengo.Connection(net.inp_b, net.cconv.input_b)

        net.p = nengo.Probe(net.cconv.output)

    return net


def integrator(dimensions, neurons_per_d, neuron_type):
    """
    Single integrator ensemble benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="integrator", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.integ = nengo.networks.EnsembleArray(neurons_per_d, dimensions)
        nengo.Connection(net.integ.output, net.integ.input, synapse=0.01)

        net.inp = nengo.Node([0] * dimensions)
        nengo.Connection(net.inp, net.integ.input, transform=0.01)

        net.p = nengo.Probe(net.integ.output)

    return net


def pes(dimensions, neurons_per_d, neuron_type):
    """
    PES learning rule benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="pes", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1, -1])
        net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)

        net.inp = nengo.Node([1] * dimensions)
        net.pre = nengo.Ensemble(neurons_per_d * dimensions, dimensions)
        net.post = nengo.Node(size_in=dimensions)

        nengo.Connection(net.inp, net.pre)

        conn = nengo.Connection(net.pre, net.post, learning_rule_type=nengo.PES())

        nengo.Connection(net.post, conn.learning_rule, transform=-1)
        nengo.Connection(net.inp, conn.learning_rule)

        net.p = nengo.Probe(net.post)

    return net


def basal_ganglia(dimensions, neurons_per_d, neuron_type):
    """
    Basal ganglia network benchmark.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network(label="basal_ganglia", seed=0) as net:
        net.config[nengo.Ensemble].neuron_type = neuron_type

        net.inp = nengo.Node([1] * dimensions)
        net.bg = nengo.networks.BasalGanglia(dimensions, neurons_per_d)
        nengo.Connection(net.inp, net.bg.input)
        net.p = nengo.Probe(net.bg.output)

    return net


def mnist(use_tensor_layer=True):
    """
    A network designed to stress-test tensor layers (based on mnist net).

    Parameters
    ----------
    use_tensor_layer : bool
        If True, use individual tensor_layers to build the network, as opposed
        to a single TensorNode containing all layers.

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    with nengo.Network() as net:
        # create node to feed in images
        net.inp = nengo.Node(np.ones(28 * 28))

        if use_tensor_layer:
            nengo_nl = nengo.RectifiedLinear()

            ensemble_params = dict(
                max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0])
            )
            amplitude = 1
            synapse = None

            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
                net.inp, shape_in=(28, 28, 1)
            )
            x = nengo_dl.Layer(nengo_nl)(x, **ensemble_params)

            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
                x, shape_in=(26, 26, 32), transform=amplitude
            )
            x = nengo_dl.Layer(nengo_nl)(x, **ensemble_params)

            x = nengo_dl.Layer(
                tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
            )(x, shape_in=(24, 24, 32), synapse=synapse, transform=amplitude)

            x = nengo_dl.Layer(tf.keras.layers.Dense(units=128))(x)
            x = nengo_dl.Layer(nengo_nl)(x, **ensemble_params)

            x = nengo_dl.Layer(tf.keras.layers.Dropout(rate=0.4))(
                x, transform=amplitude
            )

            x = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)
        else:
            nl = tf.nn.relu

            # def softlif_layer(x, sigma=1, tau_ref=0.002, tau_rc=0.02,
            #                   amplitude=1):
            #     # x -= 1
            #     z = tf.nn.softplus(x / sigma) * sigma
            #     z += 1e-10
            #     rates = amplitude / (tau_ref + tau_rc * tf.log1p(1 / z))
            #     return rates

            def mnist_node(x):  # pragma: no cover (runs in TF)
                x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=nl)(x)
                x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=nl)(x)
                x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
                x = tf.keras.layers.Flatten()(x)
                x = tf.keras.layers.Dense(128, activation=nl)(x)
                x = tf.keras.layers.Dropout(rate=0.4)(x)
                x = tf.keras.layers.Dense(10)(x)

                return x

            node = nengo_dl.TensorNode(
                mnist_node, shape_in=(28, 28, 1), shape_out=(10,)
            )
            x = node
            nengo.Connection(net.inp, node, synapse=None)

        net.p = nengo.Probe(x)

    return net


def spaun(dimensions):
    """
    Builds the Spaun network.

    See [1]_ for more details.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values

    Returns
    -------
    net : `nengo.Network`
        benchmark network

    References
    ----------
    .. [1] Chris Eliasmith, Terrence C. Stewart, Xuan Choo, Trevor Bekolay,
       Travis DeWolf, Yichuan Tang, and Daniel Rasmussen (2012). A large-scale
       model of the functioning brain. Science, 338:1202-1205.

    Notes
    -----
    This network needs to be installed via

    .. code-block:: bash

        pip install git+https://github.com/drasmuss/spaun2.0.git
    """

    # pylint: disable=import-outside-toplevel
    from _spaun.configurator import cfg
    from _spaun.experimenter import experiment
    from _spaun.modules.motor import mtr_data
    from _spaun.modules.stim import stim_data
    from _spaun.modules.vision import vis_data
    from _spaun.spaun_main import Spaun
    from _spaun.vocabulator import vocab

    vocab.sp_dim = dimensions
    cfg.mtr_arm_type = None

    cfg.set_seed(1)
    experiment.initialize(
        "A",
        stim_data.get_image_ind,
        stim_data.get_image_label,
        cfg.mtr_est_digit_response_time,
        "",
        cfg.rng,
    )
    vocab.initialize(stim_data.stim_SP_labels, experiment.num_learn_actions, cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    return Spaun()


def random_network(
    dimensions,
    neurons_per_d,
    neuron_type,
    n_ensembles,
    connections_per_ensemble,
    seed=0,
):
    """
    A randomly interconnected network of ensembles.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for vector values
    neurons_per_d : int
        Number of neurons to use per vector dimension
    neuron_type : `~nengo.neurons.NeuronType`
        Simulation neuron type
    n_ensembles : int
        Number of ensembles in the network
    connections_per_ensemble : int
        Outgoing connections from each ensemble

    Returns
    -------
    net : `nengo.Network`
        benchmark network
    """

    random.seed(seed)
    with nengo.Network(label="random", seed=seed) as net:
        net.inp = nengo.Node([0] * dimensions)
        net.out = nengo.Node(size_in=dimensions)
        net.p = nengo.Probe(net.out)
        ensembles = [
            nengo.Ensemble(
                neurons_per_d * dimensions, dimensions, neuron_type=neuron_type
            )
            for _ in range(n_ensembles)
        ]
        dec = np.ones((neurons_per_d * dimensions, dimensions))
        for ens in net.ensembles:
            # add a connection to input and output node, so we never have
            # any "orphan" ensembles
            nengo.Connection(net.inp, ens)
            nengo.Connection(ens, net.out, solver=nengo.solvers.NoSolver(dec))

            posts = random.sample(ensembles, connections_per_ensemble)
            for post in posts:
                nengo.Connection(ens, post, solver=nengo.solvers.NoSolver(dec))

    return net


def lmu(theta, input_d, native_nengo=False, dtype="float32"):
    """
    A network containing a single Legendre Memory Unit cell and dense readout.

    See [1]_ for more details.

    Parameters
    ----------
    theta : int
        Time window parameter for LMU.
    input_d : int
        Dimensionality of input signal.
    native_nengo : bool
        If True, build the LMU out of Nengo objects. Otherwise, build the LMU
        directly in TensorFlow, and use a `.TensorNode` to wrap the whole cell.
    dtype : str
        Float dtype to use for internal parameters of LMU when ``native_nengo=False``
        (``native_nengo=True`` will use the dtype of the Simulator).

    Returns
    -------
    net : `nengo.Network`
        Benchmark network

    References
    ----------
    .. [1] Aaron R. Voelker, Ivana Kajić, and Chris Eliasmith. Legendre memory units:
       continuous-time representation in recurrent neural networks.
       In Advances in Neural Information Processing Systems. 2019.
       https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.
    """
    if native_nengo:
        # building LMU cell directly out of Nengo objects

        class LMUCell(nengo.Network):
            """Implements an LMU cell as a Nengo network."""

            def __init__(self, units, order, theta, input_d, **kwargs):
                super().__init__(**kwargs)

                # compute the A and B matrices according to the LMU's mathematical
                # derivation (see the paper for details)
                Q = np.arange(order, dtype=np.float64)
                R = (2 * Q + 1)[:, None] / theta
                j, i = np.meshgrid(Q, Q)

                A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
                B = (-1.0) ** Q[:, None] * R
                C = np.ones((1, order))
                D = np.zeros((1,))

                A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

                with self:
                    nengo_dl.configure_settings(trainable=None)

                    # create objects corresponding to the x/u/m/h variables in LMU
                    self.x = nengo.Node(size_in=input_d)
                    self.u = nengo.Node(size_in=1)
                    self.m = nengo.Node(size_in=order)
                    self.h = nengo_dl.TensorNode(
                        tf.nn.tanh, shape_in=(units,), pass_time=False
                    )

                    # compute u_t
                    # note that setting synapse=0 (versus synapse=None) adds a
                    # one-timestep delay, so we can think of any connections with
                    # synapse=0 as representing value_{t-1}
                    nengo.Connection(
                        self.x, self.u, transform=np.ones((1, input_d)), synapse=None
                    )
                    nengo.Connection(
                        self.h, self.u, transform=np.zeros((1, units)), synapse=0
                    )
                    nengo.Connection(
                        self.m, self.u, transform=np.zeros((1, order)), synapse=0
                    )

                    # compute m_t
                    # in this implementation we'll make A and B non-trainable, but they
                    # could also be optimized in the same way as the other parameters
                    conn = nengo.Connection(self.m, self.m, transform=A, synapse=0)
                    self.config[conn].trainable = False
                    conn = nengo.Connection(self.u, self.m, transform=B, synapse=None)
                    self.config[conn].trainable = False

                    # compute h_t
                    nengo.Connection(
                        self.x,
                        self.h,
                        transform=np.zeros((units, input_d)),
                        synapse=None,
                    )
                    nengo.Connection(
                        self.h, self.h, transform=np.zeros((units, units)), synapse=0
                    )
                    nengo.Connection(
                        self.m,
                        self.h,
                        transform=nengo_dl.dists.Glorot(distribution="normal"),
                        synapse=None,
                    )

        with nengo.Network(seed=0) as net:
            # remove some unnecessary features to speed up the training
            nengo_dl.configure_settings(
                trainable=None, stateful=False, keep_history=False
            )

            # input node
            net.inp = nengo.Node(np.zeros(input_d))

            # lmu cell
            lmu_cell = LMUCell(units=212, order=256, theta=theta, input_d=input_d)
            conn = nengo.Connection(net.inp, lmu_cell.x, synapse=None)
            net.config[conn].trainable = False

            # dense linear readout
            out = nengo.Node(size_in=10)
            nengo.Connection(
                lmu_cell.h, out, transform=nengo_dl.dists.Glorot(), synapse=None
            )

            # record output. note that we set keep_history=False above, so this will
            # only record the output on the last timestep (which is all we need
            # on this task)
            net.p = nengo.Probe(out)
    else:
        # putting everything in a tensornode

        # define LMUCell
        class LMUCell(tf.keras.layers.AbstractRNNCell):
            """Implement LMU as Keras RNN cell."""

            def __init__(self, units, order, theta, **kwargs):
                super().__init__(**kwargs)

                self.units = units
                self.order = order
                self.theta = theta

                Q = np.arange(order, dtype=np.float64)
                R = (2 * Q + 1)[:, None] / theta
                j, i = np.meshgrid(Q, Q)

                A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
                B = (-1.0) ** Q[:, None] * R
                C = np.ones((1, order))
                D = np.zeros((1,))

                self._A, self._B, _, _, _ = cont2discrete(
                    (A, B, C, D), dt=1.0, method="zoh"
                )

            @property
            def state_size(self):
                """Size of RNN state variables."""
                return self.units, self.order

            @property
            def output_size(self):
                """Size of cell output."""
                return self.units

            def build(self, input_shape):
                """Set up all the weight matrices used inside the cell."""

                super().build(input_shape)

                input_dim = input_shape[-1]
                self.input_encoders = self.add_weight(
                    shape=(input_dim, 1), initializer=tf.initializers.ones()
                )
                self.hidden_encoders = self.add_weight(
                    shape=(self.units, 1), initializer=tf.initializers.zeros()
                )
                self.memory_encoders = self.add_weight(
                    shape=(self.order, 1), initializer=tf.initializers.zeros()
                )
                self.input_kernel = self.add_weight(
                    shape=(input_dim, self.units), initializer=tf.initializers.zeros()
                )
                self.hidden_kernel = self.add_weight(
                    shape=(self.units, self.units), initializer=tf.initializers.zeros()
                )
                self.memory_kernel = self.add_weight(
                    shape=(self.order, self.units),
                    initializer=tf.initializers.glorot_normal(),
                )
                self.AT = self.add_weight(
                    shape=(self.order, self.order),
                    initializer=tf.initializers.constant(self._A.T),
                    trainable=False,
                )
                self.BT = self.add_weight(
                    shape=(1, self.order),
                    initializer=tf.initializers.constant(self._B.T),
                    trainable=False,
                )

            def call(self, inputs, states):
                """Compute cell output and state updates."""

                h_prev, m_prev = states

                # compute u_t from the above diagram
                u = (
                    tf.matmul(inputs, self.input_encoders)
                    + tf.matmul(h_prev, self.hidden_encoders)
                    + tf.matmul(m_prev, self.memory_encoders)
                )

                # compute updated memory state vector (m_t in diagram)
                m = tf.matmul(m_prev, self.AT) + tf.matmul(u, self.BT)

                # compute updated hidden state vector (h_t in diagram)
                h = tf.nn.tanh(
                    tf.matmul(inputs, self.input_kernel)
                    + tf.matmul(h_prev, self.hidden_kernel)
                    + tf.matmul(m, self.memory_kernel)
                )

                return h, [h, m]

        with nengo.Network(seed=0) as net:
            # remove some unnecessary features to speed up the training
            # we could set use_loop=False as well here, but leaving it for parity
            # with native_nengo
            nengo_dl.configure_settings(stateful=False)

            net.inp = nengo.Node(np.zeros(theta))

            rnn = nengo_dl.Layer(
                tf.keras.layers.RNN(
                    LMUCell(units=212, order=256, theta=theta, dtype=dtype),
                    return_sequences=False,
                )
            )(net.inp, shape_in=(theta, input_d))

            out = nengo.Node(size_in=10)
            nengo.Connection(rnn, out, transform=nengo_dl.dists.Glorot(), synapse=None)

            net.p = nengo.Probe(out)

    return net


def run_profile(
    net, train=False, n_steps=150, do_profile=True, reps=1, dtype="float32", **kwargs
):
    """
    Run profiler on a benchmark network.

    Parameters
    ----------
    net : `~nengo.Network`
        The nengo Network to be profiled.
    train : bool
        If True, profile the `.Simulator.fit` function. Otherwise, profile the
        `.Simulator.run` function.
    n_steps : int
        The number of timesteps to run the simulation.
    do_profile : bool
        Whether or not to run profiling
    reps : int
        Repeat the run this many times (only profile data from the last
        run will be kept).
    dtype : str
        Simulation dtype (e.g. "float32")

    Returns
    -------
    exec_time : float
        Time (in seconds) taken to run the benchmark, taking the minimum over
        ``reps``.

    Notes
    -----
    kwargs will be passed on to `.Simulator`
    """

    exec_time = 1e10
    n_batches = 1

    with net:
        nengo_dl.configure_settings(inference_only=not train, dtype=dtype)

    with nengo_dl.Simulator(net, **kwargs) as sim:
        if hasattr(net, "inp"):
            x = {
                net.inp: np.random.randn(
                    sim.minibatch_size * n_batches, n_steps, net.inp.size_out
                )
            }
        elif hasattr(net, "inp_a"):
            x = {
                net.inp_a: np.random.randn(
                    sim.minibatch_size * n_batches, n_steps, net.inp_a.size_out
                ),
                net.inp_b: np.random.randn(
                    sim.minibatch_size * n_batches, n_steps, net.inp_b.size_out
                ),
            }
        else:
            x = None

        if train:
            y = {
                net.p: np.random.randn(
                    sim.minibatch_size * n_batches, n_steps, net.p.size_in
                )
            }

            sim.compile(tf.optimizers.SGD(0.001), loss=tf.losses.mse)

            # run once to eliminate startup overhead
            start = timeit.default_timer()
            sim.fit(x, y, epochs=1, n_steps=n_steps)
            print("Warmup time:", timeit.default_timer() - start)

            for _ in range(reps):
                if do_profile:
                    tf.profiler.experimental.start("profile")
                start = timeit.default_timer()
                sim.fit(x, y, epochs=1, n_steps=n_steps)
                exec_time = min(timeit.default_timer() - start, exec_time)
                if do_profile:
                    tf.profiler.experimental.stop()

        else:
            # run once to eliminate startup overhead
            start = timeit.default_timer()
            sim.predict(x, n_steps=n_steps)
            print("Warmup time:", timeit.default_timer() - start)

            for _ in range(reps):
                if do_profile:
                    tf.profiler.experimental.start("profile")
                start = timeit.default_timer()
                sim.predict(x, n_steps=n_steps)
                exec_time = min(timeit.default_timer() - start, exec_time)
                if do_profile:
                    tf.profiler.experimental.stop()

    exec_time /= n_batches

    print("Execution time:", exec_time)

    return exec_time


@click.group(chain=True)
def main():
    """Command-line interface for benchmarks."""


@main.command()
@click.pass_obj
@click.option("--benchmark", default="cconv", help="Name of benchmark network")
@click.option("--dimensions", default=128, help="Number of dimensions")
@click.option("--neurons_per_d", default=64, help="Neurons per dimension")
@click.option("--neuron_type", default="RectifiedLinear", help="Nengo neuron model")
@click.option(
    "--kwarg",
    type=str,
    multiple=True,
    help="Arbitrary kwarg to pass to benchmark network (key=value)",
)
def build(obj, benchmark, dimensions, neurons_per_d, neuron_type, kwarg):
    """Builds one of the benchmark networks"""

    # get benchmark network by name
    benchmark = globals()[benchmark]

    # get the neuron type object from string class name
    try:
        neuron_type = getattr(nengo, neuron_type)()
    except AttributeError:
        neuron_type = getattr(nengo_dl, neuron_type)()

    # set up kwargs
    kwargs = dict((k, int(v)) for k, v in [a.split("=") for a in kwarg])

    # add the special cli kwargs if applicable; note we could just do
    # everything through --kwarg, but it is convenient to have a
    # direct option for the common arguments
    params = inspect.signature(benchmark).parameters
    for kw in ("benchmark", "dimensions", "neurons_per_d", "neuron_type"):
        if kw in params:
            kwargs[kw] = locals()[kw]

    # build benchmark and add to context for chaining
    print(
        f"Building {nengo_dl.utils.function_name(benchmark, sanitize=False)} "
        f"with {kwargs}"
    )

    obj["net"] = benchmark(**kwargs)


@main.command()
@click.pass_obj
@click.option(
    "--train/--no-train",
    default=False,
    help="Whether to profile training (as opposed to running) the network",
)
@click.option(
    "--n_steps", default=150, help="Number of steps for which to run the simulation"
)
@click.option("--batch_size", default=1, help="Number of inputs to the model")
@click.option(
    "--device",
    default="/gpu:0",
    help="TensorFlow device on which to run the simulation",
)
@click.option(
    "--unroll", default=25, help="Number of steps for which to unroll the simulation"
)
@click.option(
    "--time-only",
    is_flag=True,
    default=False,
    help="Only count total time, rather than profiling internals",
)
def profile(obj, train, n_steps, batch_size, device, unroll, time_only):
    """Runs profiling on a network (call after 'build')"""

    if "net" not in obj:
        raise ValueError("Must call `build` before `profile`")

    obj["time"] = run_profile(
        obj["net"],
        do_profile=not time_only,
        train=train,
        n_steps=n_steps,
        minibatch_size=batch_size,
        device=device,
        unroll_simulation=unroll,
    )


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
