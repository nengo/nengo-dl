from functools import partial
import itertools
import os
import pickle
import sys
import subprocess
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import click
import matplotlib.pyplot as plt
import nengo
from nengo import spa
import nengo_dl
from nengo_dl import graph_optimizer, benchmarks
import nengo_ocl
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

# spaun needs to be downloaded from https://github.com/drasmuss/spaun2.0, and
# manually added to python path
spaun_dir = os.path.join(os.path.dirname(__file__), "spaun2.0")
if not os.path.exists(spaun_dir):
    subprocess.call("git clone https://github.com/drasmuss/spaun2.0",
                    shell=True)
sys.path.append(spaun_dir)
from _spaun.configurator import cfg
from _spaun.vocabulator import vocab
from _spaun.experimenter import experiment
from _spaun.modules.vision.data import vis_data
from _spaun.modules.motor.data import mtr_data
from _spaun.spaun_main import Spaun


def filter_results(results, **kwargs):
    return [x["relative_time"] if "relative_time" in x else x["times"]
            for x in results if all(x[k] == v for k, v in kwargs.items())]


def bootstrap_ci(data, alpha=0.95, n_samples=1000, func=np.mean):
    samples = sorted(
        func(np.random.choice(data, replace=True, size=len(data)))
        for _ in range(n_samples))
    lower = int(n_samples * (1 - alpha) / 2)
    upper = int(n_samples * (alpha + (1 - alpha) / 2))
    return func(data), samples[lower], samples[upper]


def build_spaun(dimensions):
    vocab.sp_dim = dimensions
    cfg.mtr_arm_type = None

    cfg.set_seed(1)
    experiment.initialize('A', vis_data.get_image_ind,
                          vis_data.get_image_label,
                          cfg.mtr_est_digit_response_time, cfg.rng)
    vocab.initialize(experiment.num_learn_actions, cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    with Spaun() as net:
        nengo_dl.configure_settings(trainable=False, simplifications=[
            graph_optimizer.remove_constant_copies,
            graph_optimizer.remove_unmodified_resets,
            # graph_optimizer.remove_zero_incs,
            graph_optimizer.remove_identity_muls
        ])

    return net


@click.group()
@click.pass_context
@click.option("--load/--no-load", default=False, help="Load results from file")
@click.option("--reps", default=5, help="Number of data points to collect")
@click.option("--show/--no-show", default=True, help="Show plots")
@click.option("--device", default="/gpu:0",
              help="TensorFlow device to use for NengoDL")
@click.option("--save", default=None, type=str,
              help="Save figures with given file format")
def main(ctx, load, reps, show, device, save):
    ctx.obj["load"] = load
    ctx.obj["reps"] = reps
    ctx.obj["device"] = device
    ctx.obj["save"] = save


@main.resultcallback()
def main_callback(_, show, **kwargs):
    if show:
        plt.show()


@main.command()
@click.pass_context
@click.option("--batch", default=1, help="Number of batch elements")
@click.option("--n_neurons", default=9984,
              help="Number of neurons per ensemble")
def compare_backends(ctx, batch, n_neurons):
    load = ctx.obj["load"]
    reps = ctx.obj["reps"]
    device = ctx.obj["device"]
    save = ctx.obj["save"]

    bench_names = ["integrator", "cconv", "basal_ganglia", "pes"]
    n_range = [n_neurons]
    d_range = [64, 128, 192]
    neuron_types = [nengo.RectifiedLinear()]
    backends = ["nengo_dl", "nengo_ocl", "nengo"]
    sim_time = 5.0

    params = list(itertools.product(
        bench_names, n_range, d_range, neuron_types, backends))

    if load:
        with open("compare_backends_%d_data_saved.pkl" % batch, "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"times": [], "benchmark": bench, "n_neurons": n_neurons,
                    "dimensions": dimensions, "neuron_type": neuron_type,
                    "backend": backend}
                   for bench, n_neurons, dimensions, neuron_type, backend
                   in params]

    if reps > 0:
        for i, (bench, n_neurons, dimensions, neuron_type,
                backend) in enumerate(params):
            print("%d/%d: %s %s %s %s %s" % (
                i + 1, len(params), bench, n_neurons, dimensions, neuron_type,
                backend))

            net = getattr(benchmarks, bench)(
                dimensions=dimensions, neurons_per_d=n_neurons // dimensions,
                neuron_type=neuron_type)

            with net:
                nengo_dl.configure_settings(trainable=False)

            if "nengo_dl" in backend:
                sim = nengo_dl.Simulator(
                    net, unroll_simulation=25, minibatch_size=batch,
                    device=device, progress_bar=False)
            elif backend == "nengo":
                sim = nengo.Simulator(net, progress_bar=False, optimize=True)
            elif backend == "nengo_ocl":
                sim = nengo_ocl.Simulator(net, progress_bar=False)

            with sim:
                # run once to eliminate startup overhead
                sim.run(0.1, progress_bar=False)

                for _ in range(reps):
                    start = time.time()
                    for b in range(1 if "nengo_dl" in backend else batch):
                        if b > 0:
                            sim.reset()
                        sim.run(sim_time, progress_bar=False)
                    results[i]["times"].append(
                        (time.time() - start) / sim_time)

            print("   ", min(results[i]["times"]), max(results[i]["times"]),
                  np.mean(results[i]["times"]))

        with open("compare_backends_%d_data.pkl" % batch, "wb") as f:
            pickle.dump(results, f)

    # plotting
    subplots = int(np.ceil(np.sqrt(len(bench_names))))
    f, axes = plt.subplots(subplots, subplots, sharey=True, sharex=False,
                           figsize=(5 * subplots, 5 * subplots),
                           gridspec_kw={
                               "hspace": 0.2, "top": 0.95, "bottom": 0.05,
                               "left": 0.07, "right": 0.95})
    n_bars = len(d_range)
    neuron_type = nengo.RectifiedLinear()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_max = 2.5 * batch
    for k, m in enumerate(bench_names):
        subplot_idx = (k // subplots, k % subplots)
        x_pos = np.arange(n_bars)
        for j, b in enumerate(backends):
            bottoms = np.zeros(n_bars)
            c = 0
            for n in n_range:
                data = np.asarray([bootstrap_ci(t) for t in filter_results(
                    results, benchmark=m, neuron_type=neuron_type,
                    n_neurons=n, backend=b)])

                axes[subplot_idx].bar(x_pos, data[:, 0],
                                      yerr=abs(np.transpose(
                                          data[:, 1:] - data[:, [0]])),
                                      width=0.5, bottom=bottoms,
                                      color=colours[(j + 1) % len(backends)])

                for i, d in enumerate(data[:, 0]):
                    if d > y_max:
                        axes[subplot_idx].annotate(
                            "%.1f" % d, (x_pos[i], y_max * 0.9),
                            ha="center", va="center", rotation="vertical",
                            color="white")

                bottoms += data[:, 0]
                c += 1
            x_pos += n_bars + 1

        axes[subplot_idx].set_title("%s" % m)
        if k == 0 and len(n_range) > 1:
            axes[subplot_idx].legend(["N=%d" % n for n in n_range])
        axes[subplot_idx].set_xticks(np.concatenate(
            [np.arange(i * (n_bars + 1), i * (n_bars + 1) + n_bars)
             for i in range(len(backends))]))
        axes[subplot_idx].set_xticklabels([t for _ in range(len(backends))
                                           for t in d_range])
        for i, b in enumerate(backends):
            axes[subplot_idx].annotate(
                b, (((n_bars - 1) / 2 + (n_bars + 1) * i + 1) /
                    ((n_bars + 1) * len(backends)),
                    -0.1),
                xycoords="axes fraction", ha="center")

        axes[subplot_idx].set_ylim([0, y_max])
        axes[subplot_idx].set_xlim([-1, (n_bars + 1) * len(backends) - 1])

        if k % subplots == 0:
            axes[subplot_idx].set_ylabel("real time / simulated time")

    if save:
        plt.savefig("compare_backends_%d.%s" % (batch, save))


@main.command()
@click.pass_context
@click.option("--dimensions", default=128,
              help="Dimensionality of spaun model")
def compare_optimizations(ctx, dimensions):
    load = ctx.obj["load"]
    reps = ctx.obj["reps"]
    device = ctx.obj["device"]
    save = ctx.obj["save"]

    # optimizations to apply (simplifications, merging, sorting, unroll)
    params = [
        (False, False, False, False),
        (False, False, False, True),
        (False, True, False, True),
        (False, True, True, True),
        (True, True, True, True),
    ]
    # params = list(itertools.product((False, True), repeat=4))

    if load:
        with open("compare_optimizations_%d_data_saved.pkl" % dimensions,
                  "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"times": [], "simplifications": simp, "planner": plan,
                    "sorting": sort, "unroll": unro}
                   for simp, plan, sort, unro in params]

    if reps > 0:
        net = build_spaun(dimensions)
        model = nengo.builder.Model(
            dt=0.001, builder=nengo_dl.builder.NengoBuilder())
        model.build(net)

        print("neurons", net.n_neurons)
        print("ensembles", len(net.all_ensembles))
        print("connections", len(net.all_connections))

        for i, (simp, plan, sort, unro) in enumerate(params):
            print("%d/%d: %s %s %s %s" % (i + 1, len(params), simp, plan, sort,
                                          unro))
            with net:
                config = dict()
                config["simplifications"] = (
                    [graph_optimizer.remove_constant_copies,
                     graph_optimizer.remove_unmodified_resets,
                     # graph_optimizer.remove_zero_incs,
                     graph_optimizer.remove_identity_muls] if simp else
                    [])

                config["planner"] = (graph_optimizer.tree_planner if plan else
                                     graph_optimizer.greedy_planner)

                config["sorter"] = (graph_optimizer.order_signals if sort else
                                    graph_optimizer.noop_order_signals)

                nengo_dl.configure_settings(**config)

            with nengo_dl.Simulator(
                    None, model=model, unroll_simulation=50 if unro else 1,
                    device=device) as sim:
                sim.run(0.1)

                sim_time = 1.0

                for _ in range(reps):
                    start = time.time()
                    sim.run(sim_time)
                    results[i]["times"].append(
                        (time.time() - start) / sim_time)

            print("   ", min(results[i]["times"]), max(results[i]["times"]),
                  np.mean(results[i]["times"]))

        with open("compare_optimizations_%d_data.pkl" % dimensions, "wb") as f:
            pickle.dump(results, f)

    data = np.asarray([bootstrap_ci(x) for x in filter_results(results)])

    plt.figure()

    alphas = np.linspace(0.5, 1, len(results))
    colour = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    for i in range(len(results)):
        plt.bar([i], [data[i, 0]],
                yerr=abs(data[i, 1:] - data[i, [0]])[:, None],
                log=True, alpha=alphas[i], color=colour)

    labels = []
    for r in results:
        lab = "merging\n"
        if r["unroll"]:
            lab += "unrolling\n"
        if r["planner"]:
            lab += "planning\n"
        if r["sorting"]:
            lab += "sorting\n"
        if r["simplifications"]:
            lab += "simplifications\n"

        labels.append(lab[:-1])
    plt.xticks(np.arange(len(results)), labels, rotation="vertical")
    plt.ylabel("real time / simulated time")

    plt.tight_layout()

    if save:
        plt.savefig("compare_optimizations_%d.%s" % (dimensions, save))


@main.command()
@click.pass_context
@click.option("--dimensions", default=4, help="Dimensionality of spaun model")
def compare_simplifications(ctx, dimensions):
    load = ctx.obj["load"]
    reps = ctx.obj["reps"]
    device = ctx.obj["device"]

    simplifications = [
        graph_optimizer.remove_constant_copies,
        graph_optimizer.remove_unmodified_resets,
        graph_optimizer.remove_zero_incs,
        graph_optimizer.remove_identity_muls
    ]

    params = list(
        itertools.product((False, True), repeat=len(simplifications)))

    if load:
        with open("compare_simplifications_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = [
            dict([("times", [])] + [
                (s.__name__, p[i]) for i, s in enumerate(simplifications)])
            for j, p in enumerate(params)]

    net = build_spaun(dimensions)
    model = nengo.builder.Model(
        dt=0.001, builder=nengo_dl.builder.NengoBuilder())
    model.build(net)

    if reps > 0:
        for j, p in enumerate(params):
            simps = []
            for i, s in enumerate(p):
                if s:
                    simps.append(simplifications[i])

            with net:
                nengo_dl.configure_settings(simplifications=simps)

            print("%d/%d" % (j + 1, len(params)), [x.__name__ for x in simps])

            with nengo_dl.Simulator(
                    None, model=model, unroll_simulation=1, device=device,
                    progress_bar=False) as sim:
                sim.run(0.1, progress_bar=False)

                sim_time = 1.0
                for _ in range(reps):
                    start = time.time()
                    sim.run(sim_time, progress_bar=False)
                    results[j]["times"].append(
                        (time.time() - start) / sim_time)

            print("   ", min(results[j]["times"]), max(results[j]["times"]),
                  np.mean(results[j]["times"]))

        with open("compare_simplifications_data.pkl", "wb") as f:
            pickle.dump(results, f)


@main.command()
@click.pass_context
@click.option("--n_epochs", default=10, help="Number of training epochs")
def spiking_mnist(ctx, n_epochs):
    load = ctx.obj["load"]
    reps = ctx.obj["reps"]

    def build_network(neuron_type, ens_params):
        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=False)

            inp = nengo.Node([0] * 28 * 28)

            x = nengo_dl.tensor_layer(
                inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
                kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(26, 26, 32),
                filters=64, kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
                pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(12, 12, 64),
                filters=128, kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
                pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

        return net, inp, x

    if load:
        with open("spiking_mnist_data_saved.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {"pre": [], "post": [], "spiking": []}

    data = mnist.read_data_sets("MNIST_data/", one_hot=True)
    minibatch_size = 200

    # construct the rate network
    net, inp, out = build_network(
        nengo.LIFRate(amplitude=0.01),
        dict(max_rates=nengo.dists.Choice([100]),
             intercepts=nengo.dists.Choice([0]))
    )
    with net:
        out_p = nengo.Probe(out)

    train_inputs = {inp: data.train.images[:, None, :]}
    train_targets = {out_p: data.train.labels[:, None, :]}
    test_inputs = {inp: data.test.images[:, None, :]}
    test_targets = {out_p: data.test.labels[:, None, :]}

    # construct the spiking network
    spk_net, spk_inp, spk_out = build_network(
        nengo.LIF(amplitude=0.01),
        dict(max_rates=nengo.dists.Choice([100]),
             intercepts=nengo.dists.Choice([0]))
    )
    with spk_net:
        spk_out_p = nengo.Probe(spk_out, synapse=0.1)

    n_steps = 50
    test_inputs_time = {
        spk_inp: np.tile(data.test.images[:, None, :], (1, n_steps, 1))}
    test_targets_time = {spk_out_p: np.tile(v, (1, n_steps, 1)) for v in
                         test_targets.values()}

    for _ in range(reps):
        # construct the simulator
        with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
            def objective(x, y):
                return tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=x, labels=y)

            opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

            def classification_error(outputs, targets):
                return 100 * tf.reduce_mean(
                    tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                                         tf.argmax(targets[:, -1], axis=-1)),
                            tf.float32))

            results["pre"].append(sim.loss(
                test_inputs, test_targets, classification_error))
            print("error before training: %.2f%%" % results["pre"][-1])

            # run training
            sim.train(train_inputs, train_targets, opt, objective=objective,
                      n_epochs=n_epochs)

            # save the parameters to file
            sim.save_params("./mnist_params")

            results["post"].append(sim.loss(
                test_inputs, test_targets, classification_error))
            print("error after training: %.2f%%" % results["post"][-1])

        with nengo_dl.Simulator(spk_net, minibatch_size=minibatch_size,
                                unroll_simulation=10) as sim:
            sim.load_params("./mnist_params")

            results["spiking"].append(sim.loss(
                test_inputs_time, test_targets_time, classification_error))
            print("spiking neuron error: %.2f%%" % results["spiking"][-1])

        with open("spiking_mnist_data.pkl", "wb") as f:
            pickle.dump(results, f)

    print("pre", bootstrap_ci(results["pre"]))
    print("post", bootstrap_ci(results["post"]))
    print("spiking", bootstrap_ci(results["spiking"]))


@main.command()
@click.pass_context
@click.option("--dimensions", default=64, help="Dimensionality of vocabulary")
@click.option("--n_epochs", default=10, help="Number of training epochs")
def spa_optimization(ctx, dimensions, n_epochs):
    load = ctx.obj["load"]
    reps = ctx.obj["reps"]
    save = ctx.obj["save"]

    def get_binding_data(n_inputs, n_pairs, dims, seed, t_int, t_mem,
                         dt=0.001):
        int_steps = int(t_int / dt)
        mem_steps = int(t_mem / dt)
        n_steps = int_steps * n_pairs + mem_steps

        rng = np.random.RandomState(seed)
        vocab = spa.Vocabulary(dimensions=dims, rng=rng, max_similarity=1)

        # initialize arrays for input and output trajectories
        roles = np.zeros((n_inputs, n_steps, dims))
        fills = np.zeros((n_inputs, n_steps, dims))
        cues = np.zeros((n_inputs, n_steps, dims))
        binding = np.zeros((n_inputs, n_steps, dims))
        memory = np.zeros((n_inputs, n_steps, dims))
        output = np.zeros((n_inputs, n_steps, dims))

        # iterate through examples to be generated, fill arrays
        for n in range(n_inputs):
            role_names = ["ROLE_%d_%d" % (n, i) for i in range(n_pairs)]
            filler_names = ["FILLER_%d_%d" % (n, i) for i in range(n_pairs)]

            # each role/filler pair is presented for t_int seconds
            for i in range(n_pairs):
                roles[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    role_names[i]).v
                fills[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    filler_names[i]).v
                binding[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    "%s*%s" % (role_names[i], filler_names[i])).v

            # randomly select a cue
            cue_idx = rng.randint(n_pairs)

            # cue is presented during the memorization period
            cues[n, -mem_steps:, :] = vocab[role_names[cue_idx]].v

            # the goal is to output the associated filler during the
            # memorization phase
            # note: we use nan for the target prior to the memorization phase,
            # to indicate that it doesn't matter what the network output is
            output[n, -mem_steps:, :] = vocab[filler_names[cue_idx]].v
            output[n, :-mem_steps, :] = np.nan

        memory[...] = np.cumsum(binding, axis=1) * dt / t_int

        return roles, fills, cues, binding, memory, output, vocab

    def accuracy(outputs, targets, vocab=None):
        vocab_vectors = tf.constant(vocab.vectors, dtype=tf.float32)
        output = outputs[:, -1, :]
        sims = tf.matmul(vocab_vectors, tf.transpose(output))
        idxs = tf.argmax(sims, axis=0)
        match = tf.reduce_all(tf.equal(
            tf.gather(vocab_vectors, idxs), targets[:, -1]),
            axis=1)

        return tf.reduce_mean(tf.cast(match, tf.float32))

    def build_network(neurons_per_d, seed):
        with nengo.Network(seed=seed) as net:
            net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
            net.config[nengo.Ensemble].gain = nengo.dists.Uniform(0.5, 1)
            net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.1, 0.1)
            net.config[nengo.Connection].synapse = None

            net.role_inp = nengo.Node(np.zeros(dims))
            net.fill_inp = nengo.Node(np.zeros(dims))
            net.cue_inp = nengo.Node(np.zeros(dims))

            # circular convolution network to combine roles/fillers
            cconv = nengo.networks.CircularConvolution(neurons_per_d, dims)
            nengo.Connection(net.role_inp, cconv.input_a)
            nengo.Connection(net.fill_inp, cconv.input_b)

            # memory network to store the role/filler pairs
            memory = nengo.Ensemble(neurons_per_d * dims, dims)
            tau = 0.01
            nengo.Connection(cconv.output, memory, transform=tau / t_int,
                             synapse=tau)
            nengo.Connection(memory, memory, transform=1, synapse=tau)

            # another circular convolution network to extract the cued filler
            ccorr = nengo.networks.CircularConvolution(neurons_per_d, dims,
                                                       invert_b=True)
            nengo.Connection(memory, ccorr.input_a)
            nengo.Connection(net.cue_inp, ccorr.input_b)

            net.conv_probe = nengo.Probe(cconv.output, label="conv_probe")
            net.memory_probe = nengo.Probe(memory, label="memory_probe")
            net.output_probe = nengo.Probe(ccorr.output, label="output_probe")

        return net

    # we'll define a slightly modified version of mean squared error that
    # allows us to specify a weighting (so that we can specify a different
    # weight for each probe)
    def weighted_mse(output, target, weight=1):
        target = tf.where(tf.is_nan(target), output, target)
        return weight * tf.reduce_mean(tf.square(target - output))

    t_int = 0.01  # length of time to present each input pair
    t_mem = 0.03  # length of memorization period
    n_pairs = 2  # number of role/filler pairs in each input
    dims = dimensions  # dimensionality of semantic pointer vectors
    minibatch_size = 64
    optimizer = tf.train.RMSPropOptimizer(1e-4)

    params = [5, 10, 15, 20]

    if load:
        with open("spa_optimization_data_saved.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"pre_retrieval": [], "post_retrieval": [], "pre_mse": [],
                    "post_mse": [], "neurons_per_d": n} for n in params]

    n_results = len(results[0]["pre_retrieval"])
    for r in range(n_results, n_results + reps):
        print("=" * 30)
        print("REP %d" % r)

        seed = r

        # generate training data
        (train_roles, train_fills, train_cues, train_binding, train_memory,
         train_targets, _) = get_binding_data(8000, n_pairs, dims, seed, t_int,
                                              t_mem)
        # generate test data
        (test_roles, test_fills, test_cues, _, _, test_targets,
         test_vocab) = get_binding_data(1024, n_pairs, dims, seed + 1, t_int,
                                        t_mem)

        acc = partial(accuracy, vocab=test_vocab)

        for i, n in enumerate(params):
            print("neurons_per_d", n)

            net = build_network(n, seed)
            train_inputs = {net.role_inp: train_roles,
                            net.fill_inp: train_fills,
                            net.cue_inp: train_cues}
            train_outputs = {net.output_probe: train_targets,
                             net.conv_probe: train_binding,
                             net.memory_probe: train_memory}

            test_inputs = {net.role_inp: test_roles, net.fill_inp: test_fills,
                           net.cue_inp: test_cues}
            test_outputs = {net.output_probe: test_targets}

            with nengo_dl.Simulator(
                    net, seed=seed, minibatch_size=minibatch_size,
                    progress_bar=False) as sim:
                results[i]["pre_retrieval"].append(sim.loss(
                    test_inputs, test_outputs, acc))
                print('pre retrieval:', results[i]["pre_retrieval"][-1])

                results[i]["pre_mse"].append(sim.loss(
                    test_inputs, test_outputs, "mse"))
                print('pre mse:', results[i]["pre_mse"][-1])

                sim.train(train_inputs, train_outputs, optimizer,
                          n_epochs=n_epochs,
                          objective={net.output_probe: weighted_mse,
                                     net.conv_probe: partial(weighted_mse,
                                                             weight=0.25),
                                     net.memory_probe: partial(weighted_mse,
                                                               weight=0.25)})

                results[i]["post_mse"].append(sim.loss(
                    test_inputs, test_outputs, "mse"))
                print('post mse:', results[i]["post_mse"][-1])

                results[i]["post_retrieval"].append(sim.loss(
                    test_inputs, test_outputs, acc))
                print('post retrieval:', results[i]["post_retrieval"][-1])

        with open("spa_optimization_data.pkl", "wb") as f:
            pickle.dump(results, f)

    plt.figure()
    data = np.asarray([bootstrap_ci(x["pre_retrieval"]) for x in results])
    plt.plot(params, data[:, 0])
    plt.fill_between(params, data[:, 1], data[:, 2], alpha=0.5)

    data = np.asarray([bootstrap_ci(x["post_retrieval"]) for x in results])
    plt.plot(params, data[:, 0])
    plt.fill_between(params, data[:, 1], data[:, 2], alpha=0.5)

    plt.xlabel("neurons per dimension")
    plt.ylabel("retrieval accuracy")
    plt.legend(["before training", "after training"])

    plt.tight_layout()

    if save:
        plt.savefig("spa_optimization.%s" % save)


@main.command()
@click.pass_context
def all_figures(ctx):
    ctx.invoke(compare_backends)
    ctx.invoke(compare_backends, batch=10)
    ctx.invoke(compare_optimizations)
    ctx.invoke(spiking_mnist)
    ctx.invoke(spa_optimization)


@main.command()
@click.pass_context
def test(ctx):
    ctx.invoke(compare_backends, n_neurons=960)
    ctx.invoke(compare_optimizations, dimensions=1)
    ctx.invoke(spiking_mnist, n_epochs=1)
    ctx.invoke(spa_optimization, dimensions=2, n_epochs=1)


if __name__ == "__main__":
    # to generate the data + figures:
    # python whitepaper2018_plots.py all_figures

    # to generate the figures (pre-generated data):
    # python whitepaper2018_plots.py --load --reps 0 all_figures

    # to test the figure functions:
    # python whitepaper2018_plots.py --no-show --reps 1 test

    # to generate an individual figure
    # python whitepaper2018_plots.py <figure_name>

    main(obj={})
