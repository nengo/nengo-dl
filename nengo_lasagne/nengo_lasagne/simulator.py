import lasagne as lgn
import nengo
import numpy as np

from nengo_lasagne import builder


class Simulator(object):
    def __init__(self, network, dt=0.001, seed=None, model=None):
        # TODO: support seed
        self.network = network
        self.dt = float(dt)
        self.closed = False

        if model is None:
            self.model = builder.Model(network)

            train_data = (network.config[Simulator].train_inputs,
                          network.config[Simulator].train_targets)
            batch_size = network.config[Simulator].batch_size
            n_epochs = network.config[Simulator].n_epochs
            l_rate = network.config[Simulator].l_rate

            self.model.train_network(train_data=train_data,
                                     batch_size=batch_size,
                                     n_epochs=n_epochs, l_rate=l_rate)
        else:
            self.model = model

        self.data = dict([(probe, None) for probe in network.all_probes])

        self.reset()

    def step(self):
        raise NotImplementedError()

    def run_steps(self, steps):
        self.n_steps += 1

        inputs = []
        for node in self.model.inputs:
            if nengo.utils.compat.is_array_like(node.output):
                inputs += [np.tile(node.output, (steps, 1))]
            elif callable(node.output):
                inputs += [[node.output(i * self.dt) for i in range(steps)]]
            elif isinstance(node.output, nengo.processes.Process):
                inputs += [node.output.run_steps(steps, dt=self.dt)]

        # cast all to float32
        inputs = [np.asarray(x, dtype=np.float32) for x in inputs]

        output = self.model.probe_func(*inputs)
        for i, probe in enumerate(self.network.all_probes):
            self.data[probe] = output[i]

    def run(self, t):
        self.run_steps(int(np.round(float(t) / self.dt)))

    def reset(self):
        self.n_steps = 0

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
