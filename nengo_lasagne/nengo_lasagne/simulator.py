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

        else:
            self.model = model

        self.data = dict([(probe, None) for probe in network.all_probes])

        self.reset()

    def step(self):
        raise NotImplementedError()

    def run_steps(self, steps, inputs=None):
        self.steps = steps

        if inputs is None:
            # generate inputs by running Nodes
            input_vals = []
            for node in self.network.all_nodes:
                if node.size_in == 0:
                    if nengo.utils.compat.is_array_like(node.output):
                        input_vals += [np.tile(node.output, (steps, 1))]
                    elif callable(node.output):
                        input_vals += [[node.output(i * self.dt)
                                        for i in range(steps)]]
                    elif isinstance(node.output, nengo.processes.Process):
                        input_vals += [node.output.run_steps(steps,
                                                             dt=self.dt)]

            # cast all to float32
            input_vals = [np.asarray(x[None, ...], dtype=np.float32)
                          for x in input_vals]
        else:
            input_vals = []
            # we iterate over all_nodes because we need inputs to be
            # in the same order that probe_func expects them
            for node in self.network.all_nodes:
                if node in inputs:
                    assert inputs[node].shape[1] == steps
                    assert inputs[node].shape[2] == node.size_out
                    input_vals += [inputs[node]]

        output = self.model.probe_func(*input_vals)
        for i, probe in enumerate(self.network.all_probes):
            self.data[probe] = output[i]

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def trange(self):
        return np.arange(self.steps) * self.dt

    def run(self, t):
        self.run_steps(int(np.round(float(t) / self.dt)))

    def reset(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
