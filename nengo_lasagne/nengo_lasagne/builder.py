import warnings
from collections import OrderedDict

import nengo
import nengo.builder
import lasagne as lgn
import theano
import theano.tensor as T
import numpy as np

from nengo.utils.progress import ProgressTracker, TerminalProgressBar

import nengo_lasagne
from nengo_lasagne import layers


class Model:
    def __init__(self, network):
        self.network = network

        # check that it's a valid network
        print("checking network")
        self.check_network(network)

        # create the lasagne network
        print("building network")
        self.build_network(network)

    def check_network(self, net):
        for ens in net.all_ensembles:
            # can only use a subset of neuron types
            if type(ens.neuron_type) not in nengo_lasagne.nl_map:
                raise TypeError("Unsupported nonlinearity (%s)" %
                                ens.neuron_type)

            if ens.noise is not None:
                warnings.warn("Ignoring noise parameter on ensemble")
            if ens.seed is not None:
                warnings.warn("Ignoring seed parameter on ensemble")

        for node in net.all_nodes:
            pass
            # # only input nodes or passthrough nodes
            # if node.size_in != 0 and node.output is not None:
            #     raise ValueError("Only input nodes or passthrough nodes are "
            #                      "allowed")

        for conn in net.all_connections:
            if conn.synapse is not None:
                warnings.warn("Ignoring synapse parameter on connection")
            if conn.learning_rule_type is not None:
                warnings.warn("Ignoring learning rule on connection")
            if conn.seed is not None:
                warnings.warn("Ignoring seed parameter on connection")

    def build_network(self, net):
        self.params = OrderedDict()

        # build nodes
        for node in net.all_nodes:
            self.params[node] = layers.NodeLayer(node)

        lgn_inputs = [self.params[o].input.input_var for o in net.all_nodes
                      if isinstance(self.params[o].input,
                                    lgn.layers.InputLayer)]

        self.batch_size, self.seq_len, _ = lgn_inputs[0].shape

        # build ensembles
        for ens in net.all_ensembles:
            self.params[ens] = layers.EnsembleLayer(ens)

        # build connections
        # note: we need to process all the recurrent connections first (so that
        # all the ensembles have been set to recurrent before we start
        # connecting them together)
        for conn in net.all_connections:
            if self.is_recurrent(conn):
                self.params[self.base_obj(conn.pre)].make_recurrent(conn, self)
        for conn in net.all_connections:
            if not self.is_recurrent(conn):
                self.add_connection(conn)

        # probe function
        lgn_probes = []
        for probe in net.all_probes:
            if isinstance(probe.target, nengo.Node):
                target = self.params[probe.target].output
            elif isinstance(probe.target, nengo.Ensemble):
                # TODO: create decoded connection and probe that
                raise NotImplementedError()
            elif isinstance(probe.target, nengo.ensemble.Neurons):
                target = self.params[probe.target.ensemble].output

            target = lgn.layers.ReshapeLayer(target, (self.batch_size,
                                                      self.seq_len, -1))

            lgn_probes += [target]

        self.probe_func = theano.function(
            lgn_inputs, lgn.layers.get_output(lgn_probes, deterministic=True))

    def train(self, train_inputs, train_targets, optimizer=lgn.updates.adagrad,
              minibatch_size=None, n_epochs=1000, optimizer_kwargs=None,
              objective=lgn.objectives.squared_error):
        print("training network")

        # loss function
        lgn_outputs = lgn.layers.get_output(
            [lgn.layers.ReshapeLayer(self.params[o].output,
                                     (self.batch_size, self.seq_len,
                                      self.params[o].output.num_units))
             for o in train_targets],
            deterministic=False)
        target_vars = [T.ftensor3("%s_targets" % o)
                       for o in train_targets]
        losses = [objective(o, t).mean()
                  for o, t in zip(lgn_outputs, target_vars)]

        # sum the losses for all the outputs to get the overall objective
        if len(losses) == 1:
            loss = losses[0]
        else:
            loss = T.add(*losses)

        # compile training update function
        params = lgn.layers.get_all_params([self.params[o].output
                                            for o in train_targets],
                                           trainable=True)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        updates = optimizer(loss, params, **optimizer_kwargs)
        self.train_func = theano.function(
            [self.params[x].input.input_var for x in train_inputs] +
            target_vars, loss, updates=updates)

        # print("layers", lgn.layers.get_all_layers([self.params[o].output
        #                                     for o in train_targets]))
        # print("params", lgn.layers.get_all_params([self.params[o].output
        #                                     for o in train_targets]))

        # run training epochs
        with ProgressTracker(n_epochs, TerminalProgressBar()) as progress:
            n_inputs = len(list(train_inputs.values())[0])
            minibatch_size = minibatch_size or n_inputs
            for _ in range(n_epochs):
                indices = np.random.permutation(n_inputs)
                for start in range(0, n_inputs - minibatch_size + 1,
                                   minibatch_size):
                    minibatch = indices[start:start + minibatch_size]

                    self.train_func(*(
                        [train_inputs[x][minibatch] for x in train_inputs] +
                        [train_targets[x][minibatch] for x in train_targets]))

                progress.step()

        print("training complete")

    def add_connection(self, conn):
        # get lasagne layer corresponding to pre
        pre = self.params[self.base_obj(conn.pre)].output

        layer = layers.ConnectionLayer(conn, pre, self)

        # get layer corresponding to post
        post = conn.post_obj
        if isinstance(post, nengo.Node):
            post = self.params[post].input
        elif isinstance(post, nengo.Ensemble):
            post = self.params[post].vector_input
        elif isinstance(post, nengo.ensemble.Neurons):
            post = self.params[post.ensemble].input
        else:
            raise ValueError("Unknown post type (%s)" % type(conn.post))

        post.add_incoming(layer.output)
        # TODO: put a dropout layer in here?

    def is_recurrent(self, conn):
        if self.base_obj(conn.pre) is self.base_obj(conn.post):
            # TODO: support slicing on recurrent connections
            if (isinstance(conn.pre, nengo.base.ObjView) or
                    isinstance(conn.post, nengo.base.ObjView)):
                raise NotImplementedError("Cannot slice recurrent connections")

            if isinstance(conn.pre_obj, nengo.Node):
                raise NotImplementedError(
                    "Cannot do recurrent connection on node")

            return True

        return False

    def base_obj(self, obj):
        if isinstance(obj, nengo.base.ObjView):
            obj = obj.obj

        if isinstance(obj, nengo.ensemble.Neurons):
            obj = obj.ensemble

        return obj
