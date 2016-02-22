from collections import defaultdict, OrderedDict

import nengo
import nengo.builder
import lasagne as lgn
import theano
import theano.tensor as T
import numpy as np

import nengo_lasagne
from nengo_lasagne import layers


class Model:
    def __init__(self, network):
        # check that it's a valid network
        print("checking network")
        self.check_network(network)

        # create the lasagne network
        print("building network")
        self.build_network(network)

    def check_network(self, net):
        # TODO: throw warnings here about the parameters being ignored
        for ens in net.all_ensembles:
            # can only use a subset of neuron types
            assert type(ens.neuron_type) in nengo_lasagne.nonlinearity_map

        for node in net.all_nodes:
            # only input nodes or passthrough nodes
            assert node.size_in == 0 or node.output is None

        for conn in net.all_connections:
            # TODO: support recurrent connections
            assert conn.pre is not conn.post

    def build_network(self, net):
        self.lgn_objs = OrderedDict()
        self.inputs = OrderedDict()

        for node in net.all_nodes:
            if node.size_in == 0:
                self.lgn_objs[node] = lgn.layers.InputLayer(
                    (None, node.size_out), name=node.label)
                self.inputs[node] = self.lgn_objs[node]
            else:
                # TODO: add nonlinearity layer if node has output function?
                self.lgn_objs[node] = layers.ExpandableSumLayer([],
                                                                node.size_in)

        for ens in net.all_ensembles:
            self.lgn_objs[ens] = layers.EnsembleLayer(ens)

        for conn in net.all_connections:
            self.add_connection(conn)

        # probe function
        lgn_inputs = [o.input_var for o in self.inputs.values()]

        lgn_probes = []
        for probe in net.all_probes:
            if isinstance(probe.target, nengo.Node):
                lgn_probes += [self.lgn_objs[probe.target]]
            elif isinstance(probe.target, nengo.Ensemble):
                # TODO: create decoded connection and probe that
                raise NotImplementedError()
            elif isinstance(probe.target, nengo.ensemble.Neurons):
                lgn_probes += [self.lgn_objs[probe.target].output]

        self.probe_func = theano.function(lgn_inputs,
                                          lgn.layers.get_output(lgn_probes))

    def train_network(self, train_data=None, batch_size=100, n_epochs=1000,
                      l_rate=0.1):
        # run training
        print("training network")

        # loss function
        train_inputs, train_targets = train_data
        inputs = OrderedDict([(o, self.lgn_objs[o]) for o in train_inputs])
        outputs = OrderedDict([(o, self.lgn_objs[o]) for o in train_targets])

        lgn_outputs = lgn.layers.get_output(outputs.values())
        target_vars = [T.fmatrix("%s_targets" % o) for o in outputs]
        losses = [lgn.objectives.squared_error(o, t).mean()
                  for o, t in zip(lgn_outputs, target_vars)]

        # sum the losses for all the outputs to get the overall objective
        # (could do something more complicated here)
        if len(losses) == 1:
            loss = losses[0]
        else:
            loss = T.add(*losses)

        # training update
        updates = lgn.updates.sgd(
            loss, lgn.layers.get_all_params(outputs.values()),
            learning_rate=l_rate)
        self.train = theano.function([x.input_var for x in inputs.values()] +
                                     target_vars, loss, updates=updates)
        self.eval = theano.function([x.input_var for x in inputs.values()],
                                    lgn_outputs)

        #         print "params", lgn.layers.get_all_params(outputs.values())
        #         print theano.printing.debugprint(self.eval)
        #         print theano.printing.debugprint(self.train)

        n_inputs = len(train_inputs[list(inputs)[0]])
        for _ in range(n_epochs):
            indices = np.arange(n_inputs)
            np.random.shuffle(indices)
            for start in range(0, n_inputs - batch_size + 1, batch_size):
                minibatch = indices[start:start + batch_size]

                self.train(*([train_inputs[x][minibatch] for x in inputs] +
                             [train_targets[x][minibatch] for x in outputs]))

    def add_connection(self, conn):
        if isinstance(conn.pre, nengo.base.ObjView):
            pre_slice = conn.pre.slice
            pre = conn.pre.obj
        else:
            pre_slice = None
            pre = conn.pre

        if isinstance(conn.post, nengo.base.ObjView):
            post_slice = conn.post.slice
            post = conn.post.obj
        else:
            post_slice = None
            post = conn.post

        if isinstance(pre, nengo.Node):
            pre = self.lgn_objs[pre]
        elif isinstance(pre, nengo.Ensemble):
            pre = self.lgn_objs[pre].output
        elif isinstance(pre, nengo.ensemble.Neurons):
            pre = self.lgn_objs[pre.ensemble].output
        else:
            raise ValueError("Unknown pre type (%s)" % type(conn.pre))

        if pre_slice is not None:
            pre = lgn.layers.SliceLayer(pre, pre_slice)

        if isinstance(post, nengo.Node):
            post = self.lgn_objs[post]
        elif isinstance(post, nengo.Ensemble):
            post = self.lgn_objs[post].decoded_input
        elif isinstance(post, nengo.ensemble.Neurons):
            post = self.lgn_objs[post.ensemble].input
        else:
            raise ValueError("Unknown post type (%s)" % type(conn.post))

        if post_slice is not None:
            num_post = len(range(post.num_units)[post_slice])
            incoming = lgn.layers.DenseLayer(
                pre, len(range(post.num_units)[post_slice]), nonlinearity=None,
                b=None)

            # zero-pad to get full shape (TODO: more efficient solution?)
            def pad_func(x):
                z = T.zeros((x.shape[0], post.num_units))
                return T.inc_subtensor(z[:, post_slice], x)

            incoming = lgn.layers.ExpressionLayer(
                incoming, pad_func, output_shape=(None, post.num_units))

        else:
            incoming = lgn.layers.DenseLayer(pre, post.num_units,
                                             nonlinearity=None, b=None)

        # TODO: initialize the weights here based on function/transform
        # specified in connection
        post.add_incoming(incoming)
        # TODO: put a dropout layer in here?
