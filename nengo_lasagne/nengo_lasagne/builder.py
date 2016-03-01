import warnings
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
    def __init__(self, network, nef_init=True):
        self.nef_init = nef_init

        # check that it's a valid network
        print("checking network")
        self.check_network(network)

        # create the lasagne network
        print("building network")
        self.build_network(network)

    def check_network(self, net):
        for ens in net.all_ensembles:
            # can only use a subset of neuron types
            if type(ens.neuron_type) not in nengo_lasagne.nonlinearity_map:
                raise TypeError("Unsupported nonlinearity (%s)" %
                                ens.neuron_type)

            if ens.noise is not None:
                warnings.warn("Ignoring noise parameter on ensemble")
            if ens.seed is not None:
                warnings.warn("Ignoring seed parameter on ensemble")

        for node in net.all_nodes:
            # only input nodes or passthrough nodes
            if node.size_in != 0 and node.output is not None:
                raise ValueError("Only input nodes or passthrough nodes are "
                                 "allowed")

        for conn in net.all_connections:
            # TODO: support recurrent connections
            if conn.pre_obj is conn.post_obj:
                raise NotImplementedError("Recurrent connections unsupported")

            if conn.synapse is not None:
                warnings.warn("Ignoring synapse parameter on connection")
            if conn.learning_rule_type is not None:
                warnings.warn("Ignoring learning rule on connection")
            if conn.seed is not None:
                warnings.warn("Ignoring seed parameter on connection")

    def build_network(self, net):
        self.params = OrderedDict()
        self.inputs = OrderedDict()

        for node in net.all_nodes:
            if node.size_in == 0:
                self.params[node] = lgn.layers.InputLayer(
                    (None, node.size_out), name=node.label)
                self.inputs[node] = self.params[node]
            else:
                self.params[node] = layers.ExpandableSumLayer([],
                                                              node.size_in)

        for ens in net.all_ensembles:
            self.params[ens] = layers.EnsembleLayer(ens)

        for conn in net.all_connections:
            self.add_connection(conn)

        # probe function
        lgn_inputs = [o.input_var for o in self.inputs.values()]

        lgn_probes = []
        for probe in net.all_probes:
            if isinstance(probe.target, nengo.Node):
                lgn_probes += [self.params[probe.target]]
            elif isinstance(probe.target, nengo.Ensemble):
                # TODO: create decoded connection and probe that
                raise NotImplementedError()
            elif isinstance(probe.target, nengo.ensemble.Neurons):
                lgn_probes += [self.params[probe.target].output]

        self.probe_func = theano.function(lgn_inputs,
                                          lgn.layers.get_output(lgn_probes))

    def train_network(self, train_data=None, batch_size=100, n_epochs=1000,
                      l_rate=0.1):
        # run training
        print("training network")

        # loss function
        train_inputs, train_targets = train_data
        inputs = OrderedDict([(o, self.params[o]) for o in train_inputs])
        outputs = OrderedDict([(o, self.params[o]) for o in train_targets])

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
        updates = lgn.updates.adagrad(
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
        # TODO: synapses

        pre = conn.pre_obj
        pre_slice = conn.pre_slice

        post = conn.post_obj
        post_slice = conn.post_slice

        # get layer corresponding to pre
        if isinstance(pre, nengo.Node):
            pre = self.params[pre]
        elif isinstance(pre, nengo.Ensemble):
            pre = self.params[pre].output
        elif isinstance(pre, nengo.ensemble.Neurons):
            pre = self.params[pre.ensemble].output
        else:
            raise ValueError("Unknown pre type (%s)" % type(conn.pre))

        # apply pre slice
        if pre_slice != slice(None):
            pre = lgn.layers.SliceLayer(pre, pre_slice)

        # get layer corresponding to post
        if isinstance(post, nengo.Node):
            post = self.params[post]
        elif isinstance(post, nengo.Ensemble):
            post = self.params[post].decoded_input
        elif isinstance(post, nengo.ensemble.Neurons):
            post = self.params[post.ensemble].input
        else:
            raise ValueError("Unknown post type (%s)" % type(conn.post))

        if isinstance(conn.pre_obj, nengo.Node):
            # directly apply nonlinearity (note: no connection weights)
            if conn.function is None:
                incoming = pre
            else:
                # note: this won't work properly if conn.function doesn't
                # return a symbolic theano expression (e.g. if it uses numpy
                # functions or something)
                incoming = lgn.layers.ExpressionLayer(
                    pre, conn.function, output_shape=(None, conn.post.size_in))
        else:
            # get initial weights
            init_W = lgn.init.GlorotUniform()
            if self.nef_init and conn.function is not None:
                # note: we're ignoring the transform on the connection, unless
                # the function is also set. this is so that the transform can
                # be used to match up the pre/post shapes of connections (so
                # that Nengo doesn't complain), while still using lasagne's
                # initialization methods.
                init_W = self.compute_decoders(conn)

            # connection weight layer
            incoming = lgn.layers.DenseLayer(
                pre, conn.post.size_in, W=init_W, nonlinearity=None, b=None)

        if post_slice != slice(None):
            # zero-pad to get full shape (TODO: more efficient solution?)
            def pad_func(x):
                z = T.zeros((x.shape[0], post.num_units))
                return T.inc_subtensor(z[:, post_slice], x)

            incoming = lgn.layers.ExpressionLayer(
                incoming, pad_func, output_shape=(None, post.num_units))

        post.add_incoming(incoming)
        # TODO: put a dropout layer in here?

    def compute_decoders(self, conn):
        assert isinstance(conn.pre, nengo.Ensemble)

        pre = self.params[conn.pre_obj]

        eval_points = nengo.builder.connection.get_eval_points(
            self, conn, None)

        inputs = np.dot(eval_points, pre.encoders.T / conn.pre_obj.radius)
        targets = np.dot(np.apply_along_axis(conn.function, 1,
                                             eval_points[:, conn.pre_slice]),
                         conn.transform.T)
        if targets.ndim == 1:
            targets = targets[:, None]

        post_enc = None
        if conn.solver.weights:
            post_enc = self.params[conn.post_obj].encoders.T[conn.post_slice]
            post_enc /= conn.post_obj.radius

        decoders, _ = nengo.builder.connection.solve_for_decoders(
            conn.solver, conn.pre_obj.neuron_type, pre.gain, pre.bias, inputs,
            targets, np.random, post_enc)

        return decoders
