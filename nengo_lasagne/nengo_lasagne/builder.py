import warnings
from collections import OrderedDict

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
            if type(ens.neuron_type) not in nengo_lasagne.nl_map:
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
            # check if ensemble is recurrent
            recurrent = None
            for conn in net.all_connections:
                if (self.base_obj(conn.pre) is self.base_obj(ens) and
                        self.is_recurrent(conn)):
                    rec = conn
                    if rec is not None and recurrent is not None:
                        raise ValueError("Only one recurrent connection "
                                         "allowed per ensemble")
                    recurrent = recurrent or rec
            self.params[ens] = layers.EnsembleLayer(ens, recurrent=recurrent,
                                                    batch_size=self.batch_size)

        # build connections
        for conn in net.all_connections:
            if not self.is_recurrent(conn):
                self.add_connection(conn)

        # probe function
        lgn_probes = []
        for probe in net.all_probes:
            if isinstance(probe.target, nengo.Node):
                lgn_probes += [self.params[probe.target].output]
            elif isinstance(probe.target, nengo.Ensemble):
                # TODO: create decoded connection and probe that
                raise NotImplementedError()
            elif isinstance(probe.target, nengo.ensemble.Neurons):
                lgn_probes += [self.params[probe.target.ensemble].output]

        # print("params", lgn.layers.get_all_params(lgn_probes))
        # print(theano.printing.debugprint(self.eval))
        # print(theano.printing.debugprint(self.train))

        self.probe_func = theano.function(lgn_inputs,
                                          lgn.layers.get_output(lgn_probes))

    def train_network(self, train_data=None, batch_size=100, n_epochs=1000,
                      l_rate=0.1):
        # run training
        print("training network")

        # loss function
        train_inputs, train_targets = train_data

        lgn_outputs = lgn.layers.get_output(
            [lgn.layers.ReshapeLayer(self.params[o].output,
                                     (self.batch_size, self.seq_len,
                                      self.params[o].output.num_units))
             for o in train_targets])
        target_vars = [T.ftensor3("%s_targets" % o) for o in train_targets]
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
            loss, lgn.layers.get_all_params([self.params[o].output
                                             for o in train_targets]),
            learning_rate=l_rate)
        self.train = theano.function([self.params[x].input.input_var
                                      for x in train_inputs] + target_vars,
                                     loss, updates=updates)
        self.eval = theano.function([self.params[x].input.input_var
                                     for x in train_inputs], lgn_outputs)

        #         print "params", lgn.layers.get_all_params(outputs.values())
        #         print theano.printing.debugprint(self.eval)
        #         print theano.printing.debugprint(self.train)

        n_inputs = len(list(train_inputs.values())[0])
        for _ in range(n_epochs):
            indices = np.arange(n_inputs)
            np.random.shuffle(indices)
            for start in range(0, n_inputs - batch_size + 1, batch_size):
                minibatch = indices[start:start + batch_size]

                self.train(*([train_inputs[x][minibatch]
                              for x in train_inputs] +
                             [train_targets[x][minibatch]
                              for x in train_targets]))

    def add_connection(self, conn):
        # TODO: synapses

        # get lasagne layer corresponding to pre
        pre = self.params[self.base_obj(conn.pre)].output

        # apply pre slice
        if conn.pre_slice != slice(None):
            pre = lgn.layers.SliceLayer(pre, conn.pre_sice)

        # apply node nonlinearity
        if isinstance(conn.pre_obj, nengo.Node) and conn.function is not None:
            # note: this won't work properly if conn.function doesn't
            # return a symbolic theano expression (e.g. if it uses numpy
            # functions or something)
            pre = lgn.layers.ExpressionLayer(
                pre, conn.function, output_shape=(None, conn.post.size_in))

        # set up connection weight layer
        pre = lgn.layers.DenseLayer(
            pre, conn.post.size_in, W=self.get_weights(conn, pre),
            nonlinearity=None, b=None, name="%s W" % conn)

        # get layer corresponding to post
        post = conn.post_obj
        if isinstance(post, nengo.Node):
            post = self.params[post].input
        elif isinstance(post, nengo.Ensemble):
            post = self.params[post].decoded_input
        elif isinstance(post, nengo.ensemble.Neurons):
            post = self.params[post.ensemble].input
        else:
            raise ValueError("Unknown post type (%s)" % type(conn.post))

        # apply post slice
        if conn.post_slice != slice(None):
            # zero-pad to get full shape (TODO: more efficient solution?)
            def pad_func(x):
                z = T.zeros((x.shape[0], post.num_units))
                return T.inc_subtensor(z[:, conn.post_slice], x)

            pre = lgn.layers.ExpressionLayer(
                pre, pad_func, output_shape=(None, post.num_units))

        post.add_incoming(pre)
        # TODO: put a dropout layer in here?

    def get_weights(self, conn, pre):
        if not np.all(np.isfinite(conn.transform)):
            # use nan transform to indicate that we want the initial weights to
            # be randomized
            init_W = lgn.init.GlorotUniform()
        else:
            if isinstance(conn.pre_obj, nengo.Ensemble):
                # then we're dealing with decoders
                # note: the nef_init flag is just so that we can turn this
                # feature on/off to easily compare performance (without
                # having to go back and set the transforms to nans in
                # our model)
                if self.nef_init:
                    init_W = self.compute_decoders(conn)
                else:
                    init_W = lgn.init.GlorotUniform()
            else:
                # neurons/nodes, so weights are directly
                # specified with the transform parameter
                if conn.transform.ndim == 0:
                    # expand scalar to full matrix
                    init_W = np.ones((pre.output_shape[-1], conn.post.size_in),
                                     dtype=np.float32) * conn.transform
                else:
                    init_W = np.asarray(conn.transform, dtype=np.float32).T

        return init_W

    def compute_decoders(self, conn):
        assert isinstance(conn.pre, nengo.Ensemble)

        if conn.function is None:
            func = lambda x: x
        else:
            func = conn.function

        pre = self.params[conn.pre_obj]

        eval_points = nengo.builder.connection.get_eval_points(
            self, conn, None)

        inputs = np.dot(eval_points, pre.encoders.T / conn.pre_obj.radius)
        targets = np.dot(np.apply_along_axis(func, 1,
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

        return np.asarray(decoders, dtype=np.float32)

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
