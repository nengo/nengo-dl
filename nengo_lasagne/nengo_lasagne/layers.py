import lasagne as lgn
import theano
import theano.tensor as T
import numpy as np
import nengo.builder

import nengo_lasagne


class ExpandableSumLayer(lgn.layers.ElemwiseSumLayer):
    def __init__(self, incomings, num_units, **kwargs):
        super(ExpandableSumLayer, self).__init__(incomings, **kwargs)

        self.num_units = num_units

    def add_incoming(self, incoming, coeff=1):
        self.input_shapes += [incoming if isinstance(incoming, tuple)
                              else incoming.output_shape]

        assert self.input_shapes[-1][-1] == self.num_units
        if len(self.input_shapes) > 1:
            assert self.input_shapes[-1] == self.input_shapes[-2]

        self.input_layers += [None if isinstance(incoming, tuple)
                              else incoming]

        self.coeffs += [coeff]

    def get_output_shape_for(self, input_shapes):
        if len(self.input_layers) == 0:
            return None, self.num_units

        return input_shapes[0][0], self.num_units


class GainLayer(lgn.layers.Layer):
    def __init__(self, incoming, gain=lgn.init.Constant(1.0),
                 trainable=True, **kwargs):
        super(GainLayer, self).__init__(incoming, **kwargs)

        self.gain = self.add_param(gain, (incoming.output_shape[-1],),
                                   name="gain",
                                   trainable=trainable)

    def get_output_for(self, input, **kwargs):
        return input * self.gain


class NodeLayer:
    def __init__(self, node):
        self.name = str(node)

        if isinstance(node, LasagneNode):
            # directly insert the layers into the model
            assert not node.time_input

            input_layer = node.inputs[0]

            if node.size_in == 0:
                self.input = input_layer
            else:
                # need to replace the input with an expandablesumlayer
                self.input = ExpandableSumLayer([], node.size_in)

                for l in lgn.layers.get_all_layers(node.layer):
                    if (hasattr(l, "input_layer") and l.input_layer is
                        input_layer):
                        l.input_layer = self.input
                    if (hasattr(l, "input_layers") and
                                input_layer in l.input_layers):
                        for i in range(len(l.input_layers)):
                            l.input_layers[i] = self.input

            self.output = node.layer
        else:
            if node.size_in == 0:
                self.input = lgn.layers.InputLayer((None, None,
                                                    node.size_out),
                                                   name=self.name)

                # flatten the batch/sequence dimensions
                self.output = lgn.layers.ReshapeLayer(
                    self.input, (-1, node.size_out),
                    name=self.name + "_reshape")
            else:
                self.input = ExpandableSumLayer([], node.size_in,
                                                name=self.name)
                self.output = self.input

                # in order to get this stuff to work need to figure out how
                # to handle the time input
                # if callable(node.output):
                #     output = node.output
                # elif isinstance(node.output, nengo.processes.Process):
                #     raise NotImplementedError()
                # else:
                #     raise TypeError("Unknown output type")
                #
                # self.output = lgn.layers.NonlinearityLayer(self.input,
                #                                            nonlinearity=output)


class LasagneNode(nengo.Node):
    def __init__(self, output, time_input=False, **kwargs):
        assert isinstance(output, lgn.layers.Layer)

        self.layer = output
        self.time_input = False

        layers = lgn.layers.get_all_layers(output)
        self.inputs = [x for x in layers
                       if isinstance(x, lgn.layers.InputLayer)]
        assert len(self.inputs) == 1

        self.compiled = theano.function(
            [x.input_var for x in self.inputs],
            lgn.layers.get_output(output, deterministic=True),
            allow_input_downcast=True)

        def func(t, x):
            if time_input:
                return self.compiled(np.concatenate((t, x))).squeeze()
            else:
                return self.compiled(x).squeeze()

        super(LasagneNode, self).__init__(output=func, **kwargs)


class EnsembleLayer:
    def __init__(self, ens):
        self.ens = ens
        self.name = str(ens)
        self.recurrent = False
        self._vector_input = None
        self._encoders = None
        self.eval_points = nengo.builder.ensemble.gen_eval_points(
            ens, ens.eval_points, np.random)

        # neuron input
        self.input = ExpandableSumLayer(
            [], ens.n_neurons, name=self.name + "_input")

        # add gain/bias
        gain, bias, _, _ = nengo.builder.ensemble.get_gain_bias(ens)
        gain = np.asarray(gain, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        self.gain_layer = GainLayer(self.input, gain=gain, trainable=True,
                                    name=self.name + "_gain")
        self.bias_layer = lgn.layers.BiasLayer(self.gain_layer, b=bias,
                                               name=self.name + "_bias")

        # neural output
        self.output = lgn.layers.NonlinearityLayer(
            self.bias_layer, nengo_lasagne.nl_map[type(ens.neuron_type)],
            name=self.name + "_nl")

    def make_recurrent(self, conn, model):
        if self.recurrent:
            raise ValueError("Only one recurrent connection allowed per "
                             "ensemble")
        self.recurrent = True

        rec = lgn.layers.InputLayer((None, self.ens.n_neurons))

        rec = ConnectionLayer(conn, rec, model).output

        if isinstance(conn.post_obj, nengo.Ensemble):
            rec = lgn.layers.DenseLayer(rec, self.ens.n_neurons,
                                        W=self.vector_input.enc_layer.W,
                                        name=self.name + "_rec_enc",
                                        nonlinearity=None, b=None)

        rec = GainLayer(
            rec, gain=self.gain_layer.gain, name=self.name + "_rec_gain",
            trainable=(self.gain_layer.gain in
                       self.gain_layer.get_params(trainable=True)))

        layer = self.output.input_layer

        layer = lgn.layers.ReshapeLayer(
            layer, (model.batch_size, -1, self.ens.n_neurons),
            name=self.name + "_reshape_in")

        # TODO: explore optimization parameters (e.g. rollout)
        layer = lgn.layers.CustomRecurrentLayer(
            layer, lgn.layers.InputLayer((None, self.ens.n_neurons)), rec,
            nonlinearity=nengo_lasagne.nl_map[type(self.ens.neuron_type)],
            name=self.name + "_nl")

        self.output = lgn.layers.ReshapeLayer(
            layer, (-1, self.ens.n_neurons), name=self.name + "_reshape_out")

    @property
    def vector_input(self):
        if self._vector_input is None:
            self._vector_input = ExpandableSumLayer(
                [], self.ens.dimensions, name=self.name + "_decoded_input")

            # add connection to neuron inputs
            self._vector_input.enc_layer = lgn.layers.DenseLayer(
                self._vector_input, self.input.num_units, W=self.encoders.T,
                nonlinearity=None, b=None, name=self.name + "_encoders")
            self.input.add_incoming(self._vector_input.enc_layer)

        return self._vector_input

    @property
    def encoders(self):
        if self._encoders is None:
            self._encoders = nengo_lasagne.to_array(self.ens.encoders,
                                                    (self.ens.n_neurons,
                                                     self.ens.dimensions))

        return self._encoders


class ConnectionLayer:
    def __init__(self, conn, layer, model):
        # TODO: support synapses?

        self.model = model

        # apply pre slice
        if conn.pre_slice != slice(None):
            layer = lgn.layers.SliceLayer(layer, conn.pre_sice)

        # apply node nonlinearity
        if isinstance(conn.pre_obj, nengo.Node) and conn.function is not None:
            # note: this won't work properly if conn.function doesn't
            # return a symbolic theano expression (e.g. if it uses numpy
            # functions or something)
            layer = lgn.layers.ExpressionLayer(
                layer, conn.function, output_shape=(None, conn.post.size_in))

        # set up connection weight layer
        if getattr(model.network.config[conn], "insert_weights", True):
            layer = lgn.layers.DenseLayer(
                layer, conn.post.size_in, W=self.get_weights(conn, layer),
                nonlinearity=None, b=None, name="%s_W" % conn)

        # apply post slice
        if conn.post_slice != slice(None):
            # zero-pad to get full shape (TODO: more efficient solution?)
            def pad_func(x):
                z = T.zeros((x.shape[0], conn.post_obj.size_in))
                return T.inc_subtensor(z[:, conn.post_slice], x)

            layer = lgn.layers.ExpressionLayer(
                layer, pad_func, output_shape=(None, conn.post_obj.size_in))

        self.output = layer

    def get_weights(self, conn, pre):
        if isinstance(conn.transform, nengo.dists.Distribution):
            if hasattr(conn.transform, "lgn_dist"):
                init_W = conn.transform.lgn_dist
            else:
                init_W = nengo_lasagne.dists.nengo_wrap(conn.transform)
        else:
            if isinstance(conn.pre_obj, nengo.Ensemble):
                # then we're dealing with decoders
                init_W = self.compute_decoders(conn)
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

        pre = self.model.params[conn.pre_obj]

        eval_points = nengo.builder.connection.get_eval_points(
            self.model, conn, rng=None)

        inputs = np.dot(eval_points, pre.encoders.T / conn.pre_obj.radius)
        targets = np.dot(np.apply_along_axis(func, 1,
                                             eval_points[:, conn.pre_slice]),
                         conn.transform.T)
        if targets.ndim == 1:
            targets = targets[:, None]

        post_enc = None
        if conn.solver.weights:
            post_enc = self.model.params[conn.post_obj].encoders.T
            post_enc = post_enc[conn.post_slice]
            post_enc /= conn.post_obj.radius

        decoders, _ = nengo.builder.connection.solve_for_decoders(
            conn.solver, conn.pre_obj.neuron_type, pre.gain_layer.gain.eval(),
            pre.bias_layer.b.eval(), inputs, targets, np.random, post_enc)

        return np.asarray(decoders, dtype=np.float32)

# class TimeLayer(lgn.layers.Layer):
#     def __init__(self, incoming, dt=0.001, name="time"):
#         assert isinstance(incoming, tuple)
#         super(TimeLayer, self).__init__(incoming, name)
#         self.dt = dt
#
#     def get_output_for(self, input):
#         return T.tile(T.arange(0, self.input_shape[1]) * self.dt,
#                       (self.input_shape[0], 1, 1))
