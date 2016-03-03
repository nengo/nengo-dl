import lasagne as lgn
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


class NodeLayer:
    def __init__(self, node):
        self.num_units = node.size_out

        if node.size_in == 0:
            self.input = lgn.layers.InputLayer((None, None, self.num_units),
                                               name=node.label)

            # flatten the batch/sequence dimensions
            self.output = lgn.layers.ReshapeLayer(
                self.input, (-1, self.num_units), name=node.label + "_reshape")
        else:
            self.input = self.output = ExpandableSumLayer([], self.num_units,
                                                          name=node.label)


class EnsembleLayer:
    def __init__(self, ens, recurrent=None, batch_size=None):
        self.ens = ens
        self.name = ens.label or "ensemble"
        self._decoded_input = None
        self._encoders = None
        self.eval_points = nengo.builder.ensemble.gen_eval_points(
            ens, ens.eval_points, np.random)

        # neuron input
        self.input = ExpandableSumLayer(
            [], ens.n_neurons, name=self.name + "_input")

        # add gain/bias
        self.gain, self.bias, _, _ = nengo.builder.ensemble.get_gain_bias(ens)
        self.gain = np.asarray(self.gain, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)
        layer = lgn.layers.NonlinearityLayer(self.input,
                                             lambda x: x * self.gain,
                                             name=self.name + "_gain")
        layer = lgn.layers.BiasLayer(layer, name=self.name + "_bias",
                                     b=self.bias)

        # neural output
        if recurrent is None:
            self.output = lgn.layers.NonlinearityLayer(
                layer, nengo_lasagne.nl_map[type(ens.neuron_type)],
                name=self.name + "_nl")
        else:
            # TODO: support slicing on recurrent connection
            # TODO: NEF initialization
            rec = lgn.layers.InputLayer((None, ens.n_neurons))

            if isinstance(recurrent.post_obj, nengo.ensemble.Neurons):
                rec = lgn.layers.DenseLayer(rec, ens.n_neurons,
                                            name=self.name + "_rec",
                                            nonlinearity=None, b=None)
            else:
                rec = lgn.layers.DenseLayer(rec, ens.dimensions,
                                            name=self.name + "_rec",
                                            nonlinearity=None, b=None)
                rec = lgn.layers.DenseLayer(rec, ens.n_neurons,
                                            W=self.decoded_input.enc_layer.W,
                                            name=self.name + "_rec_enc",
                                            nonlinearity=None, b=None)
            rec = lgn.layers.NonlinearityLayer(rec, lambda x: x * self.gain,
                                               name=self.name + "_rec_gain")

            layer = lgn.layers.ReshapeLayer(
                layer, (batch_size, -1, ens.n_neurons),
                name=self.name + "_reshape_in")

            # TODO: explore optimization parameters (e.g. rollout)
            layer = lgn.layers.CustomRecurrentLayer(
                layer, lgn.layers.InputLayer((None, ens.n_neurons)), rec,
                nonlinearity=nengo_lasagne.nl_map[type(ens.neuron_type)],
                name=self.name + "_nl")
            # layer = lgn.layers.RecurrentLayer(
            #     layer, ens.n_neurons, name=self.name + "_nl",
            #     nonlinearity=nengo_lasagne.nl_map[type(ens.neuron_type)])

            self.output = lgn.layers.ReshapeLayer(
                layer, (-1, ens.n_neurons), name=self.name + "_reshape_out")

    @property
    def decoded_input(self):
        if self._decoded_input is None:
            self._decoded_input = ExpandableSumLayer(
                [], self.ens.dimensions, name=self.name + "_decoded_input")

            # add connection to neuron inputs
            self._decoded_input.enc_layer = lgn.layers.DenseLayer(
                self._decoded_input, self.input.num_units, W=self.encoders.T,
                nonlinearity=None, b=None, name=self.name + "_encoders")
            self.input.add_incoming(self._decoded_input.enc_layer)

        return self._decoded_input

    @property
    def encoders(self):
        if self._encoders is None:
            self._encoders = nengo_lasagne.to_array(self.ens.encoders,
                                                    (self.ens.n_neurons,
                                                     self.ens.dimensions))

        return self._encoders
