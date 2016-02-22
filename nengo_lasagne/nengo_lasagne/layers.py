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
        assert self.input_shapes[-1][1] == self.num_units

        self.input_layers += [None if isinstance(incoming, tuple)
                              else incoming]

        self.coeffs += [coeff]

    def get_output_shape_for(self, input_shapes):
        # check that all the batch sizes are the same
        if len(self.input_layers) == 0:
            return (None, self.num_units)

        assert len(np.unique([x[0] for x in input_shapes])) == 1

        return (input_shapes[0][0], self.num_units)

#     def get_output_for(self, inputs, **kwargs):
#         print self.name
#         print inputs
#
#         output = super(ExpandableSumLayer, self).get_output_for(inputs,
#                                                                 **kwargs)
#
#
#         assert output is not None
#
#         return output


class EnsembleLayer:
    def __init__(self, ens):
        self.ens = ens
        self.name = ens.label or "ensemble"
        self._decoded_input = None

        # neuron input
        self.input = ExpandableSumLayer(
            [], ens.n_neurons, name=self.name + "_input")

        # add gain/bias
        layer = self.input
        gain, bias, _, _ = nengo.builder.ensemble.get_gain_bias(ens)
        if gain is not None:
            layer = lgn.layers.NonlinearityLayer(layer, lambda x: x * gain,
                                                 name=self.name + "_gain")
        if bias is not None:
            layer = lgn.layers.BiasLayer(layer, name=self.name + "_bias")

        # neural output
        self.output = lgn.layers.NonlinearityLayer(
            layer, nengo_lasagne.nonlinearity_map[type(ens.neuron_type)],
            name=self.name + "_output")

    @property
    def decoded_input(self):
        if self._decoded_input is None:
            self._decoded_input = ExpandableSumLayer(
                [], self.ens.dimensions, name=self.name + "_decoded_input")

            # add connection to neuron inputs
            self.input.add_incoming(
                lgn.layers.DenseLayer(self._decoded_input,
                                      self.input.num_units,
                                      nonlinearity=None, b=None,
                                      name=self.name + "_encoders"))

        return self._decoded_input


# class MultiDenseLayer(lgn.layers.ElemwiseMergeLayer):
#     """A layer that receives inputs from multiple layers.
#
#     Basically DenseLayer and ElementwiseSumLayer combined together.
#     """
#
#     def __init__(self, incomings, num_units,
#                  nonlinearity=lgn.nonlinearities.rectify,
#                  Ws=lgn.init.GlorotUniform(),
#                  gain=None, b=None,
#                  **kwargs):
#         super(MultiDenseLayer, self).__init__(incomings, T.add, **kwargs)
#
#         self.num_units = num_units
#         self.nonlinearity = nonlinearity
#
#         if isinstance(Ws, list):
#             if len(Ws) != len(incomings):
#                 raise ValueError("Mismatch: got %d Ws for %d incomings" %
#                                  (len(Ws), len(incomings)))
#         else:
#             Ws = [Ws] * len(incomings)
#
#         self.Ws = [self.add_param(Ws[i], (incomings[i].output_shape[1],
#                                           num_units), name="W_%d" % i)
#                    for i in range(len(incomings))]
#
#         # TODO: do we want gains/biases to be trainable?
#         if gain is None:
#             self.gain = gain
#         else:
#             self.gain = self.add_param(gain, (num_units,), name="gain",
#                                        trainable=True)
#         if b is None:
#             self.b = b
#         else:
#             self.b = self.add_param(b, (num_units,), name="b",
#                                     trainable=True)
#
#     def get_output_shape_for(self, input_shapes):
#         # check that all the batch sizes are the same
#         assert len(np.unique([x[0] for x in input_shapes])) == 1
#
#         return (input_shapes[0][0], self.num_units)
#
#     def get_output_for(self, inputs, **kwargs):
#         inputs = [T.dot(input, W)
#                   for input, W in zip(inputs, self.Ws)]
#
#         output = super(MultiDenseLayer, self).get_output_for(inputs, **kwargs)
#         if self.gain is not None:
#             output *= self.gain
#         if self.b is not None:
#             output += self.b
#         output = self.nonlinearity(output)
#
#         return output
