import nengo
import lasagne as lgn

from nengo.params import Parameter

from .simulator import Simulator

nonlinearity_map = {nengo.RectifiedLinear: lgn.nonlinearities.rectify,
                    nengo.Sigmoid: lgn.nonlinearities.sigmoid,
                    nengo.Direct: lgn.nonlinearities.linear}


def default_config():
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    config[nengo.Ensemble].bias = nengo.dists.Choice([0])

    return config


def training_settings(net, train_inputs, train_targets, batch_size=None,
                      n_epochs=1000, l_rate=0.1):
    net.config.configures(Simulator)
    net.config[Simulator].set_param("train_inputs", Parameter(train_inputs))
    net.config[Simulator].set_param("train_targets", Parameter(train_targets))

    if batch_size is None:
        batch_size = len(list(train_inputs.values())[0])
    net.config[Simulator].set_param("batch_size", Parameter(batch_size))

    net.config[Simulator].set_param("n_epochs", Parameter(n_epochs))
    net.config[Simulator].set_param("l_rate", Parameter(l_rate))