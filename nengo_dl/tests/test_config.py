from nengo import ensemble, Network, Ensemble, Connection
from nengo.builder import Model
from nengo.exceptions import ValidationError, ConfigError, NetworkContextError
import pytest

from nengo_dl import config, builder


def test_configure_trainable():
    with Network() as net:
        conf = net.config
        config.configure_settings(trainable=None)

    assert conf[Ensemble].trainable is None
    assert conf[Connection].trainable is None
    assert conf[ensemble.Neurons].trainable is None

    # check that we can set trainable after it is set up for configuration
    conf[Ensemble].trainable = True

    # check that boolean value is enforced
    with pytest.raises(ValidationError):
        conf[Ensemble].trainable = 5

    assert conf[Ensemble].trainable is True

    # check that calling configure again overrides previous changes
    with net:
        config.configure_settings(trainable=None)

    assert conf[Ensemble].trainable is None

    # check that non-None defaults work
    with net:
        config.configure_settings(trainable=False)

    assert conf[Ensemble].trainable is False

    # check that calling configure outside network context is an error
    with pytest.raises(NetworkContextError):
        config.configure_settings(trainable=None)

    # check that passing an invalid parameter raises an error
    with net:
        with pytest.raises(ConfigError):
            config.configure_settings(troinable=None)


@pytest.mark.parametrize("as_model", (True, False))
def test_session_config(Simulator, as_model):
    with Network() as net:
        config.configure_settings(session_config={
            "graph_options.optimizer_options.opt_level": 21,
            "gpu_options.allow_growth": True})

    if as_model:
        # checking that config settings work when we pass in a model instead of
        # network
        model = Model(dt=0.001, builder=builder.NengoBuilder())
        model.build(net)
        net = None
    else:
        model = None

    with Simulator(net, model=model) as sim:
        assert sim.sess._config.graph_options.optimizer_options.opt_level == 21
        assert sim.sess._config.gpu_options.allow_growth
