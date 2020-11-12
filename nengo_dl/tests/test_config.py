# pylint: disable=missing-docstring

import numpy as np
import pytest
from nengo import Connection, Ensemble, Network, Probe, ensemble
from nengo.exceptions import ConfigError, NetworkContextError, ValidationError

from nengo_dl import config


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


@pytest.mark.parametrize("use_loop", (True, False))
def test_keep_history(Simulator, use_loop, seed):
    with Network(seed=seed) as net:
        config.configure_settings(keep_history=True, use_loop=use_loop)
        a = Ensemble(30, 1)
        p = Probe(a.neurons, synapse=0.1)

    kwargs = dict() if use_loop else dict(unroll_simulation=10)
    with Simulator(net, **kwargs) as sim:
        sim.run_steps(10)

    with net:
        net.config[p].keep_history = False

    with Simulator(net, **kwargs) as sim2:
        sim2.run_steps(10)

    assert sim.data[p].shape == (10, 30)
    assert sim2.data[p].shape == (1, 30)
    assert np.allclose(sim.data[p][[-1]], sim2.data[p])


@pytest.mark.parametrize(
    "sim_stateful, func_stateful", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("func", ("predict_on_batch", "predict", "run_steps"))
def test_stateful(Simulator, sim_stateful, func_stateful, func):
    with Network() as net:
        config.configure_settings(stateful=sim_stateful)

        Ensemble(30, 1)

    with Simulator(net) as sim:
        kwargs = dict(n_steps=5, stateful=func_stateful)

        with pytest.warns(None) as recwarns:
            getattr(sim, func)(**kwargs)
        assert sim.n_steps == (5 if func_stateful and sim_stateful else 0)

        if func == "predict" and func_stateful and not sim_stateful:
            # note: we do not get warnings for predict_on_batch/run_steps because
            # they automatically set func_stateful=sim_stateful
            assert any("Ignoring stateful=True" in str(w.message) for w in recwarns)
        else:
            assert not any("Ignoring stateful=True" in str(w.message) for w in recwarns)

        getattr(sim, func)(**kwargs)
        assert sim.n_steps == (10 if func_stateful and sim_stateful else 0)
