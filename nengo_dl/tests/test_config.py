# pylint: disable=missing-docstring

import nengo
from nengo.exceptions import ValidationError, ConfigError, NetworkContextError
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import config


def test_configure_trainable():
    with nengo.Network() as net:
        conf = net.config
        config.configure_settings(trainable=None)

    assert conf[nengo.Ensemble].trainable is None
    assert conf[nengo.Connection].trainable is None
    assert conf[nengo.ensemble.Neurons].trainable is None

    # check that we can set trainable after it is set up for configuration
    conf[nengo.Ensemble].trainable = True

    # check that boolean value is enforced
    with pytest.raises(ValidationError):
        conf[nengo.Ensemble].trainable = 5

    assert conf[nengo.Ensemble].trainable is True

    # check that calling configure again overrides previous changes
    with net:
        config.configure_settings(trainable=None)

    assert conf[nengo.Ensemble].trainable is None

    # check that non-None defaults work
    with net:
        config.configure_settings(trainable=False)

    assert conf[nengo.Ensemble].trainable is False

    # check that calling configure outside network context is an error
    with pytest.raises(NetworkContextError):
        config.configure_settings(trainable=None)

    # check that passing an invalid parameter raises an error
    with net:
        with pytest.raises(ConfigError):
            config.configure_settings(troinable=None)


@pytest.mark.parametrize("use_loop", (True, False))
def test_keep_history(Simulator, use_loop, seed):
    with nengo.Network(seed=seed) as net:
        config.configure_settings(keep_history=True, use_loop=use_loop)
        a = nengo.Ensemble(30, 1)
        p = nengo.Probe(a.neurons, synapse=0.1)

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
    with nengo.Network() as net:
        config.configure_settings(stateful=sim_stateful)

        nengo.Ensemble(30, 1)

    with Simulator(net) as sim:
        kwargs = dict(n_steps=5, stateful=func_stateful)

        with pytest.warns(None) as recwarns:
            getattr(sim, func)(**kwargs)
        assert sim.n_steps == (5 if func_stateful and sim_stateful else 0)

        if func == "predict" and func_stateful and not sim_stateful:
            # note: we do not get warnings for predict_on_batch/run_steps because
            # they automatically set func_stateful=sim_stateful
            assert (
                len([w for w in recwarns if "Ignoring stateful=True" in str(w.message)])
                > 0
            )
        else:
            assert len(recwarns) == 0

        getattr(sim, func)(**kwargs)
        assert sim.n_steps == (10 if func_stateful and sim_stateful else 0)


def test_distribute_strategy(Simulator):
    gpus = tf.config.experimental.list_physical_devices("GPU")

    # Create 2 virtual GPUs with 1GB memory each
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
        ],
    )
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

    with nengo.Network() as net:
        config.configure_settings(distribute_strategy=tf.distribute.MirroredStrategy())
        inputs = nengo.Node([0])
        predictions = nengo.Node(size_in=1)
        nengo.Connection(inputs, predictions)
        nengo.Probe(predictions)

    with Simulator(net, minibatch_size=4) as sim:
        sim.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
        sim.fit(np.zeros((1000, 1, 1)), np.zeros((1000, 1, 1)), epochs=100)
