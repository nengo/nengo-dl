import pkg_resources

import nengo.conftest
from nengo.conftest import seed, rng  # pylint: disable=unused-import
from nengo.tests import test_synapses, test_learning_rules
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import config, simulator


def pytest_runtest_setup(item):
    if getattr(item.obj, "gpu", False) and not pytest.gpu_installed:
        pytest.skip("This test requires tensorflow-gpu")
    elif (hasattr(item, "fixturenames") and
          "Simulator" not in item.fixturenames and
          item.config.getvalue("--simulator-only")):
        pytest.skip("Only running tests that require a Simulator")
    elif getattr(item.obj, "training", False) and item.config.getvalue(
            "--inference-only"):
        pytest.skip("Skipping training test in inference-only mode")


def pytest_addoption(parser):
    parser.addoption("--simulator-only", action="store_true", default=False,
                     help="Only run tests involving Simulator")
    parser.addoption("--inference-only", action="store_true", default=False,
                     help="Run tests in inference-only mode")
    parser.addoption("--dtype", default="float32",
                     choices=("float32", "float64"),
                     help="Simulator float precision")
    parser.addoption("--unroll_simulation", default=1, type=int,
                     help="unroll_simulation value for Simulator")
    parser.addoption("--device", default=None,
                     help="device parameter for Simulator")


def pytest_namespace():
    gpu_dists = [d for d in pkg_resources.working_set
                 if d.project_name in ("tensorflow-gpu", "tf-nightly-gpu")]
    return {"gpu_installed": len(gpu_dists) > 0}


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests (use this instead of
    ``nengo_dl.Simulator``).
    """

    dtype = getattr(tf, request.config.getoption("--dtype"))
    unroll = request.config.getoption("--unroll_simulation")
    device = request.config.getoption("--device")
    inference_only = request.config.getoption("--inference-only")

    def TestSimulator(net, *args, **kwargs):
        # raise Exception

        kwargs.setdefault("unroll_simulation", unroll)
        kwargs.setdefault("device", device)

        if net is not None and config.get_setting(
                net, "inference_only") is None:
            with net:
                config.configure_settings(inference_only=inference_only)

        if net is not None and config.get_setting(net, "dtype") is None:
            with net:
                config.configure_settings(dtype=dtype)

        return simulator.Simulator(net, *args, **kwargs)

    return TestSimulator


@pytest.fixture(scope="function")
def sess(request):
    """Create a TensorFlow session with a unique scope per test."""
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            yield sess


def patch_nengo_tests():
    """Monkey-patch various aspects of the Nengo test suite, so that things
    work correctly when running those tests through NengoDL."""

    # replace nengo Simulator fixture
    nengo.conftest.Simulator = Simulator

    # set looser tolerances on synapse tests
    def allclose(*args, **kwargs):
        kwargs.setdefault('atol', 5e-7)
        return nengo.utils.testing.allclose(*args, **kwargs)

    test_synapses.allclose = allclose

    # cast output of run_synapse to float64. this is necessary because
    # Synapse.filt bases its internal dtypes on the dtype of its inputs, and
    # we don't want to downcast everything there to float32.
    nengo_run_synapse = test_synapses.run_synapse

    def run_synapse(*args, **kwargs):
        output = nengo_run_synapse(*args, **kwargs)
        return tuple(x.astype(np.float64) for x in output)

    test_synapses.run_synapse = run_synapse

    # remove the correction probes from _test_pes
    def _test_pes(NengoSimulator, nl, plt, seed,
                  pre_neurons=False, post_neurons=False, weight_solver=False,
                  vin=np.array([0.5, -0.5]), vout=None, n=200,
                  function=None, transform=np.array(1.), rate=1e-3):
        vout = np.array(vin) if vout is None else vout

        model = nengo.Network(seed=seed)
        with model:
            model.config[nengo.Ensemble].neuron_type = nl()

            u = nengo.Node(output=vin)
            v = nengo.Node(output=vout)
            a = nengo.Ensemble(n, dimensions=u.size_out)
            b = nengo.Ensemble(n, dimensions=u.size_out)
            e = nengo.Ensemble(n, dimensions=v.size_out)

            nengo.Connection(u, a)

            bslice = b[:v.size_out] if v.size_out < u.size_out else b
            pre = a.neurons if pre_neurons else a
            post = b.neurons if post_neurons else bslice

            conn = nengo.Connection(pre, post,
                                    function=function, transform=transform,
                                    learning_rule_type=nengo.PES(rate))
            if weight_solver:
                conn.solver = nengo.solvers.LstsqL2(weights=True)

            nengo.Connection(v, e, transform=-1)
            nengo.Connection(bslice, e)
            nengo.Connection(e, conn.learning_rule)

            b_p = nengo.Probe(bslice, synapse=0.03)
            e_p = nengo.Probe(e, synapse=0.03)

            weights_p = nengo.Probe(conn, 'weights', sample_every=0.01)

        with NengoSimulator(model) as sim:
            sim.run(0.5)
        t = sim.trange()
        weights = sim.data[weights_p]

        tend = t > 0.4
        assert np.allclose(sim.data[b_p][tend], vout, atol=0.05)
        assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)
        assert not np.allclose(weights[0], weights[-1], atol=1e-5)

    test_learning_rules._test_pes = _test_pes


patch_nengo_tests()
