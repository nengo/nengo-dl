from distutils.version import LooseVersion
import shlex

import nengo
from nengo.tests import test_synapses, test_learning_rules
import numpy as np
import pytest
import tensorflow as tf

from nengo_dl import simulator, utils
from nengo_dl.compat import tf_compat
from nengo_dl.tests import make_test_sim


def pytest_configure(config):
    # add unsupported attribute to Simulator (for compatibility with nengo<3.0)
    # join all the lines and then split (preserving quoted strings)
    if nengo.version.version_info <= (2, 8, 0):
        unsupported = shlex.split(" ".join(config.getini("nengo_test_unsupported")))
        # group pairs (representing testname + reason)
        unsupported = [unsupported[i : i + 2] for i in range(0, len(unsupported), 2)]
        # wrap square brackets to interpret them literally
        # (see https://docs.python.org/3/library/fnmatch.html)
        for i, (testname, _) in enumerate(unsupported):
            unsupported[i][0] = "".join(
                "[%s]" % c if c in ("[", "]") else c for c in testname
            ).replace("::", ":")

        simulator.Simulator.unsupported = unsupported


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu", False) and not utils.tf_gpu_installed:
        pytest.skip("This test requires tensorflow-gpu")
    elif (
        hasattr(item, "fixturenames")
        and "Simulator" not in item.fixturenames
        and item.config.getvalue("--simulator-only")
    ):
        pytest.skip("Only running tests that require a Simulator")
    elif item.get_closest_marker("training", False) and item.config.getvalue(
        "--inference-only"
    ):
        pytest.skip("Skipping training test in inference-only mode")
    elif item.get_closest_marker("performance", False) and not item.config.getvalue(
        "--performance"
    ):
        pytest.skip("Skipping performance test")

    tf.keras.backend.clear_session()


def pytest_addoption(parser):
    parser.addoption(
        "--simulator-only",
        action="store_true",
        default=False,
        help="Only run tests involving Simulator",
    )
    parser.addoption(
        "--inference-only",
        action="store_true",
        default=False,
        help="Run tests in inference-only mode",
    )
    parser.addoption(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Simulator float precision",
    )
    parser.addoption(
        "--unroll-simulation",
        default=1,
        type=int,
        help="`unroll_simulation` parameter for Simulator",
    )
    parser.addoption("--device", default=None, help="`device` parameter for Simulator")
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )

    if nengo.version.version_info <= (2, 8, 0):
        # add the pytest options from future nengo versions
        parser.addini(
            "nengo_test_unsupported",
            type="linelist",
            help="List of unsupported unit tests with reason for exclusion",
        )


@pytest.fixture(scope="session")
def Simulator(request):
    """
    Simulator class to be used in tests (use this instead of ``nengo_dl.Simulator``).
    """

    return make_test_sim(request)


def patch_nengo_tests():
    """
    Monkey-patch various aspects of the Nengo test suite, so that things
    work correctly when running those tests through NengoDL.
    """

    if LooseVersion(nengo.__version__) < "3.0.0":
        from nengo import conftest

        # monkey patch the nengo Simulator fixture, so that we can also use the pytest
        # arguments to control nengo tests
        conftest.Simulator = Simulator
        conftest.RefSimulator = Simulator

        # set looser tolerances on synapse tests (since allclose fixture doesn't work
        # in these versions)
        def allclose(*args, **kwargs):
            kwargs.setdefault("atol", 5e-5)
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
        def _test_pes(
            NengoSimulator,
            nl,
            plt,
            seed,
            pre_neurons=False,
            post_neurons=False,
            weight_solver=False,
            vin=np.array([0.5, -0.5]),
            vout=None,
            n=200,
            function=None,
            transform=np.array(1.0),
            rate=1e-3,
        ):
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

                bslice = b[: v.size_out] if v.size_out < u.size_out else b
                pre = a.neurons if pre_neurons else a
                post = b.neurons if post_neurons else bslice

                conn = nengo.Connection(
                    pre,
                    post,
                    function=function,
                    transform=transform,
                    learning_rule_type=nengo.PES(rate),
                )
                if weight_solver:
                    conn.solver = nengo.solvers.LstsqL2(weights=True)

                nengo.Connection(v, e, transform=-1)
                nengo.Connection(bslice, e)
                nengo.Connection(e, conn.learning_rule)

                b_p = nengo.Probe(bslice, synapse=0.03)
                e_p = nengo.Probe(e, synapse=0.03)

                weights_p = nengo.Probe(conn, "weights", sample_every=0.01)

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
