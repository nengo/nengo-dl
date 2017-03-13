from nengo.conftest import seed  # noqa
import nengo.tests.test_synapses
import numpy as np


# set looser tolerances on synapse tests
def allclose(*args, **kwargs):
    kwargs.setdefault('atol', 5e-7)
    return nengo.utils.testing.allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose

# cast output of run_synapse to float64. this is necessary because
# Synapse.filt bases its internal dtypes on the dtype of its inputs, and
# we don't want to downcast everything there to float32.
nengo_run_synapse = nengo.tests.test_synapses.run_synapse


def run_synapse(*args, **kwargs):
    output = nengo_run_synapse(*args, **kwargs)
    return tuple(x.astype(np.float64) for x in output)


nengo.tests.test_synapses.run_synapse = run_synapse
