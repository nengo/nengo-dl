import os
import sys

import flake8.main
import nengo
import nengo.tests.test_synapses
import pytest


# set looser tolerances on synapse tests (due to 32 bit precision); note this
# is also necessary for the 64bit tests, because although we compute internally
# in 64 bit precision the probe output is still 32bit.
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 1e-6)
    return nengo.utils.testing.allclose(*args, **kwargs)
nengo.tests.test_synapses.allclose = allclose_tol

# run nengo tests
print("#" * 30, "NENGO TESTS 32 BIT", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_deeplearning.tests.Simulator32',
    '--ref-simulator', 'nengo_deeplearning.tests.Simulator32',
    '-p', 'nengo.tests.options',
])
print("#" * 30, "NENGO TESTS 64 BIT", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_deeplearning.tests.Simulator64',
    '--ref-simulator', 'nengo_deeplearning.tests.Simulator64',
    '-p', 'nengo.tests.options',
])

# run local tests
print("#" * 30, "NENGO_DEEPLEARNING TESTS", "#" * 30)
pytest.main()

# run flake8
sys.argv += ["."]
flake8.main.main()
