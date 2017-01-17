import os
import sys

import flake8.main
import nengo.tests.test_synapses
import pytest

from nengo_deeplearning.tests import conftest

os.environ["NENGO_DL_TEST_PRECISION"] = "64"
os.environ["NENGO_DL_TEST_UNROLL"] = "False"
os.environ["NENGO_DL_TEST_STEP_BLOCKS"] = "None"

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_deeplearning.tests.TestSimulator',
    '--ref-simulator', 'nengo_deeplearning.tests.TestSimulator',
    '-p', 'nengo.tests.options',
    '-x',
])

# run local tests
# print("#" * 30, "NENGO_DEEPLEARNING TESTS", "#" * 30)
# pytest.main()

# run flake8
sys.argv += ["."]
flake8.main.main()
