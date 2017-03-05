import os
import sys

import flake8.main
import nengo
import pytest

os.environ["NENGO_DL_TEST_PRECISION"] = "32"
os.environ["NENGO_DL_TEST_UNROLL"] = "True"
os.environ["NENGO_DL_TEST_STEP_BLOCKS"] = "10"
os.environ["NENGO_DL_TEST_DEVICE"] = "/cpu:0"

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_deeplearning.tests.Simulator',
    '--ref-simulator', 'nengo_deeplearning.tests.Simulator',
    '-p', 'nengo.tests.options',
])

# run local tests
print("#" * 30, "NENGO_DEEPLEARNING TESTS", "#" * 30)
pytest.main()

# run flake8
sys.argv += "--ignore E721 .".split()
flake8.main.main()
