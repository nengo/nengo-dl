import os
import sys

import flake8.main.cli as flake8
import nengo
import pytest

os.environ["NENGO_DL_TEST_PRECISION"] = "32"
os.environ["NENGO_DL_TEST_UNROLL"] = "1"
os.environ["NENGO_DL_TEST_DEVICE"] = "/gpu:0"

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_dl.tests.Simulator',
    '--ref-simulator', 'nengo_dl.tests.Simulator',
])

# run local tests
print("#" * 30, "NENGO_DL TESTS", "#" * 30)
pytest.main(['--gpu'])

# run flake8
sys.argv += "--ignore E721 .".split()
flake8.main()
