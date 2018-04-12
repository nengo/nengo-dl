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
    '--disable-warnings'
])

# run local tests
print("#" * 30, "NENGO_DL TESTS", "#" * 30)
pytest.main(['--gpu'])

# test whitepaper plots
sys.path.append("../docs/whitepaper")
import whitepaper2018_plots

sys.argv += "--no-show --reps 1 test".split()
whitepaper2018_plots.main(obj={})

# run flake8
sys.argv = sys.argv[:1]
sys.argv += "--ignore E721,E402 .".split()
flake8.main()
