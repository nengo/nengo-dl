import os
import sys

import nengo
from pylint import epylint
import pytest

os.environ["NENGO_DL_TEST_PRECISION"] = "32"
os.environ["NENGO_DL_TEST_UNROLL"] = "1"
os.environ["NENGO_DL_TEST_DEVICE"] = "/gpu:0"

# run pylint
print("#" * 30, "PYLINT", "#" * 30)
epylint.py_run("../nengo_dl --rcfile=setup.cfg")

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
print("=" * 30, "WHITEPAPER PLOTS", "#" * 30)
os.chdir("../tmp")  # so that we don't end up with a bunch of files in this dir
sys.path.append("../docs/whitepaper")
import whitepaper2018_plots  # pylint: disable=wrong-import-position

sys.argv += "--no-show --reps 1 test".split()
whitepaper2018_plots.main(obj={})
