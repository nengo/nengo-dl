import os
import sys
import tempfile

import nengo
from pylint import epylint
import pytest

os.environ["NENGO_DL_TEST_PRECISION"] = "32"
os.environ["NENGO_DL_TEST_UNROLL"] = "1"
os.environ["NENGO_DL_TEST_DEVICE"] = "/gpu:0"
os.environ["NENGO_DL_TEST_INFERENCE_ONLY"] = "False"

# run pylint
print("#" * 30, "PYLINT", "#" * 30)
epylint.py_run("../nengo_dl --rcfile=setup.cfg")

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    "--simulator", "nengo_dl.tests.Simulator",
    "--ref-simulator", "nengo_dl.tests.Simulator",
    "--disable-warnings"
])

# run local tests
print("#" * 30, "NENGO_DL TESTS", "#" * 30)
pytest.main(["--gpu"])

# test whitepaper plots
print("=" * 30, "WHITEPAPER PLOTS", "#" * 30)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "docs",
                             "whitepaper"))
import whitepaper2018_plots  # pylint: disable=wrong-import-position

with tempfile.TemporaryDirectory() as tmpdir:
    # run in temporary directory so that we don't end up with a bunch of files
    # in this dir
    os.chdir(tmpdir)

    sys.argv += "--no-show --reps 1 test".split()
    try:
        whitepaper2018_plots.main(obj={})
    except SystemExit:
        pass
