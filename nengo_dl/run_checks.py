import os
import sys
import tempfile

from pylint import epylint
import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["NENGO_DL_TEST_PRECISION"] = "32"
os.environ["NENGO_DL_TEST_UNROLL"] = "1"
os.environ["NENGO_DL_TEST_DEVICE"] = "/gpu:0"
os.environ["NENGO_DL_TEST_INFERENCE_ONLY"] = "False"

# run pylint
print("#" * 30, "PYLINT", "#" * 30)
epylint.py_run("../nengo_dl --rcfile=setup.cfg")

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main(["--pyargs", "nengo"])

# run local tests
print("#" * 30, "NENGO_DL TESTS", "#" * 30)
pytest.main(["--gpu"])

# run performance benchmarks
print("#" * 30, "PERFORMANCE BENCHMARKS", "#" * 30)
from nengo_dl import benchmarks  # pylint: disable=wrong-import-position
sys.argv = ("benchmarks.py performance_samples --device %s" %
            os.environ["NENGO_DL_TEST_DEVICE"]).split()
try:
    benchmarks.main(obj={})
except SystemExit:
    pass

# test whitepaper plots
print("=" * 30, "WHITEPAPER PLOTS", "#" * 30)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "docs",
                             "whitepaper"))
import whitepaper2018_plots  # pylint: disable=wrong-import-position

with tempfile.TemporaryDirectory() as tmpdir:
    # run in temporary directory so that we don't end up with a bunch of files
    # in this dir
    os.chdir(tmpdir)

    sys.argv = "whitepaper2018_plots.py --no-show --reps 1 test".split()
    try:
        whitepaper2018_plots.main(obj={})
    except SystemExit:
        pass
