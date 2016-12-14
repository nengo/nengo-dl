import os

import nengo
import pytest

# run nengo tests
print("#" * 30, "NENGO TESTS", "#" * 30)
pytest.main([
    os.path.dirname(nengo.__file__),
    '--simulator', 'nengo_deeplearning.Simulator',
    '--ref-simulator', 'nengo_deeplearning.Simulator',
    '-p', 'nengo.tests.options',
    '-x',
    # '-k test_learning_rules.py'
])

# run local tests
print("#" * 30, "NENGO_DEEPLEARNING TESTS", "#" * 30)
pytest.main()

# run flake8
