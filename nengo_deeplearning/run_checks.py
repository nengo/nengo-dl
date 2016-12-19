import os
import sys

import flake8.main
import nengo
import pytest

# run nengo tests
# print("#" * 30, "NENGO TESTS 32 BIT", "#" * 30)
# pytest.main([
#     os.path.dirname(nengo.__file__),
#     '--simulator', 'nengo_deeplearning.tests.Simulator32',
#     '--ref-simulator', 'nengo_deeplearning.tests.Simulator32',
#     '-p', 'nengo.tests.options',
# ])
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
