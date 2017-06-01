import functools
import os

import pytest
from nengo.utils.stdlib import execfile

examples_dir = os.path.join(os.path.dirname(__file__), "..")

skip_examples = ["integrator.py"]


@pytest.mark.parametrize(
    "example_file",
    [os.path.join(examples_dir, f) for f in os.listdir(examples_dir)
     if f.endswith(".py") and f not in skip_examples])
def test_no_exceptions(example_file):
    # monkeypatch plt.show to be non-blocking
    import matplotlib.pyplot as plt
    old_show = plt.show
    plt.show = functools.partial(plt.show, block=False)

    execfile(example_file, {})

    plt.close("all")
    plt.show = old_show
