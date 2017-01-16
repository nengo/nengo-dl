import tensorflow as tf

from nengo_deeplearning import Simulator


# TODO: set this up so step_blocks, unroll_simulation can be controlled by
# environment variables in travis ci

class Simulator32(Simulator):
    unsupported = Simulator.unsupported + [
        ("nengo/tests/test_synapses.py:test_alpha",
         "tolerances too small for 32 bit precision"),

        ("nengo/tests/test_synapses.py:test_triangle",
         "tolerances too small for 32 bit precision"),
    ]

    def __init__(self, *args, **kwargs):
        super(Simulator32, self).__init__(
            *args, dtype=tf.float32, unroll_simulation=False, step_blocks=None,
            **kwargs)


class Simulator64(Simulator):
    def __init__(self, *args, **kwargs):
        super(Simulator64, self).__init__(
            *args, dtype=tf.float64, unroll_simulation=False, step_blocks=None,
            **kwargs)
