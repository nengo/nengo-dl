import tensorflow as tf

from nengo_deeplearning import Simulator


class Simulator32(Simulator):
    unsupported = Simulator.unsupported + [
        ("nengo/tests/test_synapses.py:test_alpha",
         "tolerances too small for 32 bit precision"),

        ("nengo/tests/test_synapses.py:test_linearfilter",
         "tolerances too small for 32 bit precision"),

        ("nengo/tests/test_synapses.py:test_triangle",
         "tolerances too small for 32 bit precision"),
    ]

    def __init__(self, *args, **kwargs):
        super(Simulator32, self).__init__(*args, dtype=tf.float32, **kwargs)


class Simulator64(Simulator):
    def __init__(self, *args, **kwargs):
        super(Simulator64, self).__init__(*args, dtype=tf.float64, **kwargs)
