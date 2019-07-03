# pylint: disable=unused-import,ungrouped-imports

"""
Utilities to ease cross-compatibility between different versions of upstream
dependencies.
"""

from distutils.version import LooseVersion

import nengo
import numpy as np
import tensorflow as tf


tf_compat = tf.compat.v1


def RefVariable(*args, **kwargs):
    """Always returns RefVariables instead of (maybe) ResourceVariables."""

    return tf.compat.v1.Variable(*args, use_resource=False, **kwargs)


if LooseVersion(nengo.__version__) < "3.0.0":
    from nengo.utils.testing import allclose as signals_allclose

    class SimPES(nengo.builder.Operator):  # pylint: disable=abstract-method
        """Future `nengo.builder.operator.SimPES` class."""

        def __init__(
            self, pre_filtered, error, delta, learning_rate, encoders=None, tag=None
        ):
            super(SimPES, self).__init__(tag=tag)

            self.pre_filtered = pre_filtered
            self.error = error
            self.delta = delta
            self.learning_rate = learning_rate
            self.encoders = encoders

            # encoders not used in NengoDL (they'll be applied outside the op)
            assert encoders is None

            # note: in 3.0.0 the PES op changed from a set to an update
            self.sets = [delta]
            self.incs = []
            self.reads = [pre_filtered, error]
            self.updates = []

        def _descstr(self):
            return "pre=%s, error=%s -> %s" % (
                self.pre_filtered,
                self.error,
                self.delta,
            )

    # remove 'correction' from probeable attributes
    nengo.PES.probeable = ("error", "activities", "delta")

    class Convolution:
        """Dummy `nengo.transforms.Convolution` class."""

    class ConvInc:
        """Dummy `nengo.builder.transforms.ConvInc` class."""

    class SparseMatrix:
        """Dummy `nengo.transforms.SparseMatrix` class."""

    class SparseDotInc:
        """Dummy `nengo.builder.transforms.SparseDotInc` class."""

    def is_sparse(sig):
        """Check if Signal is sparse"""
        # always False since Sparse signals didn't exist, but we use getattr
        # so that dummies.Signal(sparse=False) will still work
        return getattr(sig, "sparse", False)

    class Step:
        """Future `nengo.synapses.LinearFilter.Step` class."""

        def __init__(self, A, B, C, D):
            self.A = A
            self.B = B
            self.C = C
            self.D = D

        @staticmethod
        def check(A, B, C, D):
            """Check if the given matrices represent a system of this type."""

    class NoX(Step):
        """Future `nengo.synapses.LinearFilter.NoX` class."""

        @staticmethod
        def check(A, B, C, D):
            return A.size == 0

    class OneX(Step):
        """Future `nengo.synapses.LinearFilter.OneX` class."""

        @staticmethod
        def check(A, B, C, D):
            return len(A) == 1 and (D == 0).all()

    class NoD(Step):
        """Future `nengo.synapses.LinearFilter.NoD` class."""

        @staticmethod
        def check(A, B, C, D):
            return len(A) >= 1 and (D == 0).all()

    class General(Step):
        """Future `nengo.synapses.LinearFilter.General` class."""

        @staticmethod
        def check(A, B, C, D):
            return len(A) >= 1

    def make_linear_step(synapse, input_shape, output_shape, dt):
        """Generate one of the future LinearFilter step types."""

        from nengo.utils.filter_design import cont2discrete, tf2ss

        A, B, C, D = tf2ss(synapse.num, synapse.den)

        # discretize (if len(A) == 0, filter is stateless and already discrete)
        if synapse.analog and len(A) > 0:
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt, method="zoh")

        if NoX.check(A, B, C, D):
            return NoX(A, B, C, D)
        elif OneX.check(A, B, C, D):
            return OneX(A, B, C, D)
        elif NoD.check(A, B, C, D):
            return NoD(A, B, C, D)
        else:
            assert General.check(A, B, C, D)
            return General(A, B, C, D)

    # monkey-patch in empty state attribute
    from nengo.builder.processes import SimProcess

    SimProcess.state = {}

    def make_process_step(process, shape_in, shape_out, dt, rng, _):
        """Ignore state argument."""
        return process.make_step(shape_in, shape_out, dt, rng)

    def make_process_state(process, shape_in, shape_out, dt):
        """Old processes don't have any state."""
        return {}

    # set seed upper bound to uint32 (instead of int32)
    nengo.base.Process.seed.high = np.iinfo(np.uint32).max

else:
    from nengo.builder.learning_rules import SimPES
    from nengo.builder.transforms import ConvInc, SparseDotInc
    from nengo.transforms import Convolution, SparseMatrix
    from nengo.synapses import LinearFilter
    from nengo.utils.testing import signals_allclose

    def is_sparse(sig):
        """Check if Signal is sparse"""
        return sig.sparse

    NoX = LinearFilter.NoX
    OneX = LinearFilter.OneX
    NoD = LinearFilter.NoD
    General = LinearFilter.General

    def make_linear_step(synapse, input_shape, output_shape, dt):
        """Call synapse.make_step to compute A/B/C/D."""
        state = synapse.make_state(input_shape, output_shape, dt)
        return synapse.make_step(input_shape, output_shape, dt, rng=None, state=state)

    def make_process_step(process, shape_in, shape_out, dt, rng, state):
        """Call process.make_step."""
        return process.make_step(shape_in, shape_out, dt, rng, state)

    def make_process_state(process, shape_in, shape_out, dt):
        """Call process.make_state."""
        return process.make_state(shape_in, shape_out, dt)
