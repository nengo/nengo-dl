import warnings

from nengo.exceptions import BuildError
import tensorflow as tf

from nengo_deeplearning import DEBUG


class Builder(object):
    """Manages the operator build classes known to the nengo_deeplearning build
    process."""

    builders = {}
    op_builds = {}

    @classmethod
    def pre_build(cls, op_type, ops, signals, rng):
        """Setup step for build classes, in which they compute any of the
        values that are constant across simulation timesteps.

        Parameters
        ----------
        op_type : type
            `nengo` operator type
        ops : list of nengo.builder.operators.Operator
            the operator group to build into the model
        signals : signals.SignalDict
            dictionary mapping Signals to Tensors (updated by operations)
        rng : :class:`~numpy:numpy.random.RandomState`
            random number generator instance
        """

        if DEBUG:
            print("===================")
            print("PRE BUILD", ops)
            print("sets", [op.sets for op in ops])
            print("incs", [op.incs for op in ops])
            print("reads", [op.reads for op in ops])
            print("updates", [op.updates for op in ops])

        if op_type not in cls.builders:
            raise BuildError("Cannot build operators of type %r" % op_type)

        BuildClass = cls.builders[op_type]

        kwargs = {}
        if BuildClass.pass_rng:
            kwargs["rng"] = rng

        cls.op_builds[ops] = BuildClass(ops, signals, **kwargs)

    @classmethod
    def build(cls, ops, signals):
        """Build the computations implementing a single simulator timestep.

        Parameters
        ----------
        ops : list of nengo.builder.operators.Operator
            the operator group to build into the model
        signals : signals.SignalDict
            dictionary mapping Signals to Tensors (updated by operations)
        """

        if DEBUG:
            print("===================")
            print("BUILD", ops)

        if ops not in cls.op_builds:
            raise BuildError("Operators build has not been initialized "
                             "(missed pre-build step)")

        output = cls.op_builds[ops].build_step(signals)

        if isinstance(output, (tf.Tensor, tf.Variable)):
            output = [output]
        elif isinstance(output, tuple):
            output = list(output)

        return output

    @classmethod
    def register(cls, nengo_op):
        """A decorator for adding a class to the build function registry.

        Parameters
        ----------
        nengo_op : nengo.builder.operators.Operator
            The operator associated with the build function being decorated.
        """

        def register_builder(build_class):
            if not issubclass(build_class, OpBuilder):
                warnings.warn("Build classes should inherit from OpBuilder")

            if nengo_op in cls.builders:
                warnings.warn("Operator '%s' already has a builder. "
                              "Overwriting." % nengo_op)

            cls.builders[nengo_op] = build_class
            return build_class

        return register_builder


class OpBuilder(object):
    # set pass_rng to True if this build class requires the simulation
    # random number generator to be passed to the constructor
    pass_rng = False

    """The constructor should set up any computations that are fixed for
    this op (i.e., things that do not need to be recomputed each timestep).

    Parameters
    ----------
    ops : list of nengo.builder.operators.Operator
        the operator group to build into the model
    signals : signals.SignalDict
        dictionary mapping Signals to Tensors (updated by operations)
    """
    def __init__(self, ops, signals):
        pass

    def build_step(self, signals):
        """This function builds whatever computations need to be executed in
        each simulation timestep.

        Parameters
        ----------
        signals : signals.SignalDict
            dictionary mapping Signals to Tensors (updated by operations)

        Returns
        -------
        list of `tf.Tensor`, optional
            if not None, the returned tensors correspond to outputs with
            possible side-effects, i.e. computations that need to be executed
            in the tensorflow graph even if their output doesn't appear to be
            used
        """
        raise BuildError("OpBuilders must implement a `build_step` function")
