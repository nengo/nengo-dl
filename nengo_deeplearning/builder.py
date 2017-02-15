import warnings

from nengo.exceptions import BuildError
import tensorflow as tf

from nengo_deeplearning import DEBUG


class Builder(object):
    """Manages the operator build classes known to the ``nengo_deeplearning``
    build process."""

    builders = {}
    op_builds = {}

    @classmethod
    def pre_build(cls, ops, signals, rng):
        """Setup step for build classes, in which they compute any of the
        values that are constant across simulation timesteps.

        Parameters
        ----------
        ops : list of :class:`~nengo:nengo.builder.Operator`
            the operator group to build into the model
        signals : :class:`.signals.SignalDict`
            mapping from :class:`~nengo:nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
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

        if type(ops[0]) not in cls.builders:
            raise BuildError("No registered builder for operators of type %r" %
                             type(ops[0]))

        BuildClass = cls.builders[type(ops[0])]

        kwargs = {}
        if BuildClass.pass_rng:
            kwargs["rng"] = rng

        cls.op_builds[ops] = BuildClass(ops, signals, **kwargs)

    @classmethod
    def build(cls, ops, signals):
        """Build the computations implementing a single simulator timestep.

        Parameters
        ----------
        ops : list of :class:`~nengo:nengo.builder.Operator`
            the operator group to build into the model
        signals : :class:`.signals.SignalDict`
            mapping from :class:`~nengo:nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
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
        nengo_op : :class:`~nengo:nengo.builder.Operator`
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
    """The constructor should set up any computations that are fixed for
    this op (i.e., things that do not need to be recomputed each timestep).

    Parameters
    ----------
    ops : list of :class:`~nengo:nengo.builder.Operator`
        the operator group to build into the model
    signals : :class:`.signals.SignalDict`
        mapping from :class:`~nengo:nengo.builder.Signal` to
        ``tf.Tensor`` (updated by operations)

    Attributes
    ----------
    pass_rng : bool
        set to True if this build class requires the simulation
        random number generator to be passed to the constructor
    """

    pass_rng = False

    def __init__(self, ops, signals):
        pass

    def build_step(self, signals):
        """This function builds whatever computations need to be executed in
        each simulation timestep.

        Parameters
        ----------
        signals : :class:`.signals.SignalDict`
            mapping from :class:`~nengo:nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)

        Returns
        -------
        list of ``tf.Tensor``, optional
            if not None, the returned tensors correspond to outputs with
            possible side-effects, i.e. computations that need to be executed
            in the tensorflow graph even if their output doesn't appear to be
            used
        """
        raise BuildError("OpBuilders must implement a `build_step` function")
