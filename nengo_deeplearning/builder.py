import inspect
import warnings

from nengo.exceptions import BuildError
import tensorflow as tf

from nengo_deeplearning import DEBUG


class Builder(object):
    """Manages the build functions known to the nengo_deeplearning build
    process.

    Consists of two class methods to encapsulate the build function registry.
    All build functions should use the `.Builder.register` method as a
    decorator. For example::

        @nengo.builder.Builder.register(MyRule)
        def build_my_rule(model, my_rule, rule):
            ...

    registers a build function for ``MyRule`` objects.

    Build functions should not be called directly, but instead called through
    the `.Model.build` method. `.Model.build` uses the `.Builder.build` method
    to ensure that the correct build function is called based on the type of
    the object passed to it.
    For example, to build the learning rule type ``my_rule`` from above, do::

        model.build(my_rule, connection.learning_rule)

    This will call the ``build_my_rule`` function from above with the arguments
    ``model, my_rule, connection.learning_rule``.

    Attributes
    ----------
    builders : dict
        Mapping from types to the build function associated with that type.
    """

    builders = {}
    op_builds = {}

    @classmethod
    def pre_build(cls, op_type, ops, signals, rng):
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
        """Build ``op`` into ``model``.

        This method looks up the appropriate build function for ``obj`` and
        calls it with the SignalDict and other arguments provided.

        In addition to the parameters listed below, further positional and
        keyword arguments will be passed unchanged into the build function.

        Parameters
        ----------
        op : object
            The operator to build into the model.
        signals : signals.SignalDict
            Dictionary mapping Signals to Tensors (updated by operations)
        dt : float
            Simulation time step
        rng : np.random.RandomState
            Random number generator
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

        Raises a warning if a build function already exists for the class.

        Parameters
        ----------
        nengo_op : nengo.builder.operators.Operator
            The operator associated with the build function being decorated.
        """

        def register_builder(build_class):
            # if not issubclass(build_class, OpBuilder):
            #     warnings.warn("Build classes should inherit from OpBuilder")

            if nengo_op in cls.builders:
                warnings.warn("Operator '%s' already has a builder. "
                              "Overwriting." % nengo_op)

            cls.builders[nengo_op] = build_class
            return build_class

        return register_builder


class OpBuilder(object):
    pass_rng = False

    def __init__(self, ops, signals):
        # the constructor should set up any computations that are fixed for
        # this op (i.e., things that do not need to be recomputed each
        # timestep
        pass

    def build_step(self, signals):
        # here we build whatever computations need to be executed in each
        # simulation timestep
        raise BuildError("OpBuilders must implement a build_step function")
