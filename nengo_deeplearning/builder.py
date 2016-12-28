import inspect
import warnings

import nengo
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

    @classmethod
    def build(cls, op, signals, dt, rng):
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

        if type(op) not in cls.builders:
            print(cls.builders)
            raise nengo.exceptions.BuildError(
                "Cannot build operators of type %r" % type(op))

        if DEBUG:
            print("===================")
            print("CONVERTING", op)
            print("sets", op.sets)
            print("incs", op.incs)
            print("reads", op.reads)
            print("updates", op.updates)

        for sig in op.reads + op.incs:
            if sig not in signals:
                if DEBUG:
                    print("creating variable", sig)
                signals.create_variable(sig)

        build_func = cls.builders[type(op)]
        kwargs = {}
        if build_func._pass_dt:
            kwargs["dt"] = dt
        if build_func._pass_rng:
            kwargs["rng"] = rng
        output = build_func(op, signals, **kwargs)

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
        nengo_class : nengo.builder.operators.Operator
            The operator associated with the build function being decorated.
        """

        def register_builder(build_fn):
            if nengo_op in cls.builders:
                warnings.warn("Operator '%s' already has a builder. "
                              "Overwriting." % nengo_op)

            param_names = inspect.signature(build_fn).parameters

            build_fn._pass_dt = "dt" in param_names
            build_fn._pass_rng = "rng" in param_names

            cls.builders[nengo_op] = build_fn
            return build_fn

        return register_builder
