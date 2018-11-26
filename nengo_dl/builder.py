"""
The builder manages the mapping between (groups of) Nengo operators and the
builder objects that know how to translate those operators into a
TensorFlow graph.
"""

from collections import namedtuple
import logging
import warnings

from nengo import builder
from nengo.builder import signal
from nengo.exceptions import BuildError
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class Builder:
    """Manages the operator build classes known to the ``nengo_dl``
    build process."""

    builders = {}

    @classmethod
    def pre_build(cls, ops, signals, op_builds, config):
        """
        Setup step for build classes, in which they compute any of the
        values that are constant across simulation timesteps.

        Parameters
        ----------
        ops : tuple of `~nengo.builder.Operator`
            The operator group to build into the model
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        op_builds : dict of {tuple of `~nengo.builder.Operator`, `.OpBuilder`}
            ``pre_build`` will populate this dictionary with the OpBuilder
            objects (which execute the pre-build step in their ``__init__``)
        """

        logger.debug("===================")
        logger.debug("PRE BUILD %s", ops)
        logger.debug("sets %s", [op.sets for op in ops])
        logger.debug("incs %s", [op.incs for op in ops])
        logger.debug("reads %s", [op.reads for op in ops])
        logger.debug("updates %s", [op.updates for op in ops])

        if type(ops[0]) not in cls.builders:
            raise BuildError("No registered builder for operators of type %r" %
                             type(ops[0]))

        BuildClass = cls.builders[type(ops[0])]

        op_builds[ops] = BuildClass(ops, signals, config)

    @classmethod
    def build(cls, ops, signals, op_builds):
        """
        Build the computations implementing a single simulator timestep.

        Parameters
        ----------
        ops : tuple of `~nengo.builder.Operator`
            The operator group to build into the model
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        op_builds : dict of {tuple of `~nengo.builder.Operator`, \
                             ~`.op_builders.OpBuilder`}
            Mapping from operator groups to the pre-built builder objects
        """

        logger.debug("===================")
        logger.debug("BUILD %s", ops)

        if ops not in op_builds:
            raise BuildError("Operators build has not been initialized "
                             "(missed pre-build step)")

        output = op_builds[ops].build_step(signals)

        if isinstance(output, (tf.Tensor, tf.Variable)):
            output = [output]
        elif isinstance(output, tuple):
            output = list(output)

        return output

    @classmethod
    def register(cls, nengo_op):
        """
        A decorator for adding a class to the build function registry.

        Parameters
        ----------
        nengo_op : `~nengo.builder.Operator`
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


class BuildConfig(namedtuple("BuildConfig", (
        "inference_only", "lif_smoothing", "cpu_only"))):
    """
    Stores configuration parameters that may be relevant to parts of the
    build process.

    Parameters
    ----------
    inference_only : bool
        If True the network should be constructed in "inference only" mode
        (not including any support for training operations).
    lif_smoothing : float
        Smoothing parameter for `~nengo.LIF` gradient approximation.
    cpu_only : bool
        True if TensorFlow is only running on the CPU (because that was
        specified by the user or because tensorflow-gpu is not installed).
    """

    __slots__ = ()


class OpBuilder:  # pragma: no cover
    """
    The constructor should set up any computations that are fixed for
    this op (i.e., things that do not need to be recomputed each timestep).

    Parameters
    ----------
    ops : list of `~nengo.builder.Operator`
        The operator group to build into the model
    signals : `.signals.SignalDict`
        Mapping from `~nengo.builder.Signal` to
        ``tf.Tensor`` (updated by operations)
    config : `~.builder.BuildConfig`
        General repository for config information builders might want
        (conglomerated into this object so that we can add/remove config data
        without having to change the function signature all the time).
    """

    def __init__(self, ops, signals, config):
        logger.debug(self.__class__.__name__)
        logger.debug("\n".join(str(x) for x in ops))

        self.config = config

    def build_step(self, signals):
        """
        This function builds whatever computations need to be executed in
        each simulation timestep.

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)

        Returns
        -------
        side_effects : list of ``tf.Tensor``
            If not None, the returned tensors correspond to outputs with
            possible side-effects, i.e. computations that need to be executed
            in the TensorFlow graph even if their output doesn't appear to be
            used
        """
        raise BuildError("OpBuilders must implement a `build_step` function")

    def build_post(self, ops, signals, sess, rng):
        """
        This function will be called after the graph has been built and
        session/variables initialized.

        This should be used to build any random aspects of the operator.

        Note that this function may be called multiple times per session, so
        it should modify the graph in-place.

        Parameters
        ----------
        ops : list of `~nengo.builder.Operator`
            The operator group to build into the model
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        sess : ``tf.Session``
            The initialized simulation session
        rng : `~numpy.random.RandomState`
            Seeded random number generator
        """

    @staticmethod
    def mergeable(x, y):
        """
        Compute the mergeability of two operators of this builder's type.

        Parameters
        ----------
        x : :class:`~nengo:nengo.builder.Operator`
            The operator being tested
        y : :class:`~nengo:nengo.builder.Operator`
            The operator being merged into (this is representative of a group
            of operators that have already been merged)

        Returns
        -------
        mergeable : bool
            True if ``x`` and ``y`` can be merged into a single built op,
            else ``False``.
        """

        return False


class NengoBuilder(builder.Builder):
    """
    Copy of the default Nengo builder.

    This class is here so that we can register new build functions for
    Nengo DL without affecting the default Nengo build process.
    """

    builders = {}

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        """
        Build ``obj`` into ``model``.

        This method looks up the appropriate build function for ``obj`` and
        calls it with the model and other arguments provided.

        In addition to the parameters listed below, further positional and
        keyword arguments will be passed unchanged into the build function.

        Parameters
        ----------
        model : Model
            The `~nengo.builder.Model` instance in which to store build
            artifacts.
        obj : object
            The object to build into the model.
        """

        try:
            # first try building obj using any custom build functions that have
            # been registered by Nengo DL
            return builder.Builder.build.__func__(
                NengoBuilder, model, obj, *args, **kwargs)
        except BuildError:
            # fallback on normal nengo builder
            return builder.Builder.build.__func__(
                builder.Builder, model, obj, *args, **kwargs)


class NengoModel(builder.Model):
    """
    Copy of the default Nengo model.

    This allows us to override certain model behaviours.

    Parameters
    ----------
    fail_fast : bool
        If True, try to call ``op.make_step`` when ops are added to the model.
        Note that NengoDL doesn't actually use ``make_step``, so errors in that
        function are not necessarily errors in NengoDL (which is why we want to
        disable that check). But it might still be useful when debugging
        new op/build functions, which is why we leave the option.
    """

    def __init__(self, *args, fail_fast=True, **kwargs):
        self.fail_fast = fail_fast
        super(NengoModel, self).__init__(*args, **kwargs)

    def add_op(self, op):
        """
        Add an operator to the model.

        Parameters
        ----------
        op : `~nengo.builder.Operator`
            Operator being added to the model.

        Notes
        -----
        This is a copy of the parent `nengo.builder.Model.add_op`, with the
        addition of the ``if self.fail_fast`` condition.
        """

        self.operators.append(op)
        if self.fail_fast:
            # Fail fast by trying make_step with a temporary sigdict
            signals = signal.SignalDict()
            op.init_signals(signals)
            op.make_step(signals, self.dt, np.random)
