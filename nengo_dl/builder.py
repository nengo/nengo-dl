"""
The builder manages the mapping between (groups of) Nengo operators and the
builder objects that know how to translate those operators into a
TensorFlow graph.
"""

import logging
import warnings
from collections import namedtuple

import numpy as np
import tensorflow as tf
from nengo import builder
from nengo.builder import signal
from nengo.exceptions import BuildError

from nengo_dl import utils

logger = logging.getLogger(__name__)


class Builder:
    """
    Manages the operator build classes known to the ``nengo_dl`` build process.

    Parameters
    ----------
    plan : list of tuple of `~nengo.builder.Operator`
        The groups of operators that will be built
    """

    builders = {}

    def __init__(self, plan):
        self.plan = plan
        self.op_builds = {}

        for ops in self.plan:
            if type(ops[0]) not in Builder.builders:
                raise BuildError(
                    f"No registered builder for operators of type {type(ops[0])!r}"
                )
            self.op_builds[ops] = Builder.builders[type(ops[0])](ops)

    def build_pre(self, signals, config, progress=None):
        """
        Setup step for build classes, in which they compute any of the
        values that are constant across simulation timesteps.

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        config : `.BuildConfig`
            Configuration parameters for the build process
        progress : `.utils.ProgressBar`
            Progress bar for ops in plan
        """

        for ops in self.plan:
            logger.debug("===================")
            logger.debug("PRE BUILD %s", ops)
            logger.debug("sets %s", [op.sets for op in ops])
            logger.debug("incs %s", [op.incs for op in ops])
            logger.debug("reads %s", [op.reads for op in ops])
            logger.debug("updates %s", [op.updates for op in ops])

            with self.name_scope(ops):
                self.op_builds[ops].build_pre(signals, config)

            if progress is not None:
                progress.step()

    def build_step(self, signals, progress=None):
        """
        Build the computations implementing a single simulator timestep.

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        progress : `.utils.ProgressBar`
            Progress bar for ops in plan

        Returns
        -------
        side_effects : list of ``tf.Tensor``
            Outputs with possible side-effects, i.e. computations that need to
            be executed in the TensorFlow graph even if their output doesn't
            appear to be used.
        """

        side_effects = []
        for ops in self.plan:
            logger.debug("===================")
            logger.debug("BUILD %s", ops)

            with self.name_scope(ops):
                output = self.op_builds[ops].build_step(signals)

            if isinstance(output, (tf.Tensor, tf.Variable)):
                side_effects.append(output)
            elif isinstance(output, (list, tuple)):
                side_effects.extend(list(output))

            if progress is not None:
                progress.step()

        return side_effects

    def build_post(self, signals, progress=None):
        """
        Calls post build functions for all ops in plan.

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        progress : `.utils.ProgressBar`
            Progress bar for ops in plan
        """

        for ops in self.plan:
            logger.debug("===================")
            logger.debug("POST BUILD %s", ops)

            with self.name_scope(ops):
                self.op_builds[ops].build_post(signals)

            if progress is not None:
                progress.step()

    def name_scope(self, ops):
        """Returns a new TensorFlow name scope for the given ops."""

        return tf.name_scope(
            utils.sanitize_name(Builder.builders[type(ops[0])].__name__)
        )

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
                warnings.warn(
                    f"Operator '{nengo_op}' already has a builder. Overwriting."
                )

            cls.builders[nengo_op] = build_class
            return build_class

        return register_builder


class BuildConfig(
    namedtuple(
        "BuildConfig",
        ("inference_only", "lif_smoothing", "cpu_only", "rng", "training"),
    )
):
    """
    Stores configuration parameters that may be relevant to parts of the
    build process.

    Parameters
    ----------
    inference_only : bool
        If True the network should be constructed in "inference only" mode
        (omitting any support for training operations).
    lif_smoothing : float
        Smoothing parameter for `~nengo.LIF` gradient approximation.
    cpu_only : bool
        True if TensorFlow is only running on the CPU (because that was
        specified by the user or because GPU support is not available).
    rng : `~numpy.random.RandomState`
        Seeded random number generator.
    training : ``tf.Tensor`` (bool)
        True if building in training mode, False for inference mode.
    """

    __slots__ = ()


class OpBuilder:
    """
    Base class for build classes, which implement the logic for building a group of
    Nengo Operators into TensorFlow.
    """

    def __init__(self, ops):
        """
        Initialize internal OpBuilder implementation.

        Note: this should not be building any model operations, this is purely for
        internal setup of the ``OpBuilder`` itself.

        Parameters
        ----------
        ops : list of `~nengo.builder.Operator`
            The operator group to build into the model
        """

        self.ops = ops

    def build_pre(self, signals, config):
        """
        This function should set up any computations that are fixed for
        this op (i.e., things that do not need to be recomputed each timestep).

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        config : `~.builder.BuildConfig`
            General repository for config information builders might want
            (conglomerated into this object so that we can add/remove config data
            without having to change the function signature all the time).
        """
        logger.debug(self.__class__.__name__)
        logger.debug("\n".join(str(x) for x in self.ops))

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

    def build_post(self, signals):
        """
        This function will be called after the graph has been built and
        each time the Simulator is reset.

        Note that this function may be called multiple times per session, so
        it should do any required operations in-place.

        Parameters
        ----------
        signals : `.signals.SignalDict`
            Mapping from `~nengo.builder.Signal` to
            ``tf.Tensor`` (updated by operations)
        """

    @staticmethod
    def mergeable(x, y):
        """
        Compute the mergeability of two operators of this builder's type.

        Parameters
        ----------
        x : `nengo.builder.Operator`
            The operator being tested
        y : `nengo.builder.Operator`
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
    NengoDL without affecting the default Nengo build process.
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
            # been registered by NengoDL
            return builder.Builder.build.__func__(
                NengoBuilder, model, obj, *args, **kwargs
            )
        except BuildError:
            # fallback on normal nengo builder
            return builder.Builder.build.__func__(
                builder.Builder, model, obj, *args, **kwargs
            )


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
        super().__init__(*args, **kwargs)

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

        # TODO: nengo 3.0 adds something similar to this condition, but
        # it uses an rc setting (so we can't change it in nengo-dl without
        # also changing nengo core). if the rc system is reworked to allow
        # backend-specific overrides, we could remove this class.

        self.operators.append(op)
        if self.fail_fast:
            # Fail fast by trying make_step with a temporary sigdict
            signals = signal.SignalDict()
            op.init_signals(signals)
            op.make_step(signals, self.dt, np.random)
