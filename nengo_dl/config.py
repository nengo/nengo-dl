"""
The configuration system is used to change NengoDL's default behaviour in
various ways.

See `the documentation <https://www.nengo.ai/nengo-dl/config.html>`__ for more details.
"""

from nengo import Connection, Ensemble, Network, Probe, ensemble
from nengo.builder import Model
from nengo.exceptions import ConfigError, NetworkContextError
from nengo.params import BoolParam, Parameter


def configure_settings(**kwargs):
    """
    Pass settings to ``nengo_dl`` by setting them as parameters on the
    top-level Network config.

    The settings are passed as keyword arguments to ``configure_settings``;
    e.g., to set ``trainable`` use ``configure_settings(trainable=True)``.

    Parameters
    ----------
    trainable : bool or None
        Adds a parameter to Nengo Ensembles/Connections that controls
        whether or not they will be optimized by `.Simulator.fit`.
        Passing ``None`` will use the default ``nengo_dl`` trainable settings,
        or True/False will override the default for all objects.  In either
        case trainability can be further configured on a per-object basis (e.g.
        ``net.config[my_ensemble].trainable = True``.  See `the documentation
        <https://www.nengo.ai/nengo-dl/simulator.html#choosing-which-elements-to-optimize>`__
        for more details.
    planner : graph planning algorithm
        Pass one of the `graph planners
        <https://www.nengo.ai/nengo-dl/reference.html#graph-optimization>`_ to change
        the default planner.
    sorter : signal sorting algorithm
        Pass one of the `sort algorithms
        <https://www.nengo.ai/nengo-dl/reference.html#graph-optimization>`_ to change
        the default sorter.
    simplifications: list of graph simplification functions
        Pass a list of `graph simplification functions
        <https://www.nengo.ai/nengo-dl/reference.html#graph-optimization>`_ to change
        the default simplifications applied.  The default list of simplifications
        can be found in ``nengo_dl.graph_optimizer.default_simplifications``.
    inference_only : bool
        Set to True if the network will only be run in inference mode (i.e.,
        no calls to `.Simulator.fit`).  This may result in a small
        increase in the inference speed.
    lif_smoothing : float
        If specified, use the smoothed `.SoftLIFRate` neuron
        model, with the given smoothing parameter (``sigma``),
        to compute the gradient for `~nengo.LIF` neurons (as
        opposed to using `~nengo.LIFRate`).
    dtype : ``tf.DType``
        Set the floating point precision for simulation values.
    keep_history : bool
        Adds a parameter to Nengo Probes that controls whether or not they
        will keep the history from all simulation timesteps or only the last
        simulation step.  This can be further configured on a per-probe basis
        (e.g., ``net.config[my_probe].keep_history = False``).
    stateful : bool
        If True (default), the Simulator will be built to support stateful execution
        (where internal simulation state is preserved between simulator functions such
        as `.Simulator.predict`).  Otherwise all operations will be stateless. Note that
        this can also be configured individually through the ``stateful`` parameter on
        individual functions.
    use_loop : bool
        If True (default), use a symbolic while loop to run the simulation. Otherwise,
        simulation iterations are explicitly built into the model, avoiding the
        while loop. This can improve performance, but the simulation can only run
        for exactly ``Simulator.unroll_simulation`` iterations.
    learning_phase : bool
        The learning phase is used for models that have different behaviour in inference
        versus training mode (for example, spiking neurons swap their behaviour during
        training). Normally the learning phase is set automatically depending on
        what function is being called (e.g. ``sim.predict`` will run in inference
        mode and ``sim.fit`` will run in training mode). However, sometimes it can
        be useful to override this behaviour (for example if we want to check
        what the output of the model looks like during training, we might want to
        run ``sim.predict`` in training mode).  Set ``learning_phase=True`` to
        always run the model in training mode (or ``False`` to always run in inference
        mode). Set ``learning_phase=None`` to use the default behaviour.

        .. versionadded:: 3.3.0
    """

    # get the toplevel network
    if len(Network.context) > 0:
        config = Network.context[0].config
    else:
        raise NetworkContextError(
            "`configure_settings` must be called within a Network context "
            "(`with nengo.Network(): ...`)"
        )

    try:
        params = config[Network]
    except ConfigError:
        config.configures(Network)
        params = config[Network]

    for attr, val in kwargs.items():
        if attr == "trainable":
            # for trainable, we set it on the nearest containing network (rather than
            # the top-level)
            sub_config = Network.context[-1].config
            for obj in (Ensemble, Connection, ensemble.Neurons):
                try:
                    obj_params = sub_config[obj]
                except ConfigError:
                    sub_config.configures(obj)
                    obj_params = sub_config[obj]

                obj_params.set_param(
                    "trainable", BoolParam("trainable", val, optional=True)
                )
        elif attr == "keep_history":
            config[Probe].set_param("keep_history", BoolParam("keep_history", val))
        elif attr in (
            "planner",
            "sorter",
            "simplifications",
            "inference_only",
            "lif_smoothing",
            "dtype",
            "stateful",
            "use_loop",
            "learning_phase",
        ):
            params.set_param(attr, Parameter(attr, val))
        else:
            raise ConfigError(f"{attr} is not a valid config parameter")


def get_setting(model, setting, default=None, obj=None):
    """
    Returns config settings (created by `.configure_settings`).

    Parameters
    ----------
    model : `~nengo.builder.Model` or `~nengo.Network`
        Built model or Network containing all the config settings.
    setting : str
        Name of the config option to return.
    default
        The default value to return if config option not set.
    obj : ``NengoObject``
        The object on which config setting is stored (defaults to the top-level
        network).

    Returns
    -------
    config_val
        Value of ``setting`` if it has been specified, else ``default``.
    """

    if isinstance(model, Model):
        if model.toplevel is None:
            return default
        model = model.toplevel

    if obj is None:
        obj = model

    try:
        return getattr(model.config[obj], setting, default)
    except ConfigError:
        return default
