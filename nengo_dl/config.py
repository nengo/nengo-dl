"""
The configuration system is used to change NengoDL's default behaviour in
various ways.
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
        Adds a parameter to Nengo Ensembles/Connections/Networks that controls
        whether or not they will be optimized by `.Simulator.train`.
        Passing ``None`` will use the default ``nengo_dl`` trainable settings,
        or True/False will override the default for all objects.  In either
        case trainability can be further configured on a per-object basis (e.g.
        ``net.config[my_ensemble].trainable = True``.  See `the documentation
        <https://www.nengo.ai/nengo-dl/training.html#choosing-which-elements-to-optimize>`_
        for more details.
    planner : graph planning algorithm
        Pass one of the `graph planners
        <https://www.nengo.ai/nengo-dl/graph_optimizer.html>`_ to change the
        default planner.
    sorter : signal sorting algorithm
        Pass one of the `sort algorithms
        <https://www.nengo.ai/nengo-dl/graph_optimizer.html>`_ to change the
        default sorter.
    simplifications: list of graph simplification functions
        Pass a list of `graph simplification functions
        <https://www.nengo.ai/nengo-dl/graph_optimizer.html>`_ to change the
        default simplifications applied.
    session_config: dict
        Config options passed to ``tf.Session`` initialization (e.g., to change
        the `GPU memory allocation method
        <https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth>`_
        pass ``{"gpu_options.allow_growth": True}``).
    inference_only : bool
        Set to True if the network will only be run in inference mode (i.e.,
        no calls to `.Simulator.train`).  This may result in a small
        increase in the inference speed.
    lif_smoothing : float
        If specified, use the smoothed `~.neurons.SoftLIFRate` neuron
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
    """

    # get the toplevel network
    if len(Network.context) > 0:
        config = Network.context[0].config
    else:
        raise NetworkContextError(
            "`configure_settings` must be called within a Network context "
            "(`with nengo.Network(): ...`)")

    try:
        params = config[Network]
    except ConfigError:
        config.configures(Network)
        params = config[Network]

    for attr, val in kwargs.items():
        if attr == "trainable":
            for obj in (Ensemble, Connection, ensemble.Neurons, Network):
                try:
                    obj_params = config[obj]
                except ConfigError:
                    config.configures(obj)
                    obj_params = config[obj]

                obj_params.set_param("trainable", BoolParam("trainable", val,
                                                            optional=True))
        elif attr == "keep_history":
            config[Probe].set_param("keep_history",
                                    BoolParam("keep_history", val))
        elif attr in ("planner", "sorter", "simplifications",
                      "session_config", "inference_only", "lif_smoothing",
                      "dtype"):
            params.set_param(attr, Parameter(attr, val))
        else:
            raise ConfigError("%s is not a valid config parameter" % attr)


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
