"""
Objects to be used with the Keras callback functionality.

See https://www.tensorflow.org/guide/keras/custom_callback for more information
on how to use Keras callbacks.

The short answer is that these can be passed to, e.g., `.Simulator.fit` like

.. code-block:: python

    sim.fit(..., callbacks=[nengo_dl.callbacks.NengoSummaries(...)]

"""

import contextlib

import nengo
import tensorflow as tf
from nengo.exceptions import ValidationError
from tensorflow.python.eager import context

from nengo_dl import compat, utils


class NengoSummaries(tf.keras.callbacks.Callback):
    """
    Logs the values of Nengo object parameters, to be displayed in TensorBoard.

    See https://www.tensorflow.org/tensorboard/get_started for general instructions
    on using TensorBoard.

    Parameters
    ----------
    log_dir : str
        Directory where log file will be written.
    sim : `.Simulator`
        Simulator object which will be used to look up parameter values.
    objects : list of `nengo.Ensemble` or `nengo.ensemble.Neurons` or `nengo.Connection`
        The object whose parameter values we want to record (passing an Ensemble will
        log its encoders, Neurons will log biases, and Connection will log connection
        weights/decoders).
    """

    def __init__(self, log_dir, sim, objects):
        super().__init__()

        self.sim = sim

        with contextlib.suppress() if compat.eager_enabled() else context.eager_mode():
            self.writer = tf.summary.create_file_writer(str(log_dir))

        self.summaries = []
        for obj in objects:
            if isinstance(
                obj, (nengo.Ensemble, nengo.ensemble.Neurons, nengo.Connection)
            ):
                if isinstance(obj, nengo.Ensemble):
                    param = "encoders"
                    name = f"Ensemble_{obj.label}"
                elif isinstance(obj, nengo.ensemble.Neurons):
                    param = "bias"
                    name = f"Ensemble.neurons_{obj.ensemble.label}"
                elif isinstance(obj, nengo.Connection):
                    if not compat.conn_has_weights(obj):
                        raise ValidationError(
                            f"Connection '{obj}' does not have any weights to log",
                            "objects",
                        )
                    param = "weights"
                    name = f"Connection_{obj.label}"

                self.summaries.append(
                    (utils.sanitize_name(f"{name}_{param}"), obj, param)
                )
            else:
                raise ValidationError(
                    f"Unknown summary object {obj}; should be an Ensemble, Neurons, or "
                    "Connection",
                    "objects",
                )

    def on_epoch_end(self, epoch, logs=None):
        """Log parameter values at the end of each epoch."""

        summary_vals = self.sim.data.get_params(
            *[(obj, attr) for _, obj, attr in self.summaries]
        )

        with (
            contextlib.suppress() if compat.eager_enabled() else context.eager_mode()
        ), self.writer.as_default():
            for (name, _, _), val in zip(self.summaries, summary_vals):
                tf.summary.histogram(name, val, step=epoch)

    def on_train_end(self, logs=None):
        """Close summary writer at end of training."""

        with contextlib.suppress() if compat.eager_enabled() else context.eager_mode():
            self.writer.close()


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A version of the Keras TensorBoard callback that also profiles inference.
    """

    def on_predict_batch_end(self, batch, logs=None):
        """Redirect to training function."""
        self.on_batch_end(batch, logs=logs)

    def on_predict_begin(self, logs=None):
        """Redirect to training function."""
        self.on_train_begin(logs=logs)

    def on_predict_end(self, logs=None):
        """Redirect to training function."""
        self.on_train_end(logs=logs)


class IsolateState(tf.keras.callbacks.Callback):
    """
    Isolate the internal state of the simulation from any other stateful operations.

    This will cause every batch to begin from the same initial state (the state of
    the simulation whenever this callback is created). And when this operation
    completes, the simulation state will be returned to that initial state.

    Parameters
    ----------
    sim : `.Simulator`
        The Simulator containing the state we want to control.
    """

    def __init__(self, sim):
        super().__init__()

        self.sim = sim
        self.saved_state = (
            None
            if sim.n_steps == 0
            else tf.keras.backend.batch_get_value(
                list(sim.tensor_graph.saved_state.values())
            )
        )

    def reset(self):
        """Resets the simulation state to the saved state."""

        if self.saved_state is None:
            self.sim.reset(
                include_probes=False, include_trainable=False, include_processes=False
            )
        else:
            tf.keras.backend.batch_set_value(
                list(zip(self.sim.tensor_graph.saved_state.values(), self.saved_state))
            )
            self.sim._update_steps()

    def on_train_batch_end(self, batch, logs=None):
        """Reset state at the end of each batch."""
        self.reset()

    def on_predict_batch_end(self, batch, logs=None):
        """Reset state at the end of each batch."""
        self.reset()

    def on_test_batch_end(self, batch, logs=None):
        """Reset state at the end of each batch."""
        self.reset()
