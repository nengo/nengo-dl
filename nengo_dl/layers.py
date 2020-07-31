import numpy as np
import tensorflow as tf


class RegularSpikingCell(tf.keras.layers.Layer):
    def __init__(self, units, activation, dt=0.001, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.activation = activation
        self.dt = dt

        self.output_size = (units,)
        self.state_size = (units,)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.random.uniform((batch_size, self.units), dtype=dtype)

    def call(self, inputs, states, training=None):
        voltage = states[0]

        if training:
            return self.activation(inputs), voltage
        else:
            voltage += self.activation(inputs) * self.dt
            n_spikes = tf.floor(voltage)
            voltage -= n_spikes
            spikes = n_spikes / self.dt

            return spikes, voltage


inp = x = tf.keras.layers.Input((None, 1))
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=10))(x)
x = tf.keras.layers.RNN(
    RegularSpikingCell(10, tf.nn.leaky_relu), return_sequences=True
)(x)

model = tf.keras.Model(inp, x)

model.compile(loss=tf.losses.mse, optimizer=tf.optimizers.SGD(0.01))
model.fit(np.ones((1, 100, 1)), np.ones((1, 100, 10)) * np.arange(10) * 10, epochs=1000)

y = model.predict(np.ones((1, 1000, 1)))

print(np.sum(y > 0, axis=1))
