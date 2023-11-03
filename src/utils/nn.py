import tensorflow as tf


class FF(tf.keras.layers.Layer):
    def __init__(self, dims, activations, **kwargs):
        super().__init__(**kwargs)
        assert len(dims) == len(activations)
        self.dense_layers = [
            tf.keras.layers.Dense(dim, activation=activation)
            for dim, activation in zip(dims, activations)
        ]

    def call(self, x):
        for l in self.dense_layers:
            x = l(x)
        return x
