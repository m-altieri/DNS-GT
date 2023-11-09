import tensorflow as tf


class FF(tf.keras.layers.Layer):
    """Group of densely connected layers."""

    def __init__(self, dims, activations, **kwargs):
        """Create a group of densely connected layers.
        The `dims` and `activation` parameters are zipped together as a list of pairs.

        Args:
            dims (list of int): list of layer dimensions. Must have the same length as activations.
            activations (list of str or list of functions): list of activation functions. Must have the same length as dims.
        """
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
