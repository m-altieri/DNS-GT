import tensorflow as tf


class WeightedSCCE(tf.keras.losses.Loss):

    def __init__(
        self,
        class_weights,
        from_logits=False,
        name="weighted_spatial_categorical_crossentropy",
    ):
        if class_weights is None or tf.math.reduce_all(
            tf.math.equal(class_weights, 1.0)
        ):
            self.class_weights = None
        self.reduction = tf.keras.losses.Reduction.NONE
        self.unreduced_scce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction=self.reduction
        )

    def call(self, inputs):
        pass
