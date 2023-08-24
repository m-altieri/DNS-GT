# -*- coding: utf-8 -*-
"""Discrimator classes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

from .classifiers import Sequential2DClassifier
from .utils import append_one_hot_channels


class Sequential2DDiscriminator(Sequential2DClassifier):
    """Sequential architecture.

    See :py:class:`Sequential2DClassifier` for details.
    Each layer is a convolutional layer with batch normalization.

    Note: `n_classes` denotes the number of classes used in the conditional generation,
    not the number of classes of the classifier.

    Parameters
    ----------
    in_shape, layers_params, logger
        See :class:`Sequential2DClassifier`.
    n_classes : int
        Number of classes in the conditional generation.
    """

    name = 'Sequential2DDiscriminator'

    def __init__(
        self,
        name,
        in_shape,
        n_classes,
        arch,
        device=None,
        half_precision=False,
        logger=None,
    ):

        # add one hot channels in input shape
        if n_classes:
            in_shape = (in_shape[0] + n_classes,) + in_shape[1::]

        super().__init__(
            name=name,
            in_shape=in_shape,
            n_classes=1,
            arch=arch,
            with_softmax=False,
            device=device,
            half_precision=half_precision,
            logger=logger,
        )

        self.n_classes = n_classes

    def forward(self, x, y=None):

        # add one-hot channels in outputs if labels are provided
        if self.n_classes and y is not None:
            x = append_one_hot_channels(x, y, n_classes=self.n_classes)

        x = super().forward(x)

        return x
