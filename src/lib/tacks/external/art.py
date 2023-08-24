# -*- coding: utf-8 -*-
"""Torch module class.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
from ..archs.base import TorchClassifier
from art.estimators.classification import PyTorchClassifier


def get_art_classifier(classifier):
    """Return an ART classifier from tacks model."""

    if isinstance(classifier, TorchClassifier):
        errmsg = 'Model should be a TorchClassifier'
        raise ValueError(errmsg)

    return PyTorchClassifier(
        classifier,
        loss=None,
        input_shape=classifier.in_shape,
        nb_classes=classifier.n_classes,
    )
