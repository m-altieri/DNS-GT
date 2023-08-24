# -*- coding: utf-8 -*-
"""Handling LightGBM models.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import numpy as np

from .base import BaseModelHandler


class LightGbmModelHandler(BaseModelHandler):
    """Class to handle LightGBM models.

    Parameters
    ----------
    model : lightgbm.Booster
        Model to handle.
    others :
        See :class:`BaseModelHandler`.
    """
    @staticmethod
    def load(model_path, logger=None, **kwargs):
        """Load a lightGBM model handler.

        Parameters
        ----------
        model_path : pathlib.Path or str
            Path to the saved model.
        See :class:`BaseModelHandler` for other parameters.

        Returns
        -------
        LightGbmModelHandler
        """
        model_path, device = super().load(model_path, **kwargs)
        raise NotImplementedError

    def predict(self, features):
        """Predict the results of a given batch of instances.

        Parameters
        ----------
        features : array-like
            Features.

        Returns
        -------
        prediction
        """
        outputs = self.model.predict([features])
        pred_class = np.argmax(outputs)
        pred_proba = outputs[pred_class]

        return outputs, (pred_class, pred_proba)
