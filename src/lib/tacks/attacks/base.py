# -*- coding: utf-8 -*-
"""Base class for an attack.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
from abc import ABC, abstractmethod

from ..utils.base import get_logger


class BaseAttack(ABC):
    """Base class for an attack.

    Parameters
    ----------
    model : TorchModel
        Model to attack.
    tol : float, optional
        Tolerance for numerical computation (default: 1e-7).
    logger : logging.Logger or None
        Logging system.
    """

    prefix = ''

    def __init__(self, model, tol=1e-7, logger=None, verbose=True):
        if logger is None:
            logger = get_logger('', to_file=False, to_console=False)

        self.model = model
        self.tol = tol
        self.logger = logger
        self.verbose = verbose

        # container for additional information
        self._info = {}

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def generate(self, x, tg_labels=None, tg_probas=None):
        """Abstract method to generate adversarial examples. Should be overridden.

        Parameters
        ----------
        x : torch.Tensor
            Batch of instances to attack.
        tg_labels : torch.Tensor or int
            Target classes for the given instances.
        tg_probas : torch.Tensor or float, optional
            Minimal probability value associated to the target class.

        Returns
        -------
        torch.Tensor
            Batch of perturbations.
        """
        if tg_labels is None:
            if self.targeted:
                errmsg = 'Target labels need to be provided for a targeted attack.'
                raise ValueError(errmsg)

    def __str__(self):
        return f'{self.prefix}_'
