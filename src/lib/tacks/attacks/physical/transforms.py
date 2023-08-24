# -*- coding: utf-8 -*-
"""Transforms for physical attacks.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import kornia
import numpy as np
import torch


class GammaTransform(torch.nn.Module):
    """Change gamma correction.

    :param:`gamma` is a nonnegative real number. If :param:`gamma` is smaller than 1,
    dark regions appear lighter. Conversely, if :param:`gamma` is larger than 1, dark
    regions appear darker.

    At each transformation, a random value for gamma is drawn between (1 +
    :param:`gamma_min`) and (1 + `gamma_max`). The value is then inversed (1/gamma) with
    a probability :param:`flip_p`.

    Parameters
    ----------
    gamma_min : scalar

    gamma_max : scalar

    flip_p : scalar
    """

    def __init__(self, gamma_min=0, gamma_max=1, flip_p=0.5):
        super().__init__()

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.flip_p = flip_p

    def forward(self, x):
        if self.gamma_min < self.gamma_max:
            gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        else:
            gamma = self.gamma_min

        if np.random.rand() < self.flip_p:
            gamma = 1.0 / gamma

        return kornia.enhance.adjust_gamma(x, gamma)
