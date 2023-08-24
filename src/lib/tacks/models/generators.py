# -*- coding: utf-8 -*-
"""Generator classes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch

from ..data.transforms import Unnormalize
from .base import TorchModel
from .utils import append_one_hot_vector


class TorchGenerator(TorchModel):
    """Torch model for generation.

    Parameters
    ----------
    n_classes : int or None, optional
        Number of classes for conditional generation. If None, generation is
        unconditional (default: None).
    name, in_shape, out_shape, arch, device, half_precision, logger :
        See :py:class:`TorchModel`.
    """

    def __init__(
        self,
        name,
        in_shape,
        out_shape,
        arch,
        n_classes=None,
        device=None,
        half_precision=None,
        logger=None,
    ):

        super().__init__(
            name=name,
            in_shape=in_shape,
            out_shape=out_shape,
            arch=arch,
            device=device,
            half_precision=half_precision,
            logger=logger,
        )

        if n_classes is not None:
            in_shape[0] += n_classes

        self.n_classes = n_classes

    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes):

        if n_classes is not None and n_classes < 2:
            err_msg = 'Number of classes should be higher than 1 (given: {:d}'
            raise ValueError(err_msg.format(n_classes))

        self._n_classes = n_classes

    def generate(self, n_samples, labels=None):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        self.gen_model.eval()

        gen_inputs = self.gen_model.generate_input(n_samples, self.device)
        gen_instances = self.gen_model(gen_inputs, labels).cpu()

        return self.gen_model.post_forward(gen_instances)


class Sequential2dGenerator(TorchGenerator):
    """Generator with fully-connected layers.

    Parameters
    ----------
    name, in_shape, out_shape, arch, device, half_precision, logger :
        See :py:class:`TorchModel`.
    n_classes : int or None
        Number of classes for conditional generation. If None, generation is
        unconditional (default: None).
    """

    def __init__(
        self,
        name,
        in_shape,
        out_shape,
        arch,
        n_classes=None,
        device=None,
        half_precision=None,
        logger=None,
    ):

        super().__init__(
            name=name,
            in_shape=in_shape,
            out_shape=out_shape,
            arch=arch,
            n_classes=n_classes,
            device=device,
            half_precision=half_precision,
            logger=logger,
        )

        # pre-processing to apply to real data
        self.pre_process = torchvision.transforms.Normalize((0.5,), (0.5,))
        # post-processing to apply to generated data
        self.post_forward = Unnormalize((0.5,), (0.5,))

        # generation of input data
        self.generate_input = lambda batch_size: torch.randn(
            (batch_size, self.in_shape[0]), device=self.device
        )
        self.generate_labels = lambda batch_size: (
            torch.randint(self.n_classes, size=(batch_size,), device=self.device)
            if self.n_classes > 0
            else None
        )

        self._prepare()

    def forward(self, x, y=None):

        if self.n_classes and y is not None:
            x = append_one_hot_vector(x, y, self.n_classes)

        x = x.view(x.shape + (1, 1))

        # run through convolutional transpose layers
        x = super().forward(x)

        return x
