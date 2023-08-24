"""Definition of misc layers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import torch
import torch.nn as nn


class Reshape(nn.Module):
    """Reshape layer."""

    def __init__(self, shape):
        super().__init__()
        self.shape = list(shape)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape([batch_size] + self.shape)
        return x

    def __repr__(self):
        return 'Reshape(shape={})'.format(self.shape)


class Concat(nn.Module):
    """Concatenates a list of tensors along dimension.

    Parameters
    ----------
    dimension : int
        Dimension along which to concatenate.
    """

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)


class Flatten(nn.Module):
    """Flatten layer."""

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)
