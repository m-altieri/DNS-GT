# -*- coding: utf-8
"""

References
----------

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
import torch.nn as nn

from .base import TorchModel

class Transformer(TorchModel):
    """Transformer architecture."""

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        pass

    def forward(self, x, y, x_mask, y_mask):

        x = self.encode(x, x_mask)
        x = self.decode(x, x_mask, y, y_mask)

        return x

    def encode(self, x, x_mask):

        x = self.encoder(self.x_)




class Encoder(nn.Module):
    """Encoder as a repetition of identical layers.

    Parameters
    ----------
    layer : nn.Module
        Layer to be repeated.
    n_repetitions : int
        Number of repetitions of the layer.

    """

    def __init__(self, layer, n_repetitions):

        super().__init__()
        self.layers = 



class EncoderLayer(nn.Module):
    """Encoder layer made of self-attention with feed-forward."""

    def __init__(self, size, self_attention, feed_):
        pass
