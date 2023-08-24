# -*- coding: utf-8
"""Implementation of the MalConv architecture.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""

import numpy as np
import torch
import torch.nn as nn

from ..base import TorchClassifier


class TransposeLastDims(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class MalConv(TorchClassifier):
    """Implementation of the Malconv architecture.

    Classify a PE executable given as a bytes sequence into as malware or
    goodware.

    References
    ----------
    [RBS+2018Malware] Raff, E.; Barker, J.; Sylvester, J.; Brandon, R.;
    Catanzaro, B.  & Nicholas, C. K.  `Malware detection by eating a whole EXE`

    """

    def __init__(
        self,
        layers_cfg,
        n_classes,
        with_softmax=False,
        device=None,
        half_precision=False,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            name='MalConv',
            in_shape=None,
            layers_cfg=layers_cfg,
            n_classes=2,
            with_softmax=with_softmax,
            device=device,
            half_precision=half_precision,
            logger=logger,
            **kwargs,
        )



        self.name = 'MalConv'
        self.input_shape = None
        self.n_classes = 2
        self.clip_values = None

        self.embd_dim = 8

        self.add_module(
            'embedding',
            nn.Sequential(
                nn.Embedding(257, self.embd_dim, padding_idx=0), TransposeLastDims()
            ),
        )

        self.add_module('conv1d_1', nn.Conv1d(self.embd_dim, 256, 512, stride=512))
        self.add_module(
            'conv1d_2', nn.Sequential(nn.Conv1d(8, 256, 512, stride=512), nn.Sigmoid())
        )
        self.add_module('maxpool', nn.AdaptiveMaxPool1d(1))

        self.add_module(
            'fc',
            nn.Sequential(
                nn.Flatten(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)
            ),
        )
        self.run_embedding = True

    def set_run_embedding(self, run_embedding):
        """Switch the model to use embedded features as inputs.

        Parameters
        ----------
        run_embedding : bool
            Indicates if the embedding is run or not.
        """
        self.run_embedding = run_embedding

    def forward(self, x):

        # embedding
        if self.run_embedding:
            x = self.embedding(x)

        # parallel convolutions and merge
        x1 = self.conv1d_1(x)
        x2 = self.conv1d_2(x)
        x = self.maxpool(x1 * x2)

        # fully connected
        x = self.fc(x)

        return x

    def find_closest_bytes(self, X):
        """Find the closest bytes for a given matrix of embedded vectors.

        Parameters
        ----------
        X : torch.Tensor
            8-dimensional vectors.

        Returns
        -------
        torch.Tensor
            Bytes between 0 and 256 (included).
        """
        embedding = self._modules['embedding'][0].weight.detach()
        X_tiled = X.unsqueeze(1).repeat_interleave(257, 1).to(embedding.device)

        return torch.norm(X_tiled - embedding, dim=2).argmin(1).long().cpu()
