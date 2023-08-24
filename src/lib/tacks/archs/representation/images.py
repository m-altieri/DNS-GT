# -*- coding: utf-8
"""Implementation of AutoEncoder architectures.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import numpy as np
import torch.nn as nn

from tacks.archs.utils import get_out_shape_after_conv2d
from tacks.archs.layers import Reshape
from tacks.models import TorchModel


class AE(TorchModel):
    """Implementation of an Autoencoder for the MNIST dataset."""

    task = 'autoencoder'

    def __init__(self, logger=None):
        super().__init__(logger=logger)

        self.name = 'AE'

        # list of convolutional layers
        conv_layers = [
            {
                'in_channels': self.input_shape[0],
                'out_channels': 64,
                'kernel_size': 5,
                'stride': 2,
            },
            {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1},
            {'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1},
        ]

        #######################################################################
        # Encoder

        out_shape = self.input_shape[1::]
        for idl, conv_params in enumerate(conv_layers):

            conv_name = f'enc_conv2d_{idl}'
            self.add_module(
                conv_name,
                nn.Sequential(
                    nn.Conv2d(**conv_params),
                    nn.BatchNorm2d(conv_params['out_channels']),
                    nn.ReLU(),
                ),
            )
            out_shape = get_out_shape_after_conv2d(
                out_shape,
                kernel_size=conv_params['kernel_size'],
                stride=conv_params['stride'],
            )

        flattened_dim = np.prod(out_shape) * conv_layers[-1]['out_channels']

        self.add_module(
            'enc_fc_0',
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            ),
        )
        self.add_module('enc_fc_1', nn.Linear(512, 256))

        #######################################################################
        # Decoder
        self.add_module('dec_fc_0', nn.Linear(256, 512))
        self.add_module(
            'dec_fc_1',
            nn.Sequential(
                nn.Linear(512, flattened_dim),
                nn.BatchNorm1d(flattened_dim),
                nn.ReLU(),
                Reshape([conv_layers[-1]['out_channels']] + list(out_shape)),
            ),
        )

        for idl, conv_params in enumerate(conv_layers[::-1]):

            (conv_params['in_channels'], conv_params['out_channels']) = (
                conv_params['out_channels'],
                conv_params['in_channels'],
            )

            self.add_module(
                f'dec_convt2d_{idl}',
                nn.Sequential(
                    nn.ConvTranspose2d(**conv_params),
                    nn.BatchNorm2d(conv_params['out_channels']),
                    nn.ReLU(),
                ),
            )

        self.add_module('dec_ups_1', nn.Upsample(self.input_shape[1::]))

    def encoder(self, x):

        x = self.enc_conv2d_0(x)
        x = self.enc_conv2d_1(x)
        x = self.enc_conv2d_2(x)

        x = self.enc_fc_0(x)
        x = self.enc_fc_1(x)

        return x

    def decoder(self, x):

        x = self.dec_fc_0(x)
        x = self.dec_fc_1(x)

        x = self.dec_convt2d_0(x)
        x = self.dec_convt2d_1(x)
        x = self.dec_convt2d_2(x)
        x = self.dec_ups_1(x)

        return x

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
