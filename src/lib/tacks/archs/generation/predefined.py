# -*- coding: utf-8
"""List of predefined architectures for generation.

References
----------
[Radfors2016Unsupervised] A. Radford, L. Metz, and S. Chintala, ‘Unsupervised
Representation Learning with Deep Convolutional Generative Adversarial Networks’,
preprint arXiv: 1511.06434, 2016.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

FC5 = (
    (
        ('flatten', {}),
        (
            'linear',
            {'out_features': 512},
        ),
    ),
    (
        (
            'linear',
            {
                'lin_params': {'out_features': 256},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'linear',
            {
                'lin_params': {'out_features': 512},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'linear',
            {
                'lin_params': {'out_features': 256},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'linear',
            {
                'lin_params': {'out_features': None},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        ('reshape', {}),
    ),
)


DCGAN = (
    (
        (
            'convtreg2d',
            {
                'convt_params': {
                    'out_channels': 256,
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                },
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'convtreg2d',
            {
                'convt_params': {
                    'out_channels': 128,
                    'kernel_size': (4, 4),
                    'stride': (1, 1),
                },
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'convtreg2d',
            {
                'convt_params': {
                    'out_channels': 64,
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                },
                'regularization': 'bn',
            },
        ),
    ),
    (
        (
            'convtreg2d',
            {
                'convt_params': {
                    'out_channels': 1,
                    'kernel_size': (4, 4),
                    'stride': (2, 2),
                },
                'regularization': 'bn',
                'activation': 'tanh',
            },
        ),
    ),
)
