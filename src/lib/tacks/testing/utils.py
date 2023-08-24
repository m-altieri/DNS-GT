# -*- coding: utf-8 -*-
"""Util functions for testing.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""


def generate_simple_cnn_architecture_classification(
    staged, conv_out_channels, conv_kernel_size, n_classes
):
    """Generate a simple CNN architecture for classification.

    The architecture is made of a convolutional layers with a maxpool followed by
    a feedforward layer.

    Parameters
    ----------
    staged : {'single', 'double'}
        Type of architecture.
    in_shape : tuple of ints
        Shape of inputs.
    out_shape : shape of outputs


    Returns
    -------
    dict
        Config for the architecture.
    """

    arch_cfg = (
        (
            (
                'convreg2d',
                {
                    #
                    'conv_params': {
                        'out_channels': conv_out_channels,
                        'kernel_size': conv_kernel_size,
                        'padding': True,
                        'stride': 1,
                    },
                    'regularization': 'bn',
                    'activation': 'relu',
                },
            ),
            ('maxpool2d', {'kernel_size': 2}),
        ),
        (
            ('flatten', {}),
            (
                'linreg',
                {
                    'lin_params': {'out_features': n_classes, 'bias': False},
                    'regularization': 'bn',
                    'activation': 'linear',
                },
            ),
        ),
    )

    if staged == 'single':
        arch_cfg = [layer for stage in arch_cfg for layer in stage]

    return arch_cfg
