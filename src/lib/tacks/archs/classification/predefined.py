# -*- coding: utf-8
"""List of predefined architectures for classification.

References
----------
[LBD+1990Handwritten] LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. & others
`Gradient-based learning applied to document recognition`

[PMW+2016Distillation]Papernot, N.; McDaniel, P.; Wu, X.; Jha, S. & Swami, A.
`Distillation as a defense to adversarial perturbations against deep neural networks`

[ZNR2017Efficient] Zantedeschi, V.; Nicolae, M.-I. & Rawat, A. `Efficient defenses
against adversarial attacks`

[Myrtle] https://myrtle.ai/learn/how-to-train-your-resnet/

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

LENET = (
    # stage 1
    (
        # layer 1
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 6, 'kernel_size': 5, 'padding': 0},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        ('maxpool2d', {'kernel_size': 2}),
    ),
    # stage 2
    (
        # layer 1
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 16, 'kernel_size': 5, 'padding': 0},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    # stage 3
    (
        # layer 1
        ('flatten', {}),
    ),
    # stage 4
    (
        # layer 1
        (
            'linreg',
            {
                'lin_params': {'out_features': 256},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        (
            'linreg',
            {
                'lin_params': {'out_features': 256},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 3
        (
            'linreg',
            {
                'lin_params': {'out_features': None},
                'activation': 'linear',
            },
        ),
    ),
)

PAPERNOT = (
    # stage 1
    (
        # layer 1
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 3
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    # stage 2
    (
        # layer 1
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 3
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    # stage 3
    (
        # layer 1
        ('flatten', {}),
    ),
    # stage 4
    (
        # layer 1
        (
            'linreg',
            {
                'lin_params': {'out_features': 256},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        # layer 2
        ('linreg', {'lin_params': {'out_features': 256}, 'activation': 'relu'}),
        # layer 3
        ('linreg', {'lin_params': {'out_features': None}, 'activation': 'linear'}),
    ),
)

ZANTEDESCHI_CNN = (
    (
        'convreg2d',
        {
            'conv_params': {'out_channels': 64, 'kernel_size': 8},
            'regularization': 'bn',
            'activation': 'relu',
        },
    ),
    (
        'convreg2d',
        {
            'conv_params': {'out_channels': 128, 'kernel_size': 6},
            'regularization': 'bn',
            'activation': 'relu',
        },
    ),
    (
        'convreg2d',
        {
            'conv_params': {'out_channels': 128, 'kernel_size': 5},
            'regularization': 'bn',
            'activation': 'relu',
        },
    ),
    ('flatten', {}),
    ('linreg', {'lin_params': {'out_features': None}, 'activation': 'linear'}),
)

ZANTEDESCHI_RESNET = (
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 8},
                'regularization': 'bn',
                'activation': 'relu',
            },
        )
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 6},
                'regularization': 'bn',
                'activation': 'relu',
            },
        )
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 1},
                'regularization': 'bn',
                'activation': 'relu',
            },
        )
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 1},
                'regularization': 'bn',
                'activation': 'relu',
            },
        )
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 1},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 3}),
    ),
    (('flatten', {})),
    (('linreg', {'lin_params': {'out_features': None}, 'activation': 'linear'})),
)

MYRTLE = (
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 256, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 512, 'kernel_size': 3},
                'regularization': 'bn',
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (
        (
            'residual2d',
            {
                'conv_params': {'out_channels': 512, 'kernel_size': 3},
                'regularization': 'bn',
            },
        ),
        ('maxpool2d', {'kernel_size': 4, 'stride': 4}),
    ),
    (('flatten', {}),),
    (
        (
            'linreg',
            {'lin_params': {'out_features': None}, 'activation': 'linear'},
        ),
    ),
)


CW_CIFAR = (
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 128, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (('flatten', {}),),
    (
        (
            'linreg',
            {
                'lin_params': {'out_features': 256},
                'activation': 'relu',
                'regularization': 'dropout',
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': 256},
                'activation': 'relu',
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': None},
                'activation': 'relu',
            },
        ),
    ),
)


CW_MNIST = (
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 64, 'kernel_size': 3},
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2, 'stride': 2}),
    ),
    (('flatten', {})),
    (
        (
            'linreg',
            {
                'lin_params': {'out_features': 200},
                'regularization': 'dropout',
                'activation': 'relu',
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': 200},
                'activation': 'relu',
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': None},
                'activation': 'relu',
            },
        ),
    ),
)


# Architecture defined in https://github.com/ryan-feng/GRAPHITE/blob/main/GTSRB/GTSRBNet.py
# Initial task: Traffic sign recognition on GTSRB dataset
GTSRBNET = (
    # Stage 1
    (
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 5, 'padding': 2},
                'activation': 'relu',
            },
        ),
        (
            'convreg2d',
            {
                'conv_params': {'out_channels': 32, 'kernel_size': 5, 'padding': 2},
                'activation': 'relu',
            },
        ),
        ('maxpool2d', {'kernel_size': 2}),
        ('dropout2d', {'p': 0.5}),
    ),
    # Stage 2
    (
        (
            'convreg2d',
            {'conv_params': {'out_channels': 64, 'kernel_size': 5, 'padding': 2}},
        ),
        (
            'convreg2d',
            {'conv_params': {'out_channels': 64, 'kernel_size': 5, 'padding': 2}},
        ),
        ('maxpool2d', {'kernel_size': 2}),
        ('dropout', {'p': 0.5}),
    ),
    # Stage 3
    (
        (
            'convreg2d',
            {'conv_params': {'out_channels': 128, 'kernel_size': 5, 'padding': 2}},
        ),
        (
            'convreg2d',
            {'conv_params': {'out_channels': 128, 'kernel_size': 5, 'padding': 2}},
        ),
        ('maxpool2d', {'kernel_size': 2}),
        ('dropout', {'p': 0.5}),
    ),
    # Stage 4
    (('flatten', {})),
    # Stage 5
    (
        (
            'linreg',
            {
                'lin_params': {'out_features': 1024},
                'regularization': 'dropout',
                'reg_params': {'p': 0.5},
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': 1024},
                'regularization': 'dropout',
                'reg_params': {'p': 0.5},
            },
        ),
        (
            'linreg',
            {
                'lin_params': {'out_features': None},
            },
        ),
    ),
)
