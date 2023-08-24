# -*- coding: utf-8 -*-
"""Testing of models.classifiers

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pytest
from tacks.models import TorchModel
from tacks.data.mnist import load_dataset
from tacks.layers.utils import (get_out_shape_after_conv2d,
                                get_out_shape_after_maxpool2d)
from tacks.utils import Workspace, get_config, get_logger, set_seed

TOL = 1e-8

workspace = Workspace('Testing', instance_name='models.base', args={'debug': False})
model_logger = get_logger('Testing | Model', to_file=False, to_console=False)


class TestTorchModel:
    @pytest.mark.parametrize('conv_kernel_size', [3, 4, 5])
    @pytest.mark.parametrize('conv_out_channels', [3, 4, 5])
    @pytest.mark.parametrize('n_classes', [16, 32, 64])
    @pytest.mark.parametrize('in_size', [64, 100, 134])
    def test_init(self, in_size, out_size, conv_out_channels, conv_kernel_size):

        name = 'testing_classifier'
        in_shape = (3, in_size, in_size)
        out_shape = (out_size,)
        layers_cfg = (
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
                        'lin_params': {'out_features': out_size, 'bias': False},
                        'regularization': 'bn',
                        'activation': 'relu',
                    },
                ),
            ),
        )
        device = 'cpu'

        model = TorchModel(
            name=name,
            in_shape=in_shape,
            n_classes=n_classes,
            layers_cfg=layers_cfg,
            device=device,
            half_precision=False,
            logger=model_logger,
        )

        # convreg layer
        conv_n_parameters = (
            in_shape[0] * conv_out_channels * conv_kernel_size**2
            + 2 * conv_out_channels
        )

        # shape after conv and maxpool
        conv_out_shape = get_out_shape_after_conv2d(
            in_shape[1::], conv_kernel_size, stride=1, padding=True
        )
        maxpool_out_shape = get_out_shape_after_maxpool2d(conv_out_shape, 2)

        # linear layer
        flatten_shape = conv_out_channels * maxpool_out_shape[0] * maxpool_out_shape[1]
        lin_n_parameters = flatten_shape * out_size + 2 * out_size

        n_parameters = conv_n_parameters + lin_n_parameters

        assert model.n_parameters == n_parameters
        assert model.n_trainable_parameters == n_parameters

        model.freeze_all_layers()
        assert model.n_trainable_parameters == 0

