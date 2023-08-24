# -*- coding: utf-8 -*-
"""Testing of models.base

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pytest
import torch

from tacks.layers.utils import get_out_shape_after_conv2d, get_out_shape_after_maxpool2d
from tacks.models.base import TacksModel
from tacks.testing.utils import generate_simple_cnn_architecture_classification
from tacks.utils import Workspace, get_logger


TOL = 1e-7

workspace = Workspace('Testing', instance_name='models.base', args={'debug': False})
model_logger = get_logger('Testing | Model', to_file=False, to_console=False)


class TestTacksModel:
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('n_hidden_features', [16, 32, 64])
    @pytest.mark.parametrize('out_size', [16, 32, 64])
    @pytest.mark.parametrize('in_size', [64, 100, 134])
    def test_simple_model(self, device, n_hidden_features, out_size, in_size):
        name = 'lin_model'
        in_shape = (in_size,)
        out_shape = (out_size,)

        arch_config = (
            (
                (
                    'linreg',
                    {
                        'lin_params': {'out_features': n_hidden_features},
                        'regularization': 'bn',
                        'activation': 'relu',
                    },
                ),
            ),
            (
                (
                    'linreg',
                    {
                        'lin_params': {'out_features': out_size},
                        'regularization': 'bn',
                        'activation': 'linear',
                    },
                ),
            ),
        )

        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('No GPU available.')

        model = TacksModel(
            name=name,
            in_shape=in_shape,
            out_shape=out_shape,
            device=device,
            half_precision=False,
            logger=model_logger,
        )

        model.build_layers_from_arch_config(arch_config)

        n_parameters = (
            in_size * n_hidden_features
            + n_hidden_features * 2
            + n_hidden_features * out_size
            + out_size * 2
        )

        assert model.n_parameters == n_parameters
        assert model.n_trainable_parameters == n_parameters

        assert model._modules['layer0'][0].lin.weight.device == model.device
        assert model._modules['layer1'][0].lin.weight.device == model.device

        model.freeze_all_layers()
        assert model.n_parameters == n_parameters
        assert model.n_trainable_parameters == 0
