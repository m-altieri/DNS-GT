# -*- coding: utf-8 -*-
"""Testing of layers.linear

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pytest
import torch

from tacks.layers.linear import LinearReg
from tacks.models.base import TacksModel
from tacks.testing.utils import generate_simple_cnn_architecture_classification
from tacks.utils import Workspace, get_logger


class TestLinearReg:
    def test_init(self):
        in_features = 32
        out_features = 128
        bias = True
        batch_size = 32

        lin_params = {
            'in_features': in_features,
            'out_features': out_features,
            'bias': bias,
        }

        linear_reg = LinearReg(lin_params=lin_params)

        x = torch.rand(batch_size, in_features)

        assert not linear_reg.fused
        assert linear_reg.in_features == in_features
        assert linear_reg.out_features == out_features
        assert linear_reg.regularization == 'none'

        # exceptions
        with pytest.raises(ValueError):
            LinearReg({'out_features': out_features, 'bias': bias})

        with pytest.raises(ValueError):
            LinearReg({'in_features': in_features, 'bias': bias})

        with pytest.raises(NotImplementedError):
            linear_reg.fused = True
            linear_reg(x)
