# -*- coding: utf-8 -*-
"""Testing of models.utils

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pytest

import torch
from tacks.layers import (
    ConvReg2d,
    ConvTransposeReg2d,
    LinearReg,
    Reshape,
    Residual2dBlock,
)
from tacks.models.utils import get_correct_device
from tacks.utils import Workspace

TOL = 1e-8

workspace = Workspace('Testing', instance_name='models.utils', args={'debug': False})


class TestFuncs:
    def test_get_correct_device(self):
        cpu_device = torch.device('cpu')
        cuda_device = torch.device('cuda', index=0)

        default_device = cuda_device if torch.cuda.is_available() else cpu_device

        assert get_correct_device(None, workspace.logger) == default_device
        assert get_correct_device('cpu', workspace.logger) == cpu_device
        assert get_correct_device('cpu', workspace.logger) == cpu_device

        assert get_correct_device('cuda', workspace.logger) == cuda_device
        assert get_correct_device('cuda:0', workspace.logger) == cuda_device
        assert get_correct_device('cuda:1', workspace.logger) == cuda_device
