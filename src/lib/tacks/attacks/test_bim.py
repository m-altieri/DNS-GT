# -*- coding: utf-8 -*-
"""Testing of util functions and classes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pytest

import torch


from tacks.attacks.utils import ToATanhSpace
from tacks.utils import Workspace

workspace = Workspace(
    'Testing', subworkspace_name='attacks.utils', args={'debug': True}
)


class TestBIM:
    @pytest.mark.parametrize('n_samples', [10, 1000, 100000])
    @pytest.mark.parametrize('b', [0.5, 1, 2])
    @pytest.mark.parametrize('a', [0.5, 1, 2])
    def test_boundaries_vector(self, a, b, n_samples):

        bounds = [a * (b - 1), a * (b + 1)]

        workspace.logger.info('Generating data in %s', bounds)

        x = torch.linspace(bounds[0], bounds[1], n_samples)
        workspace.logger.info('x= %s', x)

        # init transform
        to_tanh_space = ToATanhSpace(a=a, b=b)

        y = to_tanh_space(x)
        workspace.logger.info('y= %s', y)

        z = to_tanh_space.inverse_transform(y)
        workspace.logger.info('z= %s', z)

        assert torch.all(z >= bounds[0])
        assert torch.all(z <= bounds[1])
        torch.testing.assert_allclose(x, z)

    @pytest.mark.parametrize('m', [10, 1000])
    @pytest.mark.parametrize('n', [10, 1000])
    @pytest.mark.parametrize('b', [0.5, 1, 2])
    @pytest.mark.parametrize('a', [0.5, 1, 2])
    def test_boundaries_matrix(self, a, b, n, m):

        bounds = [a * (b - 1), a * (b + 1)]

        workspace.logger.info('Generating data in %s', bounds)

        x = torch.rand((3, n, m)) * (bounds[1] - bounds[0]) + bounds[0]
        workspace.logger.info('x= %s', x)

        # init transform
        to_tanh_space = ToATanhSpace(a=a, b=b)

        y = to_tanh_space(x)
        workspace.logger.info('y= %s', y)

        z = to_tanh_space.inverse_transform(y)
        workspace.logger.info('z= %s', z)

        assert torch.all(z >= bounds[0])
        assert torch.all(z <= bounds[1])
        torch.testing.assert_allclose(x, z)
