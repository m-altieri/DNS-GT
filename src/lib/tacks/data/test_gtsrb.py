# -*- coding: utf-8 -*-
"""Testing of data.gtsrb

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""


import pytest

from tacks.data.gtsrb import GTSRBDataset
from tacks.utils import Workspace

workspace = Workspace("Testing", instance_name="models.base", args={"debug": False})


class TestGTSRBDataset:
    def test_init(self):
        
        dataset = GTSRBDataset('train')
