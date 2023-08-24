# -*- coding: utf-8 -*-
"""Testing of models.base
Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import pytest
import torch

from tacks.attacks.base import BaseAttack

from tacks.classifiers import TacksClassifier

@pytest.fixture(scope='class')
def get_models(request):
    """Get pretrained models for testing."""

    tacks_model = TacksModel.load(
        tacks_config.get_path('paths', 'models') / 'GTSRB_gtsrbnet_checkpoint_us.pt',
        logger=workspace.logger,
    )
    tacks_model.clip_values = (0, 1)
    tacks_model.with_softmax = False
    tacks_model._debugger = None
    tacks_model.eval()

    orig_model = graphite_bridge.get_gtsrbnet_model()
    orig_model.to(tacks_model.device)
    orig_model.eval()

    request.cls.tacks_model = tacks_model
    request.cls.orig_model = orig_model

class TestAttack(BaseAttack):
    def generate(self, x, tg_labels=None, tg_probas=None):
        super().generate(x=x, tg_labels=tg_labels, tg_probas=tg_probas)
        return x


class TestTacksModel:
    @pytest.mark.parametrize('model', ['model1', 'model2'])
    def test_methods(self):
        attack = TestAttack()

        asse
