# -*- coding: utf-8 -*-
"""Testing of CW attacks.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
        Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>
"""
import pytest

import time
import numpy as np

import torch

from art.attacks.evasion.carlini import CarliniL2Method
from art.estimators.classification import PyTorchClassifier

from tacks.archs.models import TorchModel
from tacks.attacks.cw import CWAttack
from tacks.data.mnist import load_dataset
from tacks.utils import get_config, Workspace


workspace = Workspace('Testing', instance_name='attacks.cw', args={'debug': False})


@pytest.fixture(scope='class')
def get_data(request):

    # get MNIST loaders
    train_dataset = load_dataset('train')
    test_dataset = load_dataset('test')

    in_shape = train_dataset.in_shape
    n_classes = len(train_dataset.classes)

    request.cls.n_classes = n_classes
    request.cls.in_shape = in_shape
    request.cls.train_dataset = train_dataset
    request.cls.test_dataset = test_dataset


@pytest.fixture(scope='class')
def get_model(request):

    # load simple MNISt model
    tacks_model = TorchModel.load_model(
        get_config().get_path('paths', 'models') / 'MNIST_lenet.pt'
    )
    tacks_model.clip_values = (0, 1)
    tacks_model.with_softmax = False
    tacks_model.logger = workspace.logger
    tacks_model.eval()

    # ART classifier
    art_classifier = PyTorchClassifier(
        tacks_model,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=tacks_model.clip_values,
        input_shape=tacks_model.in_shape,
        nb_classes=tacks_model.n_classes,
    )

    # store data in class
    request.cls.tacks_model = tacks_model
    request.cls.art_classifier = art_classifier


@pytest.mark.usefixtures('get_data')
@pytest.mark.usefixtures('get_model')
class TestCW:
    @pytest.mark.parametrize('tg_label', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None])
    # @pytest.mark.parametrize('tg_label', [0, None])
    # @pytest.mark.parametrize('delta', [0.1, 0.3, 0.5])
    @pytest.mark.parametrize('norm', [2])
    @pytest.mark.parametrize('batch_size', [1])
    def test_successful_attacks(self, batch_size, norm, tg_label):

        workspace.logger.info(
            'Batch size: %d | Norm: %s | TG Label: %s',
            batch_size,
            norm,
            tg_label,
        )
        
        # get a batch of samples
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        instances, labels = next(iter(train_loader))

        # get predictions on batch
        outputs, (pred_labels, pred_probas) = self.tacks_model.predict(instances)

        workspace.logger.info(
            'Groundtruth: %s - Prediction: %s (%s)',
            str(labels.tolist()),
            str(pred_labels.tolist()),
            str([f'{item:.2f}' for item in pred_probas.tolist()]),
        )

        # prepare inputs data for tacks
        if tg_label is not None:
            tacks_tg_labels = torch.ones(batch_size).long() * tg_label
        else:
            tacks_tg_labels = None

        # tg_probas = torch.ones(batch_size) * 0.25
        tacks_tg_probas = None

        # tacks attack
        workspace.logger.info('  tacks attack:')

        time_tacks1 = time.time()
        attack = CWAttack(self.tacks_model, norm=norm)
        tacks_adv_imgs = attack.generate(
            instances, tg_labels=tacks_tg_labels, tg_probas=tacks_tg_probas
        )
        time_tacks = time.time() - time_tacks1

        tacks_pert = tacks_adv_imgs.to(self.tacks_model.device) - \
            instances.to(self.tacks_model.device)

        tacks_outputs, (
            tacks_pred_labels,
            tacks_pred_probas,
        ) = self.tacks_model.predict(tacks_adv_imgs)

        workspace.logger.info(
            '    => %d (%.2f)',
            tacks_pred_labels[0].item(),
            tacks_pred_probas[0].item(),
        )

        # ART attack
        workspace.logger.info('ART attack:')
        art_attack = CarliniL2Method(
            self.art_classifier,
            batch_size=batch_size,
            targeted=tg_label is not None,
            max_iter=1000#0
        )

        if tg_label is not None:
            art_tg_labels = tacks_tg_labels.numpy()
            print('*****', art_tg_labels)
        else:
            art_tg_labels = None

        art_instances = instances.numpy()

        time_art1 = time.time()
        art_adv_imgs = torch.from_numpy(
            art_attack.generate(x=art_instances, y=art_tg_labels)
        )
        time_art = time.time() - time_art1
        art_pert = art_adv_imgs - instances

        # prediction
        art_outputs, (art_pred_labels, art_pred_probas) = self.tacks_model.predict(
            art_adv_imgs
        )

        workspace.logger.info(
            '    => %d (%.2f)',
            art_pred_labels[0].item(),
            art_pred_probas[0].item(),
        )

        # Check time difference
        workspace.logger.info(f'  Testing: Processing time by tacks: {time_tacks}' )
        workspace.logger.info(f'  Testing: Processing time by art: {time_art}')

        # testing
        workspace.logger.info('  Testing: mean abs value of adv example tacks' + 
                              str(np.mean(np.abs(tacks_adv_imgs.cpu().numpy()))))
        workspace.logger.info('  Testing: mean abs value of adv example Art' + 
                              str(np.mean(np.abs(art_adv_imgs.cpu().numpy()))))
        
        workspace.logger.info('  Testing: equality between adversarial images.')
        np.testing.assert_almost_equal(tacks_adv_imgs.cpu().numpy(),
                                       art_adv_imgs.cpu().numpy(), decimal=1)

        workspace.logger.info('  Testing: equality between perturbations.')
        np.testing.assert_almost_equal(tacks_pert.cpu().numpy(), art_pert.cpu().numpy(),
                                       decimal=1)
        
        
