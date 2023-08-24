# -*- coding: utf-8 -*-
"""Testing of FGM attacks.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import numpy as np
import pytest
import torch
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from art.estimators.classification import PyTorchClassifier
from tacks.archs.base import TorchModel
from tacks.attacks.fgm import BIMAttack, FGMAttack
from tacks.data.mnist import load_dataset
from tacks.utils import Workspace, get_config, get_logger, set_seed

TOL = 1e-8

workspace = Workspace('Testing', instance_name='attacks.fgm', args={'debug': False})
model_logger = get_logger('Testing | Model', to_file=False, to_console=False)


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
        get_config().get_path('paths', 'models') / 'MNIST_lenet_sgd_128_mean.pt'
    )
    tacks_model.clip_values = (0, 1)
    tacks_model.with_softmax = False
    tacks_model.logger = model_logger
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
class TestFGM:
    @pytest.mark.parametrize('seed', list(range(3)))
    @pytest.mark.parametrize('tg_label', list(range(10)) + [None])
    @pytest.mark.parametrize('perturbation_size', [0.0, 0.1, 0.3, 0.6])
    @pytest.mark.parametrize('norm', [1, 2, np.inf])
    @pytest.mark.parametrize('batch_size', [1])
    def test_successful_attacks(
        self, batch_size, norm, perturbation_size, tg_label, seed
    ):

        workspace.logger.info(
            'Batch size: %d | Norm: %s | Perturbation size: %.2f | Tg label: %s | Seed: %d',
            batch_size,
            norm,
            perturbation_size,
            tg_label,
            seed,
        )

        set_seed(seed)

        # get a batch of samples
        workspace.logger.info(
            'Get a batch of random samples over the training set of MNIST'
        )
        instances, labels = next(
            iter(
                torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=batch_size, shuffle=True
                )
            )
        )

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

        workspace.logger.info('Tacks attack:')

        # define tacks attack
        attack = FGMAttack(
            self.tacks_model,
            norm=norm,
            perturbation_size=perturbation_size,
            loss_function=torch.nn.CrossEntropyLoss(),
            tol=TOL,
            logger=workspace.logger,
            targeted=tg_label is not None,
        )
        tacks_adv_imgs = attack.generate(
            instances, tg_labels=tacks_tg_labels, tg_probas=tacks_tg_probas
        )

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
        art_attack = FastGradientMethod(
            self.art_classifier,
            norm=norm,
            batch_size=batch_size,
            minimal=False,
            eps=perturbation_size,
            targeted=tg_label is not None,
        )

        if tg_label is not None:
            art_tg_labels = tacks_tg_labels.numpy()
        else:
            art_tg_labels = tg_label

        art_instances = instances.numpy()

        art_adv_imgs = torch.from_numpy(
            art_attack.generate(x=art_instances, y=art_tg_labels)
        )

        # prediction
        art_outputs, (art_pred_labels, art_pred_probas) = self.tacks_model.predict(
            art_adv_imgs
        )

        workspace.logger.info(
            '    => %d (%.2f)',
            art_pred_labels[0].item(),
            art_pred_probas[0].item(),
        )

        # testing of outputs
        # NOTE: ART does a step even when pred_label == gt_label. Testing with ART is
        # then only done when it is not the case.
        is_same = pred_labels == tg_label

        # case
        workspace.logger.info('  Testing equality between adversarial images.')
        np.testing.assert_almost_equal(
            tacks_adv_imgs[~is_same, ...].numpy(),
            art_adv_imgs[~is_same, ...].numpy(),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            tacks_adv_imgs[is_same, ...].numpy(),
            instances[is_same, ...].numpy(),
            decimal=3,
        )

        workspace.logger.info('  Testing equality between prediction outputs.')
        np.testing.assert_almost_equal(
            tacks_outputs[~is_same, ...].numpy(),
            art_outputs[~is_same, ...].numpy(),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            tacks_outputs[is_same, ...].numpy(),
            outputs[is_same, ...].numpy(),
            decimal=3,
        )


@pytest.mark.usefixtures('get_data')
@pytest.mark.usefixtures('get_model')
class TestBIM:
    @pytest.mark.parametrize('seed', list(range(3)))
    @pytest.mark.parametrize('tg_label', list(range(10)) + [None])
    @pytest.mark.parametrize('perturbation_size', [0.0, 0.1, 0.3, 0.6])
    @pytest.mark.parametrize('norm', [1, 2, np.inf])
    @pytest.mark.parametrize('n_iterations', [1, 10])
    @pytest.mark.parametrize('batch_size', [1])
    def test_successful_attacks(
        self, batch_size, n_iterations, norm, perturbation_size, tg_label, seed
    ):

        step_size = 0.01

        workspace.logger.info(
            'Batch size: %d | Norm: %s | Perturbation_size: %f | TG Label: %s | Seed: %d',
            batch_size,
            norm,
            perturbation_size,
            tg_label,
            seed,
        )

        set_seed(seed)

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

        tacks_attack = BIMAttack(
            self.tacks_model,
            norm=norm,
            perturbation_size=perturbation_size,
            n_iterations=n_iterations,
            step_size=step_size,
            loss_function=torch.nn.CrossEntropyLoss(),
            tol=TOL,
            targeted=tg_label is not None,
            logger=workspace.logger,
        )
        tacks_adv_imgs = tacks_attack.generate(
            instances, tg_labels=tacks_tg_labels, tg_probas=tacks_tg_probas
        )

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
        art_attack = ProjectedGradientDescent(
            self.art_classifier,
            batch_size=batch_size,
            eps=perturbation_size,
            eps_step=step_size,
            targeted=tg_label is not None,
            max_iter=n_iterations,
            norm=norm,
        )

        if tg_label is not None:
            art_tg_labels = tacks_tg_labels.numpy()
        else:
            art_tg_labels = tg_label

        art_instances = instances.numpy()

        art_adv_imgs = torch.from_numpy(
            art_attack.generate(x=art_instances, y=art_tg_labels)
        )

        # prediction
        art_outputs, (art_pred_labels, art_pred_probas) = self.tacks_model.predict(
            art_adv_imgs
        )

        workspace.logger.info(
            '    => %d (%.2f)',
            art_pred_labels[0].item(),
            art_pred_probas[0].item(),
        )
        # testing of outputs
        # NOTE: ART does a step even when pred_label == gt_label. Testing with ART is
        # then only done when it is not the case.
        is_same = pred_labels == tg_label

        # case
        workspace.logger.info('  Testing equality between adversarial images.')
        np.testing.assert_almost_equal(
            tacks_adv_imgs[~is_same, ...].numpy(),
            art_adv_imgs[~is_same, ...].numpy(),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            tacks_adv_imgs[is_same, ...].numpy(),
            instances[is_same, ...].numpy(),
            decimal=3,
        )

        workspace.logger.info('  Testing equality between prediction outputs.')
        np.testing.assert_almost_equal(
            tacks_outputs[~is_same, ...].numpy(),
            art_outputs[~is_same, ...].numpy(),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            tacks_outputs[is_same, ...].numpy(),
            outputs[is_same, ...].numpy(),
            decimal=3,
        )
