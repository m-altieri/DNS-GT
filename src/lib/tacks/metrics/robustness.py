# -*- coding: utf-8 -*-
"""Robustness metrics.

Authors:  Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>
          Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import numpy as np
from tqdm.auto import tqdm

from .base import Meter
from .classification import ClassificationMeter


def compute_adversarial_robustness(attack, data_loader, verbose=True):
    """Get an empirical measure of robust accuracy for a given attack.

    The metric can be seen as an upper bound on the adversarial robustness/risk of the
    model. It is calculated for a specific model, on a specific data set with on type of
    attack.

    Parameters
    ----------
    model : TorchModel
        The model for which the robust accuracy should be estimated.
    data_loader: torch.utils.data.dataloader
        Loader for the dataset.
    attack: BaseAttack
        Attack used to estimate the robust accuracy.

    Returns
    -------
    float
        Accuracy over original instances.
    float
        Accuracy over adversarial instances.
    """
    n_instances = len(data_loader.dataset)
    n_classes = len(data_loader.dataset.classes)

    running_metrics = ClassificationMeter(n_classes=n_classes)
    running_adv_metrics = ClassificationMeter(n_classes=n_classes)

    pb_batches = tqdm(data_loader, disable=not verbose)
    for instances, gt_labels in pb_batches:

        # get predictions over instances
        outputs, (pred_labels, pred_probas) = attack.model.predict(instances)
        running_metrics(outputs, gt_labels)

        # generate the adversarial samples
        adv_instances = attack.generate(instances, tg_labels=None)

        # get predictions over adversarial instances
        adv_outputs, (pred_labels_adv, pred_probas_adv) = attack.model.predict(
            adv_instances
        )
        running_adv_metrics(adv_outputs, gt_labels)

    return running_metrics.get_accuracy(), running_adv_metrics.get_accuracy()


def get_robust_accuracy(attack_iterator, data_loader, strategy='worst', logger=None):
    """Get an empirical measure of robust accuracy for a given ensemble of attacks.

    Various ensemble strategies are implemented:
    TODO

    Parameters
    ----------
    attack_iterator : iterator
        Iterator returning an adversarial attack from an attack ensemble. The robust
        accuracy is estimated over all attacks.
    data_loader: torch.utils.data.dataloader
        Loader for the dataset.
    strategy : ['worst', 'mean', 'all'], optional
        Ensemble strategy (default: 'worst').
    logger : logging.Logger, optional
        Logging system (default: None).

    Returns
    -------
    accuracy, statistic(robust_accuracy), std(robust_accuracy)

    """

    accuracy = Meter(store=True)
    adv_accuracy = Meter(store=True)

    # loop over attacks and get the accuracy
    for attack in attack_iterator:

        logger.info('Computing adversarial robustness for attack %s.', str(attack))

        attack_accuracy, attack_adv_accuracy = compute_adversarial_robustness(
            attack, data_loader
        )

        accuracy(attack_accuracy)
        adv_accuracy(attack_adv_accuracy)

    logger.info('End of processing. Processing results...')

    if strategy == 'worst':
        return accuracy, (adv_accuracy.min, np.std(adv_accuracy.items))

    elif strategy == 'mean':

        return accuracy, (adv_accuracy.avg, np.std(adv_accuracy.items))

    elif strategy == 'all':
        return accuracy, adv_accuracy
