# -*- coding: utf-8 -*-
"""Definition of routines for adversarial training.

Authors: Ronan Hamon <ronan.hamon@ec.europa.eu>
         Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>
"""

import random

import numpy as np
import torch


def get_adversarial_collate_fn(attack_iterator, ratio=0.05):
    """Get a collate function for adversarial training.

    The collate function is to be used in a DataLoader. For each batch, a given ratio of
    instances are attacked using a predefined list of attacks, and replaced by the
    adversarially altered instances. Attacks are done for each batch, using the current
    state of parameters of the model being trained.

    Parameters
    ----------
    attack_iterator : iterator
        Iterator returning an adversarial attack. The iterator is run at the beginning
        of each generation of adversarial example.
    adv_ratio : float, optional
        Ratio of instances to be replaced by adversarial examples (default: 0.05).

    Returns
    -------
    function
        Returns the modified collate function that is then given to a torch data loader.

    """

    def collate_fn(batch):
        """Collate function used in a torch.DataLoader.

        Parameters
        ----------
        batch : torch.tensor or numpy.array
            A batch processed by a torch data.loader object.

        Returns
        -------
        torch.tensor, torch.tensor
            Returns a tensor containing the properly transformed and stacked data and a
            second tensor with the properly transformed labels.

        """

        def get_adversarial_examples(instances):
            """Get an adversarial example from an instance.


            Parameters
            ----------
            instances : torch.tensor
                Instances contained in the batch.

            Returns
            -------
            img : torch.tensor
                Either the same image, or replaced by an adversarial example according
                to the ratio.

            """
            if random.random() < ratio:
                attack = next(attack_iterator)
                attack.model.eval()
                instances = attack.generate(instances.unsqueeze(0)).squeeze(0)
                attack.model.train()

            return instances

        imgs, labels = list(zip(*batch))
        imgs = torch.stack(list(map(get_adversarial_examples, imgs)))

        return imgs, torch.tensor(labels)

    return collate_fn
