# -*- coding: utf-8 -*-
"""Fast Gradient Method (FGM) implementation.

This attack was originally implemented by Goodfellow et al. (2015) with the sup norm and
known as the "Fast Gradient Sign Method"). This implementation extends the attack to
norm 1 and 2, and is therefore called the Fast Gradient Method (FGM).


References
----------
Goodfellow, I. J.; Shlens, J. & Szegedy, C. "Explaining and Harnessing Adversarial
Examples" Proceedings of the International Conference on Learning Representations (ICLR)
2015


Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

from tqdm import tqdm
import torch as torch


from .base import BaseAttack
from ..optim import LIST_OPTIMIZERS, LIST_SCHEDULERS, DEFAULT_OPTIM_PARAMS


class PGMAttack(BaseAttack):
    """Projected Gradient Method attack.

    Parameters
    ----------
    model : TorchModel
        Model for classification.
    norm : { 1, 2, 'inf'}, optional
        Order of the norm, (default: 2).
    eps : float, optional
        Size of the attack.
    n_random_init : int, optional
        Number of random initialisations within the epsilon ball. If 0, start at the
        original input.
    with_minimal_perturbation : bool, optional
        Indicates whether the minimal perturbation is computed or not.
    """

    CV_A = 1 / 2 + 1e-7
    CV_B = 1

    def __init__(self, model, norm=2, optim_params=None):

        super().__init__(model)

        if optim_params is None:
            optim_params = DEFAULT_OPTIM_PARAMS

            # update default max iterations
            optim_params['max_iter'] = 1000
            optim_params['lr'] = 1e-2

        self.optim_params = optim_params

        if norm not in [1, 2, 'inf']:
            err_msg = 'Norm order must be either 1, 2, or inf (found: {}).'
            raise ValueError(err_msg.format(norm))

        self.norm = norm

    def generate(self, instances, tg_labels=None, tg_probas=None):
        """Generate adversarial examples.

        If the attack is targeted, the loss of the target label is minimized. If not,
        the loss of the predicted label is maximized.

        Parameters
        ----------
        instances : torch.Tensor
            Instances to be attacked.
        tg_labels : torch.Tensor or None, optional
            Target labels for instances, if targeted attack. Otherwise, None.
        tg_probas : torch.Tensor or None, optional
            Minimal probability value associated to each target class. Ignored for this
            attack.

        Returns
        -------
        torch.Tensor
            Batch of adversarial examples.

        Raises
        ------
        ValueError
            If the attack is set as targeted, but no target labels are provided.
        """

        # Check whether attack is targeted or not
        targeted = tg_labels is not None

        # get instance shape
        batch_size, n_channels, height, width = instances.shape

        if batch_size > 1:
            raise ValueError('Attack is currently working only on single image.')

        # copy instances
        adv_x = instances.clone().to(self.model.device).requires_grad_(True)

        # send data to device
        x = instances.clone().to(self.model.device)

        # possibly convert images to arctanh-space
        # y = torch.atanh((x) / self.CV_A - self.CV_B)
        y = x

        # define perturbations tensor as optimization variable
        perturbations = torch.zeros_like(x)
        perturbations.requires_grad = True

        # define optimizer
        optimizer = LIST_OPTIMIZERS[self.optim_params['optimizer_name']](
            [perturbations], **self.optim_params['optimizer_params']
        )

        if self.optim_params['lrscheduler_name'] is not None:
            lr_scheduler, lrs_update = LIST_SCHEDULERS[
                self.optim_params['lrscheduler_name']
            ]
            lr_scheduler = lr_scheduler(
                optimizer, **self.optim_params['lrscheduler_params']
            )
        else:
            lr_scheduler = None

        # define adversarial image
        m = torch.ones_like(perturbations)
        # m[:, :, 0:200, ::] = 0
        # adv_x = self.CV_A * (torch.tanh(y + perturbations * m) + self.CV_B)
        adv_x = x + perturbations

        # get the current prediction
        outputs = self.model(adv_x)

        # if not targeted, get the predicted label
        if not targeted:
            _, tg_labels = torch.max(outputs, 1)

        # progress bar
        pb_iterations = tqdm(
            range(self.optim_params['max_iter'])  # , disable=not self.verbose
        )

        for iteration in pb_iterations:

            # reset optimizer
            optimizer.zero_grad()

            loss = self.loss_function(outputs, tg_labels[0])
            loss.backward()
            optimizer.step()

            # update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(loss.detach().item())
                else:
                    lr_scheduler.step()

            # get the new adversarial image
            adv_x = self.CV_A * (torch.tanh(y + perturbations * m) + self.CV_B)

            # get the current prediction
            outputs = torch.softmax(self.model(adv_x), 1)

            # if not targeted, get the predicted label
            pred_probas, pred_labels = torch.max(outputs, 1)

            str_optimizer = '{:d} LR:{:.3g} Loss:{:.2f}'.format(
                iteration,
                optimizer.param_groups[0]['lr'],
                loss.detach().cpu().item(),
            )

            str_output = 'Pred:{:d}({:.2f}) - Tg:{:d}({:.2f})'.format(
                pred_labels[0],
                pred_probas[0],
                tg_labels[0].item(),
                outputs[0, tg_labels[0]],
            )
            self.desc = '{} {}'.format(str_optimizer, str_output)

            pb_iterations.desc = self.desc

            if pred_labels[0] == tg_labels[0]:
                break

        self._perturbations = perturbations.detach().cpu()

        return adv_x.detach().cpu()

    def loss_function(self, outputs, tg_label):
        return outputs[..., 1 - tg_label] - outputs[..., tg_label]
