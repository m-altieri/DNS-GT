# -*- coding: utf-8 -*-
"""Carlini-Wagner Attack (CWA) implementation.

References
----------
[Carlini2017evaluating] N. Carlini and D. Wagner, ‘Towards evaluating the robustness of
neural networks’, presented at the IEEE symposium on security and privacy (SP), 2017.

Original CW implementation:
https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    
IBM Adversarial Robustness Toolbox implementation:
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/
    evasion/carlini.py

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
Author: Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>

TODOS:
    - define labels outside of methods with no hard coded part connected to
    CIFAR labels
    - implement other norm losses than l2
    - implement other objective functions than f6 in CW to check their claims
    - implement batch processing
    - explore other optimization/search strategies for c than the CW binary search 
"""

from tqdm import tqdm
import torch as torch


from .base import BaseAttack
from .utils import ToATanhSpace
from ..optim import LIST_OPTIMIZERS, LIST_SCHEDULERS, DEFAULT_OPTIM_PARAMS


class CWAttack(BaseAttack):
    """CW attacks object.

    Parameters
    ----------
    model : TorchModel
        Model for classification.
    norm : { 1, 2, 'inf'}, optional
        Order of the norm, (default: 2).
    c : float
        CW constant for defining the right target loss rate.
    """

    def __init__(
        self,
        model,
        norm=2,
        c=0.1,
        c_lower_bound_init=0.0,
        c_upper_bound_init=100,
        logit_confidence=0,
        c_search_steps=10,
        optim_params=None,
        logger=None,
        verbose=True,
    ):

        super().__init__(model, logger=logger, verbose=verbose)

        if optim_params is None:
            optim_params = DEFAULT_OPTIM_PARAMS

            # update default max iterations
            optim_params['max_iter'] = 1000#0

        self.optim_params = optim_params

        if norm not in [1, 2, 'inf']:
            err_msg = 'Norm order must be either 1, 2, or inf (found: {}).'
            raise ValueError(err_msg.format(norm))

        self.norm = norm
        self.c_init = c
        self.c_lower_bound_init = c_lower_bound_init
        self.c_upper_bound_init = c_upper_bound_init
        self.c_search_steps = c_search_steps
        self.logit_confidence = logit_confidence

    def loss_function(self, x, x_adv, outputs, tg_label, c, targeted):

        # TODO: define labels outside of methods with no hard coded part connected to
        # CIFAR labels
        labels = torch.arange(10).to(x.device)

        # TODO: other norm losses
        # norm distance loss image to adversarial image
        if self.norm == 2:
            # 2-norm is standard result
            normdist_loss = torch.linalg.norm(x_adv - x)
        else:
            raise NotImplementedError()

        # target loss according to CW
        Z_other = torch.max(outputs[..., labels != tg_label])
        Z_t = outputs[..., labels == tg_label]

        # TODO: other objective functions
        # loss adapted according to ART implementation, makes more sense, see
        # ART implementation L2 attack, line 161-165
        if targeted:
            f6_loss = torch.maximum(
                Z_other - Z_t + self.logit_confidence, torch.Tensor([0]).to(x.device)
            )
        else:
            f6_loss = torch.maximum(
                Z_t - Z_other + self.logit_confidence, torch.Tensor([0]).to(x.device)
            )

        return normdist_loss + c * f6_loss

    def generate(self, instances, tg_labels=None, tg_probas=None):
        """Generate adversarial examples.

        If the attack is targeted, the loss of the target label is minimized.
        If not, the loss of the predicted label is maximized.

        Parameters
        ----------
        instances : torch.Tensor
            Instances to be attacked.
        tg_labels : torch.Tensor or None, optional
            Target labels for instances, if targeted attack. Otherwise, None.
        tg_probas : torch.Tensor or None, optional
            Minimal probability value associated to each target class. Ignored
            for this attack.

        Returns
        -------
        torch.Tensor
            Batch of adversarial examples.

        Raises
        ------
        ValueError
            If the attack is set as targeted, but no target labels are provided.
        """

        # init transform to atanh space
        to_atanh_space = ToATanhSpace()

        # Check whether attack is targeted or not
        targeted = tg_labels is not None

        # get instance shape
        batch_size, n_channels, height, width = instances.shape

        # TODO: batch processing
        if batch_size > 1:
            raise ValueError('Attack is currently working only on single image.')

        # # copy instances
        # adv_x = instances.clone().to(self.model.device).requires_grad_(True)

        # send data to device
        x = instances.clone().to(self.model.device)

        # convert images to tanh-space and define transformed variable y
        y = to_atanh_space(x)

        # define perturbations tensor as optimization variable
        perturbations = torch.zeros_like(x)
        perturbations.requires_grad = True

        # define optimizer
        optimizer = LIST_OPTIMIZERS[self.optim_params['optimizer_name']](
            [perturbations], **self.optim_params['optimizer_params']
        )

        if self.optim_params['lrscheduler_name'] is not None:
            lrscheduler_name = self.optim_params['lrscheduler_name']
            lr_scheduler = LIST_SCHEDULERS[lrscheduler_name][0](
                optimizer, **self.optim_params['lrscheduler_params']
            )
        else:
            lr_scheduler = None

        # define adversarial image x + delta
        adv_x = to_atanh_space.inverse_transform(y + perturbations)
        # adv_x = x + perturbations

        # get the current prediction
        outputs = self.model(adv_x)

        # if not targeted, get the most likely predicted label to maximize its loss
        if not targeted:
            _, tg_labels = torch.max(outputs, 1)

        # progress bar
        pb_iterations = tqdm(
            range(self.optim_params['max_iter'])  # , disable=not self.verbose
        )

        # Successful flag
        is_successful = False

        # Initiate trade-off constant between loss function terms
        c = self.c_init
        c_low = self.c_lower_bound_init
        c_up = self.c_upper_bound_init

        # Outer loop for the c optimization strategy from the CW paper
        # TODO: explore other optimization/search strategies
        for c_step in range(self.c_search_steps):

            for iteration in pb_iterations:

                # check if a batch of images is to be checked against a single target
                if outputs.shape[0] >= 1 and tg_labels.shape[0] == 1:
                    tg_labels = (
                        torch.ones(outputs.shape[0]).long().to(self.model.device)
                        * tg_labels[0]
                    )

                # reset optimizer
                optimizer.zero_grad()
                loss = self.loss_function(
                    x, adv_x, outputs, tg_labels, c, targeted=targeted
                )

                # do next step
                loss.backward()
                optimizer.step()

                # update learning rate
                if lr_scheduler is not None:
                    if isinstance(
                        lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        lr_scheduler.step(loss.detach().item())
                    else:
                        lr_scheduler.step()

                # get the new adversarial image
                adv_x = to_atanh_space.inverse_transform(y + perturbations)

                # get the current prediction
                outputs = self.model(adv_x)

                # get the currently predicted label
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

                # stop optimization once target has been reached
                if targeted:
                    if pred_labels[0] == tg_labels[0]:
                        is_successful = True
                        break
                else:
                    if pred_labels[0] != tg_labels[0]:
                        is_successful = True
                        break

            # binary search update constant c as defined by CW. See CW implementation l2
            # attack line 223 and following
            if is_successful:
                c_up = min(c_up, c)
                if c_up <= 1e9:
                    c = (c_up + c_low) / 2
            else:
                c_low = max(c_low, c)
                if c_up <= 1e9:
                    c = (c_up + c_low) / 2

            self._perturbations = perturbations.detach().cpu()

        return adv_x.detach()
