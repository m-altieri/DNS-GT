# -*- coding: utf-8 -*-
"""Implementation of gradient sign attack methods.

Current attacks:
*   Fast Gradient Method (FGM)
*   Basic Iteration Method (BIM).

FGM attack was originally implemented by Goodfellow et al. (2015) with the sup norm and
known as the "Fast Gradient Sign Method"). This implementation extends the attack to

* attacks targeting a specific label (see [Kurakin2017Adversarial])
* extensions to norm 1 and 2 (from
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/fast_gradient.py)

References
----------

[Goodfellow2015Explaining]I. J. Goodfellow, J. Shlens, and C. Szegedy, ‘Explaining and
harnessing adversarial examples’, preprint arXiv:1412.6572v3, 2015. [Online]. Available:
https://arxiv.org/abs/1412.6572

[Kurakin2017Adversarial]A. Kurakin, I. J. Goodfellow, and S. Bengio, ‘Adversarial
Machine Learning at Scale’, presented at the International Conference on Learning
Representations (ICLR), 2017. [Online]. Available: https://arxiv.org/abs/1611.01236

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
Author: Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>

"""

import warnings

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAttack


class BIMAttack(BaseAttack):
    """Basic Iterative Method (BIM) gradient attack.

    The BIM attack consists in several iterations of the fast gradient attack, with a
    given step size.

    If the attack is targeted, the loss of the target label is minimized, i.e.,
    log(P(y_t|x) is maximized in the case of the cross-entropy loss.

    If the attack is not targeted, the loss of the predicted label is maximized.  i.e.,
    -log(P(y_t|x) is maximized in the case of the cross-entropy loss.

    Parameters
    ----------
    model, logger, tol, verbose
        See :class:`BaseAttack`.
    norm : { 1, 2, 'inf'}, optional
        Order of the norm (default: 'inf').
    perturbation_size : float, optional
        Size of the perturbation (default: 0.3).
    n_iterations: int, optional
        Maximum number of iterations (default: 50).
    step_size : float, optional
        Iterative step size (default: 0.01).
    loss_function : func or None, optional
        Loss function to use (default: nn.CrossEntropyLoss).
    targeted : bool, optional
        Indicates if the attack is targeted or not (default: False).
    batch_strategy : ['disjoint', 'joint'], optional
        Strategy to use for batch attack (default: 'disjoint').
    """

    prefix = 'BIM'

    def __init__(
        self,
        model,
        norm='inf',
        perturbation_size=0.3,
        step_size=0.01,
        n_iterations=50,
        loss_function=None,
        tol=1e-7,
        targeted=False,
        batch_strategy='disjoint',
        logger=None,
        verbose=True,
    ):

        super().__init__(model, tol=tol, logger=logger, verbose=verbose)

        if n_iterations < 1:
            err_msg = 'Number of iteratins should be greater than 0 (given: {}).'
            raise ValueError(err_msg.format(n_iterations))

        self.norm = norm
        self.perturbation_size = perturbation_size
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.targeted = targeted
        batch_strategy = batch_strategy

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        self.loss_function = loss_function

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        if norm not in [1, 2, 'inf', np.inf]:
            err_msg = '`norm` must be either 1, 2, or inf (current: {}).'
            raise ValueError(err_msg.format(norm))

        if norm == 'inf':
            norm = np.inf

        self._norm = norm

    @property
    def perturbation_size(self):
        return self._perturbation_size

    @perturbation_size.setter
    def perturbation_size(self, perturbation_size):
        if perturbation_size < 0:
            err_msg = (
                '`perturbation_size` must be greater than or equal to 0 (current: {})'
            )
            raise ValueError(err_msg.format(perturbation_size))

        self._perturbation_size = perturbation_size

    def generate(self, instances, tg_labels=None, tg_probas=None):
        """Generate adversarial examples.

        If the attack is targeted, the loss of the target label is minimized. If not,
        the loss of the predicted label is maximized.

        Parameters
        ----------
        instances : torch.Tensor
            Instances to be attacked.
        tg_labels : torch.Tensor or None, optional
            Target labels for instances, if targeted attack. Ignored if the attack is
            not targeted.
        tg_probas : torch.Tensor or float, optional
            Indicates the minimal probability value associated to the target class.
            Ignored for this attack.

        Returns
        -------
        torch.Tensor
            Batch of adversarial examples.

        Raises
        ------
        ValueError
            If the attack is set as targeted, but no target labels are provided.
        UserWarning
            If thi attack is set as untargeted, but target labels are provided
        """

        self.logger.info('%s | Generation of adversarial samples.', self.prefix)

        # if perturbation_size is 0, return the instances without changes
        if self.perturbation_size == 0:
            self.logger.debug(
                '%s | `perturbation_size` set to 0. Infeasible attack.',
                self.prefix,
            )
            return instances

        # process target labels
        if self.targeted:

            if tg_labels is None:
                raise ValueError(
                    'Target labels should be provided when targetted attack.'
                )

            # send target labels into device
            tg_labels = tg_labels.to(self.device)
        else:
            if tg_labels is not None:
                warnings.warn('Target labels ignored as attack is untargeted.')

        # get batch size
        batch_size = instances.shape[0]
        self.logger.debug(
            '%s | Found %d samples in the batch.', self.prefix, batch_size
        )

        if batch_size > 1:
            err_msg = 'Batch size larger that 1 are not yet supported.'
            raise ValueError(err_msg)

        # copy data and send to device
        x = instances.clone().to(self.device)
        x.requires_grad = True

        # if target probas are not defined, set them to zero
        if tg_probas is None:
            tg_probas = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        else:
            tg_probas = tg_probas.to(self.device)

        # init perturbations
        perturbations = torch.zeros_like(x, device=self.device)

        # get the current prediction
        outputs = self.model(x + perturbations)
        (pred_probas, pred_labels) = outputs.detach().softmax(1).max(1)

        self.logger.debug('Outputs: %s', outputs.tolist())
        self.logger.debug(
            '%s | Predicted labels: %s', self.prefix, str(pred_labels.tolist())
        )

        # if not targeted, get the predicted label to maximize its loss
        if self.targeted:
            self.logger.info(
                '%s | Targeted attack. Minimize the error on the target labels.',
                self.prefix,
            )

            self.logger.info(
                '%s | Target labels: %s', self.prefix, str(tg_labels.tolist())
            )
            self.logger.info(
                '%s | Target probas: %s', self.prefix, str(tg_probas.tolist())
            )

        else:
            _, tg_labels = torch.max(outputs, 1)

            self.logger.info(
                '%s | Untargeted attack. Maximize the error on the predicted labels.',
                self.prefix,
            )

        # init iteration
        iteration = 0

        # vector indicating which instances are still actively attacked
        active_instances = torch.ones(batch_size, dtype=torch.bool)

        while iteration < self.n_iterations:

            if self.targeted:
                # deactivate samples for which the predicted and target labels are the
                # same with a probability greater than target probabilities
                active_instances[
                    torch.logical_and(
                        pred_labels == tg_labels, pred_probas >= tg_probas
                    )
                ] = False

            else:
                # deactivate samples for which the predicted and target labels are
                # different with a probability greater than target probabilities
                active_instances[
                    torch.logical_and(
                        pred_labels != tg_labels, pred_probas >= tg_probas
                    )
                ] = False

            self.logger.info(
                '%s | Iteration %d - Number of active instances: %d',
                self.prefix,
                iteration,
                active_instances.sum().int().item(),
            )

            # if all instances are inactive, break the loop
            if not torch.any(active_instances):
                break

            self.logger.debug(
                '%s | Iteration %d - PL: %s (%s)',
                self.prefix,
                iteration,
                str(pred_labels.tolist()),
                str(pred_probas.tolist()),
            )

            # compute loss value
            loss = (-1) ** self.targeted * self.loss_function(outputs, tg_labels)
            self.logger.debug('%s | Loss: %.2e', self.prefix, loss.item())

            # zero the parameter gradients of the model
            self.model.zero_grad()

            # zero the gradients of the input variable
            if x.grad is not None:
                x.grad.zero_()

            # compute gradients of model in backward pass
            loss.backward()

            # compute additional perturbations based on gradients
            perturbations += self.step_size * self._compute_perturbation(x.grad.data)

            # get perturbations after clipping values on adversarial instances
            perturbations = (x.detach() + perturbations).clip(0, 1) - x.detach()

            # project perturbations in domain of validity
            if self.norm == np.inf:

                # clip perturbation
                perturbations = perturbations.clip(
                    -self.perturbation_size, self.perturbation_size
                )

            elif self.norm in [1, 2]:

                # compute factor to bound the norm of perturbation
                pert_norms = self._compute_norm(perturbations)

                factor = (
                    pert_norms > self.perturbation_size
                ) * self.perturbation_size / pert_norms + (
                    pert_norms <= self.perturbation_size
                ) * 1

                # # rescale perturbations
                perturbations = perturbations * factor

            self.logger.info('Mean perturbation: %.2e', perturbations.mean().item())

            # return predictions for adversarial instances
            adv_x = x + perturbations
            outputs = self.model(adv_x)
            (pred_probas, pred_labels) = outputs.detach().softmax(1).max(1)

            self.logger.debug('Outputs: %s', outputs.tolist())
            self.logger.debug('Mean adv instances: %.2e', adv_x.mean().item())

            # go to next iteration
            iteration += 1

        # store meaningful information
        self._info['last_iteration'] = iteration
        self._info['pred_labels'] = pred_labels.cpu().tolist()
        self._info['pred_probas'] = pred_probas.cpu().tolist()

        self._perturbations = perturbations.to(instances.device)

        return instances + self._perturbations

    def _compute_perturbation(self, gradients):
        """Compute the perturbation corresponding to the norm for each image of a batch.

        Parameters
        ----------
        gradients : torch.Tensor
            Gradients of the images.

        Returns
        -------
        torch.Tensor
        """

        self.logger.debug('Grad mean: %e', gradients.mean().item())

        if self.norm == np.inf:
            return gradients.sign()

        elif self.norm == 1:
            return gradients / (gradients.abs().sum([1, 2, 3], keepdim=True) + self.tol)

        elif self.norm == 2:
            return gradients / (
                torch.sqrt(torch.pow(gradients, 2).sum([1, 2, 3], keepdim=True))
                + self.tol
            )

    def _compute_norm(self, x):
        """Compute the average norm of instances from a given batch of instances.

        Parameters
        ----------
        x : torch.tensor
            Batch.

        Returns
        -------
        torch.Tensor
            Value of the norm for each instance of the batch.
        """
        x = x.reshape(x.shape[0], -1, 1, 1)
        return torch.linalg.norm(x, ord=self.norm, dim=1, keepdim=True) / x.shape[1]

    def __str__(self):

        params = '_'.join(
            [
                f'norm:{self.norm}',
                f'delta:{self.perturbation_size}',
                f'n_iterations:{self.n_iterations}',
            ]
        )

        return f'{self.prefix}_{params}'


class FGMAttack(BIMAttack):
    """One step Fast Gradient Method attack.

    A FGM attack is a single iteration of the BIM attack.

    Parameters
    ----------
    model : TorchModel
        Model for classification.
    """

    prefix = 'FGM'

    def __init__(
        self,
        model,
        norm='inf',
        perturbation_size=0.3,
        loss_function=None,
        tol=1e-7,
        targeted=False,
        logger=None,
        verbose=True,
    ):
        super().__init__(
            model,
            norm=norm,
            perturbation_size=perturbation_size,
            step_size=perturbation_size,
            n_iterations=1,
            loss_function=loss_function,
            tol=tol,
            targeted=targeted,
            logger=logger,
            verbose=verbose,
        )

    def __str__(self):

        params = '_'.join([f'norm:{self.norm}', f'delta:{self.perturbation_size}'])

        return f'{self.prefix}_{params}'
