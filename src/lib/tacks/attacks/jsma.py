# -*- coding: utf-8 -*-
"""Jacobian Saliency Map Attack (JSMA) implementation

References
----------
[PMJ+2016limitations] Papernot, N.; McDaniel, P.; Jha, S.; Fredrikson, M.; Celik, Z. B.;
Swami, A. `The limitations of deep learning in adversarial settings`

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import numpy as np
import torch as torch
from math import ceil
from tqdm.auto import tqdm

from .base import BaseAttack


class JSMAAttack(BaseAttack):
    """Jacobian Saliency Map Attack (JSMA).

    JSMA is a targeted attack that relies on the modification of the input instances
    using the Jacobian matrix of the outputs w.r.t. the inputs. At each iteration, the
    most significant features are changed in order to modify the outputs such that the
    input is misclassified.

    The choice of the relevant features is made using their contribution to the outputs,
    based on the gradient. Two quantities are calculated:
    (1). the contribution of features to the target outputs
    (2). the sum of contributions of features to the non targets outputs

    Two modes are considered:
    * *increasing*: the feature or group of features with a high positive value of (1)
    and a high negative value of (2) are increased.
    * *decreasing*: the feature or group of features with a high negative value of (1)
    and a high positive value of (2) are decreased.

    To do:
    [ ] Implement *decreasing mode*
    [ ] Test and comparison with `Adversarial Robustness Toolbox` (ART)


    Parameters
    ----------
    model, workspace
        See :class:`BaseAttack`.
    delta : float, optional
        Perturbation applied at each iteration (default: 0.1).
    max_mean_distortion : float, optional
        Maximum mean distortion of the inputs, between 0 and 1 (default: 0.1).
    n_retained_features : {1, 2}, optional
        Number of features retained to select the best features (default: 2).
    mode : {'increasing', 'decreasing'}
        Indicates if the features are decreased or increased (default: 'increasing').
    contribution_nontarget : {'all', 'prediction'} or None, optional
        Indicates how the contribution of non-target outputs is take into account in the
        selection of the features (default: 'all').
    on_logits : bool, optional
        Indicates if the Jacobian is computed over the logits or not (default: False).
    """

    def __init__(
        self,
        model,
        delta=0.1,
        max_mean_distortion=0.1,
        n_retained_features=2,
        mode='increasing',
        contribution_nontarget='all',
        on_logits=False,
        logger=None,
        verbose=True,
    ):

        super().__init__(model=model, logger=logger, verbose=verbose)

        self.delta = delta
        self.max_mean_distortion = max_mean_distortion
        self.n_retained_features = n_retained_features
        self.mode = mode
        self.contribution_nontarget = contribution_nontarget
        self.on_logits = on_logits

        # containers
        self._outputs = None

    def generate(self, instances, tg_labels, tg_probas=None):
        """Generate adversarial examples.

        Parameters
        ----------
        instances : torch.Tensor
            Instances to be attacked.
        tg_labels : torch.Tensor
            Target class for each instance.
        tg_probas : torch.Tensor or None, optional
            Minimal probability value associated to each target label. If None, no
            minimal probability is required (default: None).

        Returns
        -------
        torch.Tensor
            Adversarial instances.
        """

        # init variables for convenience
        n_classes = self.model.n_classes
        clip_min, clip_max = self.model.clip_values
        batch_size = instances.shape[0]
        n_features = torch.prod(torch.tensor(instances.shape[1::])).item()

        # set probability target to zero if not provided
        if tg_probas is None:
            tg_probas = torch.zeros(batch_size).float()

        # copy the input tensor
        X = instances.clone()

        # define perturbation
        perturbation = torch.zeros_like(X)

        # define adversarial instances
        adv_X = X + perturbation

        # compute an upper bound for the number of iterations
        n_iterations = ceil(n_features * (self.max_mean_distortion / self.delta + 1))

        # init container for saved outputs
        self._outputs = torch.zeros((n_iterations + 1, batch_size, n_classes))

        # get initial predictions
        outputs, (pred_labels, pred_probas) = self.model.predict(adv_X)
        self._outputs[0, ...] = outputs.clone()

        # define active features
        active_features = torch.ones(adv_X.shape, dtype=torch.bool)

        # define active instances
        active_instances = torch.any(active_features, 1) & (
            (tg_labels != pred_labels)
            | ((tg_labels == pred_labels) & (pred_probas < tg_probas))
        )

        # define the template for debug purposes
        template_debug = '{:04d}-{}#{:d} Tg:{} - Pr:{} ({:.3f})'

        # start iterative process
        pb_iterations = tqdm(total=n_iterations, disable=not self.verbose)
        iteration = 0

        for no_instance in range(batch_size):
            self.model.logger.debug(
                template_debug.format(
                    iteration,
                    'o' if active_instances[no_instance] else 'x',
                    no_instance,
                    tg_labels[no_instance],
                    pred_labels[no_instance],
                    pred_probas[no_instance],
                )
            )

        if self.n_retained_features == 2:
            pair_features = torch.triu(
                torch.ones((n_features, n_features), dtype=torch.bool), diagonal=1
            )

        while torch.any(active_instances):

            iteration += 1
            pb_iterations.update(1)

            # compute Jacobian matrix
            out_grad = torch.zeros((batch_size, n_classes, 2))
            out_grad[torch.arange(batch_size), tg_labels, 0] = 1

            if self.contribution_nontarget == 'all':
                # sum of gradient of the non-target output w.r.t. the inputs
                out_grad[:, :, 1] = 1
                out_grad[torch.arange(batch_size), tg_labels, 1] = 0
            elif self.contribution_nontarget == 'prediction':
                # sum of gradient of the current best output w.r.t. the inputs
                out_grad[torch.arange(batch_size), pred_labels, 1] = 1

            all_grads, _ = self.model.compute_gradients(
                (adv_X.view(adv_X.shape)[active_instances, ...]),
                grad_tensors=out_grad[active_instances, ...],
                on_logits=self.on_logits,
            )

            grads = torch.zeros(batch_size, n_features, 2)
            grads[active_instances, ...] = all_grads.view((-1, n_features, 2))
            grads[~active_features, :] = 0

            # if two features are retained, compute pairs of features
            if self.n_retained_features == 2:
                idx_pair_features = torch.where(pair_features)

            # sum of gradients for each pair of features
            if self.n_retained_features == 1:
                alphas = torch.maximum(grads[..., 0], torch.Tensor([0.0]))
                betas = torch.minimum(grads[..., 1], torch.Tensor([0.0]))
            elif self.n_retained_features == 2:
                alphas = torch.maximum(
                    grads[:, idx_pair_features, 0].sum(-1), torch.Tensor([0.0])
                )
                betas = torch.minimum(
                    grads[:, idx_pair_features, 1].sum(-1), torch.Tensor([0.0])
                )

            # get most significant pair of features
            ida = torch.argmax(-alphas * betas, 1)

            if self.n_retained_features == 1:
                feats = ida
            elif self.n_retained_features == 2:
                feats = idx_pair_features[ida]

            # add a perturbation of the given features
            idx_instances = torch.arange(batch_size)[active_instances].unsqueeze(1)
            if self.mode == 'increasing':
                adv_X[idx_instances, feats[active_instances]] += self.delta
            elif self.mode == 'decreasing':
                adv_X[idx_instances, feats[active_instances]] -= self.delta

            # if any of the entries is outside the value range, clip the entries and
            # deactivate the corresponding features
            if self.mode == 'increasing':
                if torch.any(adv_X[idx_instances, feats[active_instances]] >= clip_max):
                    adv_X[idx_instances, feats[active_instances]] = clip_max
                    active_features[idx_instances, feats[active_instances]] = 0

                    if self.n_retained_features == 2:
                        pair_features[~active_features, ~active_features] = 0
                        idx_pair_features = np.argwhere(pair_features)

            elif self.mode == 'decreasing':
                if torch.any(adv_X[idx_instances, feats[active_instances]] <= clip_min):
                    adv_X[idx_instances, feats[active_instances]] = clip_min
                    active_features[idx_instances, feats[active_instances]] = 0

                    if self.n_retained_features == 2:
                        pair_features[~active_features, ~active_features] = 0
                        idx_pair_features = np.argwhere(pair_features)

            # compute the mean distortion
            mean_distortion = (adv_X - adv_X.view(adv_X.shape)).mean(1)

            # compute the new prediction
            outputs, (pred_labels, pred_probas) = self.model.predict(
                adv_X.view(adv_X.shape)
            )
            self.saved_outputs[iteration, :] = outputs

            # update while condition
            active_instances = (
                torch.any(active_features, 1)
                & (mean_distortion < self.max_mean_distortion)
                & (
                    (tg_labels != pred_labels)
                    | ((tg_labels == pred_labels) & (pred_probas < tg_probas))
                )
            )

            for no_instance in range(batch_size):
                self.model.logger.debug(
                    template_debug.format(
                        iteration,
                        'A' if active_instances[no_instance] else '',
                        no_instance,
                        tg_labels[no_instance],
                        pred_labels[no_instance],
                        pred_probas[no_instance],
                    )
                )

        self.model.logger.info('End at iteration {}.'.format(iteration))
        self.model.logger.info(
            'Success rate: {:.1f}%'.format(
                (pred_labels == tg_labels).float().mean() * 100
            )
        )

        self.saved_outputs = self.saved_outputs[0 : (iteration + 1), ...].cpu()

        return perturbation
