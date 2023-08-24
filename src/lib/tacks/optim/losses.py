# -*- coding: utf-8 -*-
"""Loss functions.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch


class FocalLoss(torch.nn.CrossEntropyLoss):
    """Focal loss."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction='mean',
    ):
        super(torch.nn.CrossEntropyLoss, self).__init__(
            weight, size_average, reduce, reduction
        )

    def forward(self, input, target):
        ce_loss = super().forward(input, target)

        pt = torch.exp(-ce_loss)
        alpha = 0.25
        gamma = 2
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()


class WassersteinLoss:
    """Wasserstein loss used to evaluate Critic and Generator models in GANs.

    When the loss is used to evaluate the Critic, a gradient penalty is computed on
    interpolation of training and generated instances (see [Gulrajani2017]).

    Parameters
    ----------
    disc_model : torchModule or None
        Discriminator (a.k.a. Critic) model.
    c_lambda : float
        Constant to control the importance of the gradient penalty.

    References
    ----------
    [Arjovsky2017] M. Arjovsky, S. Chintala, and L. Bottou, ‘Wasserstein Generative
    Adversarial Networks’, in Proceedings of the International Conference on Machine
    Learning, 2017, pp.  214–223

    [Gulrajani2017] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A.
    Courville, ‘Improved training of Wasserstein GANs’, in Proceedings of the 31st
    International Conference on Neural Information Processing Systems, 2017, pp.
    5769–5779.

    """

    name = 'WassersteinLoss'

    def __init__(self, disc_model=None, c_lambda=None):

        self.disc_model = disc_model
        self.c_lambda = c_lambda

    def __call__(
        self,
        pred,
        labels,
        train_instances=None,
        gen_instances=None,
        train_labels=None,
        gen_labels=None,
    ):

        loss = torch.nan_to_num(pred[labels == 0].mean()) - torch.nan_to_num(
            pred[labels == 1].mean()
        )

        # compute gradient penalty
        if self.c_lambda is not None and self.c_lambda > 0.0:

            epsilon = torch.rand(
                len(train_instances),
                1,
                1,
                1,
                device=train_instances.device,
                requires_grad=True,
            )
            mixed_instances = train_instances * epsilon + gen_instances * (1 - epsilon)
            mixed_labels = train_labels
            gradient = self.get_gradient(mixed_instances, mixed_labels)
            gradient = gradient.reshape((gradient.shape[0], -1))
            gradient = gradient.norm(2, dim=1)

            penalty = ((1 - gradient) ** 2).mean()

            loss += self.c_lambda * penalty
        return loss

    def get_gradient(self, instances, labels):

        # run instances on the instances
        disc_scores = self.disc_model(instances, labels)

        # compute the gradient with respect to images
        gradient = torch.autograd.grad(
            inputs=instances,
            outputs=disc_scores,
            grad_outputs=torch.ones_like(disc_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        return gradient
