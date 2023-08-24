# -*- coding: utf-8 -*-
"""Generation of archetypal examples.

References
----------

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch

from tacks.utils import LIST_OPTIMIZERS, LIST_SCHEDULERS


class ArchetypalExamples:
    """Generates an image that maximizes a certain label using gradient
    ascent."""
    def __init__(self, mh, lambda1=0.0, lambda2=0.0, ratio_change=1.0):
        self.mh = mh
        self.ratio_change = ratio_change
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def generate(self,
                 target_label,
                 init_instance,
                 n_epochs,
                 optimizer_name,
                 optimizer_params,
                 lrscheduler_name=None,
                 lrscheduler_params=None):
        """Generates class specific image.

        Parameters
        ----------
        target_label : int
            Target label for the archetypal example.
        init_instance : torch.Tensor
            Instance used at initialization.
        optimizer_name : str
            Optimizer used for training.
        optimizer_params : dict
            Parameters of the optimizer.
        lrscheduler_name : str, optional
            Name of the scheduler of the learning rate (default: None).
        lrscheduler_params : disc, optional
            Parameters of the scheduler of the learning rate (default: None).
        n_epochs : int, optional
            Number of epochs for the optimization.

        Returns
        -------
        torch.Tensor
        """
        self.mh.model.eval()

        # send instance on device for optimization
        init_instance = init_instance.to(self.mh.device)
        instance = init_instance.clone()
        instance.requires_grad = True

        # define the optimizer
        optimizer = LIST_OPTIMIZERS[optimizer_name]([instance],
                                                    **optimizer_params)

        lr_scheduler = LIST_SCHEDULERS[lrscheduler_name](
            optimizer, **
            lrscheduler_params) if lrscheduler_name is not None else None

        self.mh.logger.info(
            f'Start generation of an example for label {target_label}')

        pb_epochs = self.mh.tqdm_func(range(0, n_epochs))

        #  mask = torch.ones(instance.shape, dtype=torch.bool)
        #  a = torch.randint(low=0, high=instance.shape[2]-20, size=(1, )).item()
        #  b = torch.randint(low=0, high=instance.shape[3]-20, size=(1, )).item()
        #  mask[:, :, a:(a+20), b:(b+20)] = 0

        for epoch in pb_epochs:

            instance.requires_grad = True
            # reset gradients
            optimizer.zero_grad()

            # compute loss
            with torch.set_grad_enabled(True):
                output = torch.softmax(self.mh.model(instance), 1)

            instance_delta = instance - init_instance
            image_norm1 = self.lambda1 * instance_delta.abs().mean().item()
            image_norm2 = self.lambda2 * instance_delta.pow(2).mean().item()

            target_loss = (output.sum() - 2 *
                           output[0, target_label]) + image_norm1 + image_norm2
            target_loss.backward()

            #  mask = torch.rand(instance.shape) > self.ratio_change
            mask = instance.grad.abs() < 0.8 * instance.grad.abs().max()
            instance.grad[mask] = 0
            instance.grad[~mask] = torch.sign(instance.grad[~mask])
            #  min_grad = instance.grad[~mask].min()
            #  instance.grad *= (1./256) / 1e-3 / min_grad * (7*7)

            #
            # smooth grad
            weight = (torch.ones((1, 1, 9, 9))).float().to(instance.device)
            weight /= weight.sum()
            instance.grad[:] = torch.nn.functional.conv2d(instance.grad,
                                                          weight,
                                                          padding=4)

            optimizer.step()

            instance.requires_grad = False
            instance[:] = instance.clamp(init_instance.min(),
                                         init_instance.max())

            pb_epochs.write('{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(
                target_loss.item(), output[0, 0], output[0, 1], image_norm1,
                image_norm2))

            # update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(target_loss.item())
                else:
                    lr_scheduler.step()

        return instance.detach().cpu()
