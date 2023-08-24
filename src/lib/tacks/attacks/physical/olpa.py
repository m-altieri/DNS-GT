# -*- coding: utf-8 -*-
"""Physical adversarial attack classes and functions.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
from tqdm.auto import tqdm

from ...data.image import TorchImageDataset
from ...optim import LIST_OPTIMIZERS, LIST_SCHEDULERS
from ...utils.colors import generate_color_scheme
from ..base import BaseAttack


def compute_nps(img, colors):
    """Compute a score of color compliance.

    The score is based on the Non-Printability Score (NPS). It consists in computing the
    distance between the pixel of the image and a given list of colors. The smaller the
    score, the closer pixels of the images with one of the color.

    Parameters
    ----------
    img : torch.tensor
        Image in the format CWH, with RGB channels.
    colors : torch.tensor
        Tensor containing a list of colors.

    Returns
    -------
    torch.tensor
        NPS score.

    References
    ----------
    M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, ‘Accessorize to a Crime: Real
    and Stealthy Attacks on State-of-the-Art Face Recognition’, in Proceedings of the
    2016 ACM SIGSAC Conference on Computer and Communications Security, New York, NY,
         USA, 2016, pp. 1528–1540, doi: 10.1145/2976749.2978392.
    """

    if img.ndim != 3:
        err_msg = 'Image should be of of dimension 3 (found: {}).'
        raise ValueError(err_msg.format(img.ndim))

    if img.shape[0] != 3:
        err_msg = 'Number of channels should be 3 (found: {}).'
        raise ValueError(err_msg.format(img.shape[0]))

    n_colors = colors.shape[1]

    # get norm of the difference between pixels and colors
    x = torch.norm(img.view(3, -1).unsqueeze(2) - colors.unsqueeze(1), p=1.5, dim=0)

    # # get the product of norms over colors
    x = torch.prod(x, 1)

    # return the sum over pixels
    return torch.sum(x)


def compute_tv(img):
    """Total variation.

    This compute the total variation of a 2D image.

    Parameters
    ----------
    img : torch.tensor
        Image in the format CWH, with RGB channels.

    Returns
    -------
    torch.tensor
        Total variation.

    """
    tv_norm = torch.norm(img[:, 1::, :] - img[:, 0:-1, :], 2, dim=(1, 2)) + torch.norm(
        img[:, :, 1::] - img[:, :, 0:-1], 2, dim=(1, 2)
    )
    return torch.sum(tv_norm)


class OLPALoss:
    def __init__(self, lambda_nps, lambda_tv, tg_label, mask, colors, device):

        self.lambda_nps = lambda_nps
        self.lambda_tv = lambda_tv
        self.mask = mask
        self.tg_label = tg_label
        self.device = device

        if colors is None:
            colors = generate_color_scheme()

        self.colors = colors

        self._loss_function = torch.nn.CrossEntropyLoss()

    def __call__(self, outputs, adv_img, iteration):

        # self.adv_loss = outputs.sum() - 2 * outputs[..., self.tg_label].sum()
        self.adv_loss = self._loss_function(
            outputs,
            torch.ones(outputs.shape[0], dtype=torch.long, device=self.device)
            * self.tg_label,
        )

        patch = adv_img[0, ...] * self.mask

        self.nps_loss = compute_nps(patch, colors=self.colors)
        # self.tv_loss = compute_tv(patch)
        self.tv_loss = torch.tensor([0.0], device=self.device)

        if iteration > 0:
            self.loss = (
                self.adv_loss
                + self.lambda_nps * self.nps_loss
                + self.lambda_tv * self.tv_loss
            )
        else:
            self.loss = self.adv_loss

        return self.loss

    def __str__(self):

        return 'L:{:.2f} AL:{:.2f} NPS:{:.2f} TV:{:.2}'.format(
            self.loss.detach().cpu().item(),
            self.adv_loss.detach().cpu().item(),
            self.nps_loss.detach().cpu().item(),
            self.tv_loss.detach().cpu().item(),
        )


class OLPAttack(BaseAttack):
    def generate(
        self,
        instances,
        tg_labels,
        tg_probas=None,
        mask=None,
        attack_params=None,
        optim_params=None,
    ):

        batch_size, n_channels, height, width = instances.shape

        # get parameters for the attacks
        lambda_nps = attack_params['lambda_nps']
        lambda_tv = attack_params['lambda_tv']
        colors = attack_params['colors']
        transform = attack_params['transform']
        n_instances = attack_params['n_instances']

        # send colors to device
        if colors is not None:
            colors = colors.to(self.model.device)

        if mask is None:
            raise NotImplementedError

        if mask is not None:
            # send mask to device
            mask.to(self.model.device)

        # define loss function
        loss_function = OLPALoss(
            lambda_nps,
            lambda_tv,
            tg_labels[0],
            mask.mask,
            colors,
            self.model.device,
        )

        # copy instances
        x = instances.clone().to(self.model.device)

        # init perturbation
        perturbation = mask.atoms

        # randomly generate colors from set of admissible colors
        # perturbation += colors[
        #     :, choices(range(colors.shape[1]), k=perturbation.shape[0] // 3)
        # ].view(-1)

        # add gradient to tensor
        perturbation.requires_grad = True

        # define perturbation
        adv_x = x.clone()
        mask.apply_on_img(adv_x[0, ...])
        adv_x = adv_x.clip(*self.model.clip_values)

        # generate several transformations of the image
        adv_x = torch.vstack(
            [adv_x]
            + [
                next(
                    iter(
                        torch.utils.data.DataLoader(
                            TorchImageDataset(adv_x, transform),
                            batch_size=n_instances,
                        )
                    )
                )
                for _ in range(n_instances)
            ]
        )

        # get the current prediction
        outputs = self.model(adv_x)

        # get the predicted label
        pred_labels = torch.max(outputs, 1)

        # define optimizer
        optimizer = LIST_OPTIMIZERS[optim_params['optimizer_name']](
            [perturbation], **optim_params['optimizer_params']
        )

        # define scheduler
        if optim_params['lrscheduler_name'] is not None:

            lr_scheduler, _ = LIST_SCHEDULERS[optim_params['lrscheduler_name']]

            lr_scheduler = lr_scheduler(optimizer, **optim_params['lrscheduler_params'])

        else:
            lr_scheduler = None

        # progress bar
        pb_iterations = tqdm(range(optim_params['n_epochs']), disable=not self.verbose)

        for iteration in pb_iterations:

            # reset optimizer
            optimizer.zero_grad()
            loss = loss_function(outputs, adv_x, iteration)

            loss.backward()

            # replace gradients by signed gradients
            # perturbation.grad = perturbation.grad.sign()

            optimizer.step()

            # update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(loss.detach().item())
                else:
                    lr_scheduler.step()

            adv_x = x.clone()

            # add perturbations
            mask.apply_on_img(adv_x[0, ...])
            adv_x = adv_x.clip(*self.model.clip_values)

            # generate several transformations of the image
            adv_x = torch.vstack(
                [
                    next(
                        iter(
                            torch.utils.data.DataLoader(
                                TorchImageDataset(adv_x, transform),
                                batch_size=n_instances,
                            )
                        )
                    )
                    for _ in range(n_instances)
                ]
            )

            # if not targeted, get the predicted label
            outputs = self.model(adv_x)
            sm_outputs = torch.softmax(outputs.detach(), 1)
            pred_probas, pred_labels = torch.max(sm_outputs, 1)

            str_loss = str(loss_function)

            str_optimizer = '{:d} LR:{:.3g} {}'.format(
                iteration, optimizer.param_groups[0]['lr'], str_loss
            )

            str_output = 'Pred:{:d}({:.2f}) - Tg:{:d}({:.2f})'.format(
                pred_labels[0],
                pred_probas[0],
                tg_labels[0],
                sm_outputs[0, tg_labels[0]],
            )

            pb_iterations.desc = '{} {}'.format(str_optimizer, str_output)

            if all(
                [
                    pred_labels[idi] == tg_labels[0] and pred_probas[idi] > tg_probas[0]
                    for idi in range(x.shape[0])
                ]
            ):
                break

        # send back mask to cpu and detach atoms and clip them
        mask.atoms = mask.atoms.detach().clip(*self.model.clip_values)
        mask.to('cpu')

        # create final perturbed image
        adv_x = x.clone().cpu()
        mask.apply_on_img(adv_x[0, ...])

        # create black image with perturbation
        perturbation = torch.zeros((n_channels, width, height), device='cpu')
        mask.apply_on_img(perturbation)

        self._perturbation = perturbation

        return adv_x
