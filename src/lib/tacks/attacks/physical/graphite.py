# -*- coding: utf-8 -*-
"""Implementation of the GRAPHITE attack.

Original implementation comes from 'github/ryan-feng' for reproducibility purposes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import cv2
import kornia
import numpy as np
import torch
import torchvision
from tqdm.auto import tqdm

from .transforms import GammaTransform
from ..base import BaseAttack
from ..utils import Patch
from ...external.graphite_bridge import (
    LegacyGaussianBlur,
    LegacyPerspectiveTransform,
)


class GRAPHITEWhiteBoxAttack(BaseAttack):
    """Implementation of the GRAPHITE WhiteBox attack.

    Includes a original mode that reproduce as closely as possible the behaviour of the
    original GRAPHITE attack.

    Parameters
    ----------
    model, tol, logger, verbose
        See :class:`BaseAttack`.
    original : bool, optional
        Indicates if Legacy node is on (default: False).
    """

    def __init__(
        self,
        model,
        tol=1e-7,
        logger=None,
        verbose=True,
        original=False,
        broken=False,
        debugger=None,
    ):
        super().__init__(model, tol=1e-7, logger=logger, verbose=verbose)
        self.original = original
        self.broken = broken
        self._debugger = debugger

    def generate(
        self,
        instances,
        tg_labels=None,
        tg_probas=None,
        mask=None,
        attack_params=None,
        optim_params=None,
    ):
        # init mask as the whole image if missing
        if isinstance(mask, Patch):
            patch = mask
            mask = torch.tile(mask.mask, (3, 1, 1))
        else:
            patch = None
            mask = mask

        # get settings
        gt_label = attack_params['gt_label']
        aligned = attack_params['aligned']
        n_transforms = attack_params['n_transforms']
        min_eot_threshold = attack_params['min_eot_threshold']

        step_size = optim_params['step_size']
        n_iterations = optim_params['n_iterations']
        n_iterations_first_epoch = optim_params['n_iterations_first_epoch']
        n_max_epochs = optim_params['n_max_epochs']
        loss_function = optim_params['loss_function']

        # override parameters for original
        if self.original:
            n_max_epochs = None
            n_iterations_first_epoch = 500

        # define patch_params
        patch_params = {
            'n_patches': attack_params['n_patches'],
            'patch_removal_size': (
                attack_params['patch_removal_size']
                if attack_params['patch_removal_size'] >= 0
                else (30 if not aligned else 30.5)
            ),
            'patch_removal_interval': attack_params['patch_removal_interval'],
        }

        # generate transform params
        self.generate_transform_params(n_transforms)

        # move to device
        mask = mask.to(self.model.device)
        tg_labels = tg_labels.to(self.model.device)
        instances = instances.to(self.model.device)

        # initialize variables
        self.n_queries = 0
        epoch = 0

        prev_eot_robustness = None
        prev_mask_size = None

        prev_adv_img = instances[0, ...].clone()

        while n_max_epochs is None or epoch < n_max_epochs:
            # generate random image
            if self.original:
                init_img = torch.FloatTensor(3, 32, 32).uniform_(-8 / 255, 8 / 255)
            else:
                init_img = torch.rand((3, 32, 32)) * (16 / 255) - (8 / 255)

            # resize image to attack size
            np_init_img = init_img.permute(1, 2, 0).numpy()
            np_init_img = cv2.resize(np_init_img, (244, 244))
            init_img = (
                torch.from_numpy(np_init_img).permute(2, 0, 1).to(self.model.device)
            )

            # generate the adversarial image
            adv_img = instances[0, ...].clone()
            if self.original:
                adv_img = torch.where(mask > 0.5, prev_adv_img, instances[0, ...])
            else:
                adv_img = torch.where(mask > 0.5, 0, instances[0, ...])
            adv_img = adv_img + mask * init_img
            adv_img = torch.clamp(adv_img, 0, 1)
            adv_img.requires_grad = True

            # for a fixed mask, get a patch
            self._run_eot_attack(
                adv_img,
                mask,
                gt_label,
                tg_labels,
                n_iterations if epoch > 0 else n_iterations_first_epoch,
                loss_function,
                step_size,
                min_eot_threshold,
            )

            # collect results of the EOT attack
            avg_grad = self.__avg_grad
            adv_img = self.__adv_img
            eot_robustness = self.__eot_robustness

            if eot_robustness < min_eot_threshold:
                self.logger.warn('Threshold not reached.')

                if epoch == 0:
                    self.logger.info('Failed attack.')
                else:
                    self.logger.info('Eot robustness: %.2f', prev_eot_robustness)
                    self.logger.info('Mask size: %s', prev_mask_size)
                    self.logger.info('Number of queries: %d', self.n_queries)
                    self.logger.info('Number of epochs: %d', epoch)
                break

            # store outputs for last successful attack
            prev_adv_img = adv_img.detach().clone()
            prev_eot_robustness = eot_robustness
            prev_mask = mask.clone().cpu()
            prev_mask_size = torch.where(prev_mask > 0.5, 1, 0).sum().item() / 3

            pert = adv_img - instances[0, ...]
            avg_grad = mask * avg_grad * pert
            px_avg_grads = torch.sum(torch.abs(avg_grad), dim=0)

            # prune the mask
            self._prune_mask(mask, px_avg_grads, patch_params)

            if self.broken:
                prev_mask = mask.clone().cpu()
                prev_mask_size = torch.where(prev_mask > 0.5, 1, 0).sum().item() / 3

            epoch += 1

        if epoch == 0:
            adv_img = None
            self.mask = None
            self.eot_robustness = None
            self.patch = None
        else:
            adv_img = prev_adv_img.cpu()
            self.mask = torch.where(prev_mask > 0.5, 1, 0).bool()
            self.eot_robustness = prev_eot_robustness

            # update patch
            self.patch = (
                Patch(
                    adv_img.shape[1::],
                    size_mm=None,
                    cell_size=1,
                    is_transparent=False,
                    is_bw=False,
                    logger=self.logger,
                )
                if patch is None
                else patch
            )

            self.patch.mask = self.mask[0, ...]
            self.patch.values = adv_img

        return adv_img

    def _run_eot_attack(
        self,
        adv_img,
        mask,
        gt_label,
        tg_labels,
        n_iterations,
        loss_function,
        step_size,
        min_eot_threshold,
    ):
        """Run the EOT adversarial attack with current mask."""

        n_transforms = len(self.tparams_list)
        pb_iterations = tqdm(range(n_iterations), disable=not self.verbose)

        for iteration in pb_iterations:
            # container for average gradient
            avg_grad = torch.zeros(adv_img.size()).to(self.model.device)

            for tparams in self.tparams_list:
                # apply the transformation on the image
                transform_img = self._apply_transformation(
                    imgs=adv_img.clone().unsqueeze(0),
                    mask=mask,
                    transform_params=tparams,
                    gt_label=gt_label if self.original else None,
                )

                transform_img = transform_img - 0.5

                logits = self.model(transform_img.to(self.model.device))
                self.n_queries += 1

                loss = loss_function(logits, tg_labels)
                grad = torch.autograd.grad(loss, adv_img)[0]

                avg_grad += grad / n_transforms

            # set nan values to zero
            avg_grad[torch.isnan(avg_grad)] = 0

            # set inf values to 0
            if not self.original:
                avg_grad[torch.isinf(avg_grad)] = 0

            # take the sign of grad, without or with resizirg
            avg_grad_sign = avg_grad.sign()

            adv_img = adv_img - (mask == 1) * step_size * avg_grad_sign
            adv_img = adv_img.clamp(0, 1)

            eot_robustness = self.compute_eot_robustness(
                adv_img.unsqueeze(0), mask, gt_label, tg_labels[0]
            )
            pb_iterations.desc = 'EoT robustness: {:d}% - Mask size: {:d}'.format(
                int(eot_robustness * 100), mask.sum().int().item()
            )

            if eot_robustness >= min_eot_threshold:
                self.logger.info('Threshold reached: leaving loop.')
                break

        # store outputs
        self.__avg_grad = avg_grad
        self.__eot_robustness = eot_robustness
        self.__adv_img = adv_img

    def compute_eot_robustness(self, x, mask, gt_label, tg_label):
        """Compute the robustness over transformations.

        Parameters
        ----------
        orig_x : torch.Tensor

        x : torch.Tensor

        mask : torch.Tensor

        gt_label : int
            Ground truth label.
        tg_label : int
            Target label.
        """
        n_successes = 0
        preds = []

        n_transforms = len(self.tparams_list)

        x = x.to(self.device)
        mask = mask.to(self.device)

        for tparams in self.tparams_list:
            with torch.no_grad():
                if len(x.shape) == 3:
                    x = x.unsqueeze(0)

                transformed_x = self._apply_transformation(
                    imgs=x,
                    mask=mask,
                    transform_params=tparams,
                    gt_label=gt_label if self.original else None,
                )

                transformed_x = transformed_x - 0.5

                logits = self.model(transformed_x)
                self.n_queries += 1
                preds.append(
                    torch.nn.functional.softmax(logits, dim=1)[0, tg_label].item()
                )
                success = int(
                    logits.argmax(dim=1).detach().cpu().numpy()[0] == tg_label
                )
                n_successes += success
            # if self._debugger:
            # self._debugger.trace(
            #     '/',
            #     (
            #         transformed_x.mean().item(),
            #         logits.mean().item(),
            #         success,
            #         n_successes,
            #     ),
            # )

        return n_successes / n_transforms

    def _prune_mask(self, mask, px_avg_grads, patch_params):
        """Prune the mask to remove least significant regions.

        Least significant patches are removed, according to parameters given in
        :param:`patch_params`.

        Parameters
        ----------
        mask : torch.Tensor
            Current mask.
        px_avg_grads : torch.Tensor
            Average gradients per pixel.
        patch_params : dict
            Parameters of patch removal.
        """

        n_patches = patch_params['n_patches']
        patch_removal_size = patch_params['patch_removal_size']
        patch_removal_interval = patch_params['patch_removal_interval']

        _, img_height, img_width = mask.shape

        self.logger.info('Removing %d patches...', n_patches)

        for idp in range(n_patches):
            min_patch_grad = None
            min_patch_grad_idx = None

            i_range = np.arange(
                0, img_height - patch_removal_size + 0.0001, patch_removal_interval
            )
            j_range = np.arange(
                0, img_width - patch_removal_size + 0.0001, patch_removal_interval
            )

            for i in i_range:
                for j in j_range:
                    i_patch_slice = slice(
                        int(round(i)), int(round(i + patch_removal_size))
                    )
                    j_patch_slice = slice(
                        int(round(j)), int(round(j + patch_removal_size))
                    )

                    n_masked = mask[0, i_patch_slice, j_patch_slice].sum()

                    if n_masked > 0:
                        avg_patch_grad = (
                            px_avg_grads[i_patch_slice, j_patch_slice].sum() / n_masked
                        ).item()

                        # Note: avg_patch_grad may be equal to inf
                        if (
                            min_patch_grad_idx is None
                            or avg_patch_grad < min_patch_grad
                        ):
                            min_patch_grad = avg_patch_grad
                            min_patch_grad_idx = (i, j)

            if min_patch_grad_idx is None:
                break

            min_i, min_j = min_patch_grad_idx
            i_patch_slice = slice(
                int(round(min_i)), int(round(min_i + patch_removal_size))
            )
            j_patch_slice = slice(
                int(round(min_j)), int(round(min_j + patch_removal_size))
            )
            mask[0, i_patch_slice, j_patch_slice] = 0
            mask[1, i_patch_slice, j_patch_slice] = 0
            mask[2, i_patch_slice, j_patch_slice] = 0

            self.logger.debug('Removing patch #%d: (%d, %d)', idp, min_i, min_j)

    def _apply_transformation(self, imgs, mask, transform_params, gt_label=None):
        """Apply the transformation.

        Parameters
        ----------
        imgs : torch.tensor
            Images in the format BCHW.

        """

        # get parameters
        if self.original:
            (
                angle,
                dist,
                gamma,
                blur_kernel_size,
                crop_percent,
                crop_off_x,
                crop_off_y,
                obj_width,
                focal,
                _,
            ) = transform_params
        else:
            blur_kernel_size, angle, distortion_scale, gamma = transform_params

        transform = []

        # blur
        if self.original:
            transform.append(LegacyGaussianBlur(mask, blur_kernel_size))
        else:
            if blur_kernel_size > 0:
                transform.append(
                    kornia.augmentation.RandomGaussianBlur(
                        (blur_kernel_size, blur_kernel_size), (1.0, 1.0), p=1.0
                    )
                )

        # perspective
        if self.original:
            transform.append(
                LegacyPerspectiveTransform(
                    imgs.size()[2:4],
                    angle,
                    focal,
                    dist,
                    obj_width,
                    crop_percent,
                    crop_off_x,
                    crop_off_y,
                    gt_label,
                    whitebox=True,
                )
            )
        else:
            transform.append(
                torchvision.transforms.Compose(
                    [
                        kornia.augmentation.RandomRotation(np.abs(angle), p=1.0),
                        kornia.augmentation.RandomPerspective(
                            distortion_scale=0.2, p=1.0
                        ),
                        kornia.augmentation.RandomResizedCrop(
                            size=imgs.shape[2:4], p=1.0
                        ),
                    ]
                )
            )

        # gamma
        transform.append(GammaTransform(gamma_min=gamma, gamma_max=gamma, flip_p=0.0))

        # resizing
        transform.append(
            kornia.augmentation.Resize(32, align_corners=False),
        )

        # create transform object
        transform = torchvision.transforms.Compose(transform)

        return transform(imgs)

    def generate_transform_params(
        self,
        n_transforms,
        max_dist=15,
        min_angle=-50,
        max_angle=50,
        blur_kernel_sizes=(0, 3, 5, 7),
        nps=False,
    ):
        """Get parameters for the transformations.

        Order matters for reproducibility purposes.
        """

        self.tparams_list = []

        for _ in range(n_transforms):
            if self.original:
                angle = np.random.uniform(min_angle, max_angle)
                dist = np.random.uniform(3.0, max_dist)
                gamma = np.random.uniform(1.0, 3.5)
                flip_flag = int(np.round(np.random.uniform(0.0, 1.0)))

                crop_percent = np.random.uniform(-0.03125, 0.03125)
                crop_off_x = np.random.uniform(-0.03125, 0.03125)
                crop_off_y = np.random.uniform(-0.03125, 0.03125)

                blur_kernel_size = blur_kernel_sizes[
                    int(np.floor(np.random.uniform(0.0, 1.0) * len(blur_kernel_sizes)))
                ]

                if flip_flag == 1:
                    gamma = 1.0 / gamma

                params = (
                    angle,
                    dist,
                    gamma,
                    blur_kernel_size,
                    crop_percent,
                    crop_off_x,
                    crop_off_y,
                    30,
                    3,
                    nps,
                )
            else:
                blur_kernel_size = blur_kernel_sizes[
                    int(np.floor(np.random.uniform(0.0, 1.0) * len(blur_kernel_sizes)))
                ]
                angle = np.random.uniform(min_angle, max_angle)
                distortion_scale = np.random.uniform(1.0)

                gamma = np.random.uniform(1.0, 3.5)
                flip_flag = int(np.round(np.random.uniform(0.0, 1.0)))

                params = (blur_kernel_size, angle, distortion_scale, gamma)

            # add to list
            self.tparams_list.append(params)
