# -*- coding: utf-8 -*-
"""External code from 'github/ryan-feng' and bridge function for testing purposes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch


# add github repo in path
ORIG_GRAPHITE_PATH = (Path('~') / 'repos' / 'GRAPHITE').expanduser()


if not ORIG_GRAPHITE_PATH.exists():
    err_msg = "Missing repo. Please clone 'ryan-feng/GRAPHITE' as {:s}"
    raise FileNotFoundError(err_msg.format(str(ORIG_GRAPHITE_PATH)))


def get_external_functions(graphite_path=None):
    if graphite_path is None:
        graphite_path = ORIG_GRAPHITE_PATH

    sys.path.append(str(graphite_path))

    import transforms  # noqa: E402

    get_transform_params = transforms.get_transform_params
    transform_wb = transforms.transform_wb

    sys.path.remove(str(graphite_path))

    return get_transform_params, transform_wb


class LegacyGaussianBlur(torch.nn.Module):
    def __init__(self, mask, kernel_size):
        super().__init__()

        self.mask = mask
        self.kernel_size = kernel_size

        if self.kernel_size > 0:
            kernel = np.zeros((self.kernel_size * 2 - 1, self.kernel_size * 2 - 1))
            kernel[self.kernel_size - 1, self.kernel_size - 1] = 1
            kernel = cv2.GaussianBlur(kernel, (self.kernel_size, self.kernel_size), 0)
            kernel = kernel[
                self.kernel_size // 2 : self.kernel_size // 2 + self.kernel_size,
                self.kernel_size // 2 : self.kernel_size // 2 + self.kernel_size,
            ]
            kernel = kernel[np.newaxis, :, :]
            kernel = np.repeat(kernel[np.newaxis, :, :, :], 3, axis=0)
            self.kernel_torch = torch.from_numpy(kernel)

    def forward(self, x):
        if self.kernel_size > 0:
            y = torch.where(self.mask > 0.5, x, torch.zeros(x.size(), device=x.device))

            blur = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=self.kernel_size,
                groups=3,
                bias=False,
                padding=self.kernel_size // 2,
            )

            blur.weight.data = self.kernel_torch.to(x.dtype)
            blur.to(x.device)
            blur.weight.requires_grad = False

            x = torch.where(self.mask > 0.5, blur(y), x)

            return x

        else:
            return x


class LegacyPerspectiveTransform(torch.nn.Module):
    def __init__(
        self,
        img_size,
        angle,
        focal,
        dist,
        obj_width,
        crop_percent,
        crop_off_x,
        crop_off_y,
        gt_label,
        whitebox=False,
        graphite_path=None,
    ):
        super().__init__()

        def dist2pixels(dist, width, obj_width=30):
            dist_inches = dist * 12
            return 1.0 * dist_inches * width / obj_width

        if graphite_path is None:
            graphite_path = ORIG_GRAPHITE_PATH

        self.graphite_path = graphite_path

        self.angle = math.radians(angle)
        self.h, self.w = img_size
        self.f = dist2pixels(focal, self.h, obj_width)
        self.d = dist2pixels(dist, self.h, obj_width)
        self.crop_percent = crop_percent
        self.crop_off_x = crop_off_x
        self.crop_off_y = crop_off_y
        self.pt_file = (
            self.graphite_path / 'inputs' / 'GTSRB' / 'Points' / f'{gt_label}.csv'
        )
        self.whitebox = whitebox

        x_cam_off = self.w / 2 - math.sin(self.angle) * self.d
        z_cam_off = -math.cos(self.angle) * self.d
        y_cam_off = self.h / 2

        R = np.array(
            [
                [math.cos(self.angle), 0, -math.sin(self.angle), 0],
                [0, 1, 0, 0],
                [math.sin(self.angle), 0, math.cos(self.angle), 0],
                [0, 0, 0, 1],
            ]
        )
        C = np.array(
            [
                [1, 0, 0, -x_cam_off],
                [0, 1, 0, -y_cam_off],
                [0, 0, 1, -z_cam_off],
                [0, 0, 0, 1],
            ]
        )

        RT = np.matmul(R, C)

        H = np.array(
            [
                [self.f * RT[0, 0], self.f * RT[0, 1], self.f * RT[0, 3]],
                [self.f * RT[1, 0], self.f * RT[1, 1], self.f * RT[1, 3]],
                [RT[2, 0], RT[2, 1], RT[2, 3]],
            ]
        )

        self.H = H

        x_off, y_off, crop_size = self.get_offset_and_crop_size(ratio=self.f / self.d)

        M_aff = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]])
        M = np.matmul(M_aff, H)

        if self.h > self.w:  # tall and narrow
            self.crop_x = crop_size
            self.crop_y = int(round(crop_size / self.w * self.h))
        else:  # wide and short or square
            self.crop_y = crop_size
            self.crop_x = int(round(crop_size / self.h * self.w))

        self.M = M

    def get_offset_and_crop_size(self, ratio):
        pts = []
        if self.pt_file is not None and self.pt_file != '':
            with open(self.pt_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    pts.append(
                        np.array([[float(row[0])], [float(row[1])], [float(row[2])]])
                    )

            for i in range(len(pts)):
                pts[i][0, 0] *= self.w / pts[i][2, 0]
                pts[i][1, 0] *= self.h / pts[i][2, 0]
                pts[i][2, 0] *= 1.0 / pts[i][2, 0]

        else:
            pts = [
                np.array([[0], [0], [1.0]]),
                np.array([[0], [self.h], [1.0]]),
                np.array([[self.w], [0], [1.0]]),
                np.array([[self.w], [self.h], [1.0]]),
            ]

        min_x = self.w
        min_y = self.h
        max_x = 0
        max_y = 0

        for pt in pts:
            new_pt = np.matmul(self.H, pt)
            new_pt /= new_pt[2, 0]

            if new_pt[0, 0] < min_x:
                min_x = new_pt[0, 0]
            if new_pt[0, 0] > max_x:
                max_x = new_pt[0, 0]
            if new_pt[1, 0] < min_y:
                min_y = new_pt[1, 0]
            if new_pt[1, 0] > max_y:
                max_y = new_pt[1, 0]

        if self.pt_file is not None and self.pt_file != '':
            if (max_x - min_x) / (
                max_y - min_y
            ) < self.w / self.h:  # result is tall and narrow
                diff_in_size = (max_y - min_y) / self.h * self.w - (max_x - min_x)
                orig_size = (
                    max_y - min_y
                    if self.w > self.h
                    else (max_y - min_y) / self.h * self.w
                )
                crop_size = int(round(orig_size * (1.0 - self.crop_percent)))
                y_off = -min_y - int(round(self.crop_percent / 2 * orig_size))
                x_off = -min_x + int(
                    round(diff_in_size / 2 - self.crop_percent / 2 * orig_size)
                )

            else:  # result is wide and short
                diff_in_size = (max_x - min_x) / self.w * self.h - (max_y - min_y)
                orig_size = (
                    max_x - min_x
                    if self.h > self.w
                    else (max_x - min_x) / self.w * self.h
                )
                crop_size = int(round(orig_size * (1.0 - self.crop_percent)))
                x_off = -min_x - int(round(self.crop_percent / 2 * orig_size))
                y_off = -min_y + int(
                    round(diff_in_size / 2 - self.crop_percent / 2 * orig_size)
                )

            return (
                x_off + self.crop_off_x * crop_size,
                y_off + self.crop_off_y * crop_size,
                crop_size,
            )

        else:
            min_x -= (self.w * ratio - (max_x - min_x)) // 2
            min_y -= (self.h * ratio - (max_y - min_y)) // 2

            crop_size = int(
                round((1.0 - self.crop_percent) * min(self.w, self.h) * ratio)
            )

            return (
                -min_x - int(round(self.crop_percent / 2 * self.w * ratio)),
                -min_y - int(round(self.crop_percent / 2 * self.h * ratio)),
                crop_size,
            )

    def forward(self, x):
        if not self.whitebox:
            dst = cv2.warpPerspective(
                x, self.M, (self.crop_x, self.crop_y), borderMode=cv2.BORDER_REPLICATE
            )
        else:
            dst = kornia.geometry.transform.warp_perspective(
                x,
                torch.from_numpy(self.M).float().to(x.device).unsqueeze(0),
                (self.crop_y, self.crop_x),
                align_corners=True,
                padding_mode='border',
            )

        return dst


def get_gtsrbnet_model(graphite_path=None):
    """Get the GTSRBNet model with last checkpoint."""

    if graphite_path is None:
        graphite_path = ORIG_GRAPHITE_PATH

    sys.path.append(str(graphite_path))

    from GTSRB.GTSRBNet import GTSRBNet  # noqa: E402

    model = GTSRBNet()

    # load checkpoint
    checkpoint = torch.load(graphite_path / 'GTSRB' / 'checkpoint_us.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    sys.path.remove(str(graphite_path))
    return model


def run_graphite_attack(params, graphite_path=None):
    """Run the GRAPHITE attack.

    Parameters
    ----------
    params : dict
        Parameters of the attack. See github/ryan-feng for details.

    Returns
    -------
    dict
        Outputs of the attack.

    Notes
    -----
    The script assumes a CUDA device is available.
    """

    if graphite_path is None:
        graphite_path = ORIG_GRAPHITE_PATH

    sys.path.append(str(graphite_path))

    script_path = graphite_path / 'whitebox' / 'whitebox_attack.py'

    old_cwd = os.getcwd()
    os.chdir(str(script_path.parent))

    # params['iters'] = '10'
    argv = [script_path.name, '--out', '/tmp']
    [
        argv.extend(
            [
                f'--{key}',
                str(value) if not isinstance(value, bool) else ('Y' if value else 'N'),
            ]
        )
        for key, value in params.items()
    ]

    # run graphite whitebox attack
    outs = {'__name__': '__main__'}
    old_argv = sys.argv

    sys.argv = argv

    with open(script_path) as infile:
        script = infile.read()
    exec(script, outs)
    sys.argv = old_argv

    torch.cuda.empty_cache()

    # revert previous working directory
    os.chdir(old_cwd)

    sys.path.remove(str(graphite_path))

    return {
        'adv_img': outs['prev_attack'].cpu(),
        'mask': outs['previous_mask'].cpu(),
        'eot_robustness': outs['prev_transform_robustness'],
        'n_queries': outs['num_queries'],
        'debugger': outs['debugger'] if 'debugger' in outs else None,
    }
