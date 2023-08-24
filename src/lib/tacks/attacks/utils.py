# -*- coding: utf-8 -*-
"""Util functions and classes for attacks.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
Author: Henrik Junklewitz <henrik.junklewitz@ec.europa.eu>
"""
import cv2
import numpy as np
from pathlib import Path


import torch
from PIL import Image

from ..utils import check_path, get_logger
from ..utils.image import permute_axes_img, torch_to_opencv_img

# filling colour
FILL_COLOR = (255, 255, 255)
UNFILL_COLOR = (0, 0, 0)

# 1 mm in inch
IN_MM = 0.03937008


def mm_to_in(x):
    return x * IN_MM


def in_to_mm(x):
    return x / IN_MM


def pt_to_in(x, dpi=72):
    return x / dpi


def in_to_pt(x, dpi=72):
    return x * dpi


def mm_to_pt(x, dpi=72):
    return in_to_pt(mm_to_in(x), dpi=dpi)


def pt_to_mm(x, dpi=72):
    return round(in_to_mm(pt_to_in(x, dpi=dpi)))


class ToATanhSpace:
    """Transformation to tanh space.

    The inverse Hyperbolic tangent is a monotonically increasing function from [-1, 1]
    to [-inf, +inf]. It is therefore used to project normalized input data into an
    unbounded space.

    It performs the following transform: let x \\in [0, 1]:
        y = atanh(x / a + b)
    leading to
        -inf < y < +inf.

    The inverse transformation is defined by:
        x = a (tanh(x) - b)

    Parameters 'a' and 'b' are used to scale the data to [-1, 1]:
    * 'a' is the ratio between the sizes of the input and output spaces.
    ' 'b' is the difference between the minimal value of the input space and -1.

    Parameters
    ----------
    a : float, optional
        Ratio between the difference between lower and upper bound of input space and 2
        (default: 0.5).
    b : float, optional
        Difference between the lower bound of input space and -1 (default: 1).

    References
    ----------
    [Carlini2017evaluating] N. Carlini and D. Wagner, ‘Towards evaluating the robustness
    of neural networks’, presented at the IEEE symposium on security and privacy (SP),
    2017.
    """

    def __init__(self, a=0.5, b=1):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        """Perform the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to transform.

        Returns
        -------
        torch.Tensor
        """
        y = x / self.a - self.b

        if not (torch.all(y >= -1) and torch.all(y <= 1)):
            err_msg = 'Data should be between -1 and 1.'
            raise ValueError(err_msg)
        return torch.atanh(y)

    def inverse_transform(self, y):
        """Invert the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Tensor to invert.

        Returns
        -------
        torch.Tensor
        """
        return self.a * (torch.tanh(y) + self.b)


class Patch:
    """Patch for RGB images.

    A patch applies to an image of fixed size. IT is made of:
        - a mask that determines which pixels of the image are covered by the patch
        - an underlying RGB image that gives the values of each pixel


    The underlying RGB image is made of cells whose size is given by `cell_size`.
    The number of channels of the image defines the colors of the patch:
        1. black and white;
        2. black and white + transparency
        3. RGB
        4. RGB + transparency


    At initialisation, mask is empty. The atoms are randomly generated from a uniform
    distribution between 0.1 and 0.9.

    Parameters
    ----------
    size_px : int or 2-tuple of ints
        Size of the mask.
    size_mm : float or 2-tuple of float or None, optional
        Size of the image in mm. Used for printing as pdf. If None, the size in pixels
        is used (default: None).
    cell_size : int, optional
        Size of a cell in pixel (default: 1).
    logger : logging.Logger or None
        Logging system.
    """

    def __init__(
        self,
        size_px,
        size_mm=None,
        cell_size=1,
        is_transparent=False,
        is_bw=False,
        logger=None,
    ):
        if isinstance(size_px, int):
            size_px = (size_px, size_px)

        if isinstance(cell_size, int):
            cell_size = (cell_size, cell_size)

        # if `size_mm` is not provided, 1px is equal to 1mm
        if size_mm is None:
            size_mm = size_px
        elif isinstance(size_mm, int):
            size_mm = (size_mm, size_mm)

        if logger is None:
            logger = get_logger('Patch', to_file=False, to_console=False)

        self.size_px = size_px
        self.size_mm = size_mm
        self.logger = logger

        self.cell_size = cell_size
        self.is_bw = is_bw
        self.is_transparent = is_transparent
        self.n_channels = 1 if is_bw else 3

        # check that cell size is compatible with mask size
        if any([self.size_px[idd] % self.cell_size[idd] for idd in range(2)]):
            err_msg = 'Mask size ({:s}) incompatible with cell size ({:s})'
            raise ValueError(err_msg)

        n_cells = np.prod(
            [self.size_px[idd] // self.cell_size[idd] for idd in range(2)]
        )

        self.mask = torch.zeros(size_px, dtype=torch.bool)
        self.cell_values = torch.rand(self.n_channels, n_cells)
        self.alpha = torch.rand(1) if self.is_transparent else None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask.shape != self.size_px:
            err_msg = 'Shape of init mask ({}) different from given size ({})'
            raise ValueError(err_msg.format(mask.shape, self.size_px))

        self._mask = mask

    @property
    def px_size(self):
        """Size of a pixel in mm."""
        return [
            self.cell_size[idd] * self.size_mm[idd] * IN_MM / self.size_px[idd]
            for idd in range(2)
        ]

    @property
    def size_pt(self):
        return tuple(map(lambda x: mm_to_pt(x), self.size_mm))

    @property
    def values(self):
        """Return the patch as an image."""

        cv = self.cell_values.clone()

        if self.is_bw:
            cv = cv.repeat((3, 1))

        cv = cv.repeat_interleave(self.cell_size[0], dim=1)
        cv = cv.reshape(3, -1, self.size_px[0])
        cv = cv.repeat_interleave(self.cell_size[1], dim=1)

        return cv

    @values.setter
    def values(self, values):
        # copy values
        values = values.clone()

        # clear unmasked elements
        values[:, ~self.mask] = 1.0

        # replace cell values
        self.cell_values = values.reshape(self.n_channels, -1)
        self.n_cells = (1, 1)

    @staticmethod
    def load(patch_path, logger=None):
        """Load a patch.

        Parameters
        ----------
        patch_path : str or pathlib.Path
            Path to the patch.
        """
        # convert into Path
        patch_path = Path(patch_path).expanduser()

        # check that path exists
        check_path(patch_path)

        # load the container
        np_patch = np.load(patch_path, allow_pickle=True)

        mask = np_patch['mask']
        cell_values = np_patch['cell_values']
        params = np_patch['params']

        patch = Patch(
            mask.shape,
            size_mm=None,
            cell_size=1,
            is_transparent=False,
            is_bw=False,
            logger=logger,
        )

        patch.mask = torch.from_numpy(mask)
        patch.cell_values = torch.from_numpy(cell_values)

        patch.size_mm = tuple(params[0, :].astype(int))
        patch.cell_size = tuple(params[1, :].astype(int))
        patch.alpha = float(params[2, 0])
        patch.is_bw = params[3, 0] == 1
        patch.is_transparent = params[3, 1] == 1

        return patch

    def save(self, patch_path):
        """Save the patch.

        Use numpy to save with compression.

        Parameters
        ----------
        patch_path : str or pathlib.Path
            Path to the patch.

        """
        # convert into Path
        patch_path = Path(patch_path).expanduser()

        # check that parent path exists
        check_path(patch_path.parent)

        params = np.zeros((4, 2))
        params[0, :] = self.size_mm
        params[1, :] = self.cell_size
        params[2, 0] = self.alpha
        params[3, :] = self.is_bw, self.is_transparent

        np.savez_compressed(
            patch_path,
            mask=self.mask,
            cell_values=self.cell_values,
            params=params,
        )

        self.logger.info(f'Patch saved at {patch_path}')

    def add_patch(self, patch, negative=False):
        """Add a patch to the mask.

        Coordinates are in (x, y) format, starting from topleft corner.

        Parameters
        ----------
        patch : dict
            Dictionary with arguments for the patch.

        negative : bool, optional
            Indicates whether the patch adds positive or negative masking (default:
            False).

        Returns
        -------
        torch.Tensor
            Mask with activated regions corresponding to patches.
        """

        def raise_missing_attribute(attribute_name):
            if attribute_name not in patch:
                raise ValueError(f'Missing `{attribute_name}`.')

        raise_missing_attribute('shape')

        mask = np.tile(np.expand_dims(self.mask.numpy().astype(np.uint8) * 255, 2), 3)

        if patch['shape'] == 'circle':
            raise_missing_attribute('centre')
            raise_missing_attribute('radius')

            centre = patch['centre']
            radius = patch['radius']

            mask = cv2.circle(
                mask,
                centre,
                radius,
                FILL_COLOR if not negative else UNFILL_COLOR,
                -1,
            )

        elif patch['shape'] == 'rectangle':
            raise_missing_attribute('topleft')
            raise_missing_attribute('bottomright')

            # convert to int
            topleft = [int(patch['topleft'][idi]) for idi in range(2)]
            bottomright = [int(patch['bottomright'][idi]) for idi in range(2)]

            # convert relative coordinates in absolute coordinates
            topleft = [
                item if isinstance(item, int) else int(item * self.size_px[idi])
                for idi, item in enumerate(topleft)
            ]
            bottomright = [
                item - 1 if isinstance(item, int) else int(item * self.size_px[idi]) - 1
                for idi, item in enumerate(bottomright)
            ]

            mask = cv2.rectangle(
                mask,
                topleft,
                bottomright,
                FILL_COLOR if not negative else UNFILL_COLOR,
                -1,
            )

        elif patch['shape'] == 'ellipse':
            raise_missing_attribute('centre')
            raise_missing_attribute('major_axis_length')
            raise_missing_attribute('minor_axis_length')

            centre = [patch['centre'][idi] for idi in range(2)]
            major_axis_length = patch['major_axis_length']
            minor_axis_length = patch['minor_axis_length']

            # convert relative coordinates in absolute coordinates
            centre = [
                item if isinstance(item, int) else int(item * self.size_px[idi])
                for idi, item in enumerate(centre)
            ]
            major_axis_length = (
                major_axis_length
                if isinstance(major_axis_length, int)
                else int(major_axis_length * self.size_px[0])
            )
            minor_axis_length = (
                minor_axis_length
                if isinstance(minor_axis_length, int)
                else int(minor_axis_length * self.size_px[1])
            )

            # set default values for optional parameters
            if 'rotation_angle' not in patch:
                patch['rotation_angle'] = 0

            if 'start_angle' not in patch:
                patch['start_angle'] = 0

            if 'end_angle' not in patch:
                patch['end_angle'] = 360

            # create shape in the mask
            mask = cv2.ellipse(
                mask,
                centre,
                (major_axis_length, minor_axis_length),
                patch['rotation_angle'],
                patch['start_angle'],
                patch['end_angle'],
                FILL_COLOR if not negative else UNFILL_COLOR,
                -1,
            )

        # convert back mask as torch tensor
        self.mask = torch.from_numpy(mask[:, :, 0].astype(bool))

    def get_real_size_px(self, dpi=72):
        """Return the real size in pixels for a given DPI."""
        return [int(self.size_px[idd] * self.px_size[idd] * dpi) for idd in range(2)]

    def to(self, device):
        """Send all torch object to device.

        Parameters
        ----------
        device

        """
        self.mask = self.mask.to(device)
        self.cell_values = self.cell_values.to(device)

        if self.is_transparent:
            self.alpha = self.alpha.to(device)

    def __call__(self, img):
        """Apply in place the mask on the image with current values of masked values.

        Parameters
        ----------
        img : torch.tensor
            Image on which to apply the mask.
        """

        if self.is_transparent:
            img[:, self.mask] *= 1 - self.alpha
            img[:, self.mask] += self.values[:, self.mask] * self.alpha
        else:
            img[:, self.mask] = self.values[:, self.mask]

    def save_pdf(self, pdf_path, pagesize='A4'):
        """Save a pdf with patches.

        Parameters
        ----------

        """

        from reportlab.lib.pagesizes import A4, A3
        from reportlab.pdfgen import canvas

        # create blank page
        height, width = self.size_px
        page = torch.ones(3, height, width)

        # create the pdf canvas
        if pagesize.upper() == 'A4':
            pagesize = A4
        elif pagesize.upper() == 'A3':
            pagesize = A3

        pdf = canvas.Canvas(str(pdf_path), pagesize=pagesize)

        # get the different components of the patch
        n_components, components = cv2.connectedComponents(
            self.mask.numpy().astype(np.int8)
        )

        for idc in range(1, n_components):
            page = torch.ones(3, *self.size_px)

            page[:, components == idc] = self.values[:, components == idc]
            page = torch_to_opencv_img(page)

            min_y, min_x = np.argwhere(components == idc).min(0)
            max_y, max_x = np.argwhere(components == idc).max(0)

            shift_y = height - max_y - 5
            shift_x = -min_x + 5

            T = np.array(
                [
                    [
                        1,
                        0,
                        shift_x,
                    ],
                    [0, 1, shift_y],
                ],
                dtype=np.float32,
            )

            tpage = cv2.warpAffine(page, M=T, dsize=(width, height))
            tpage[0:shift_y, :] = 255
            tpage[:, width + shift_x : width] = 255

            pil_page = Image.fromarray(tpage)
            width_pt, height_pt = self.size_pt

            pdf.drawInlineImage(pil_page, 0, 0, width=width_pt, height=height_pt)
            pdf.showPage()

        pdf.save()

    def apply_transformation(self, transform, only_values=False):
        """Apply a transformation on the patch.

        Parameters
        ----------
        transform : kornia.augmentation.container.ImageSequential
            Container for transformation.
        only_values : bool, optional
            Applies to transformation only on values.

        """
        if only_values:
            self.values = transform(self.values)[0, ...]
        else:
            if transform.same_on_batch is None:
                transform.same_on_batch = True

            inputs = torch.stack(
                [torch.tile(self.mask, (3, 1, 1)).float(), self.values]
            )
            tr_inputs = transform(inputs)
            tr_inputs = tr_inputs.clip(0, 1)

            self.mask = tr_inputs[0, 0, ...] > 0.5
            self.values = tr_inputs[1, ...]

    def copy(self):
        copy_patch = Patch(
            size_px=self.size_px,
            size_mm=self.size_mm,
            cell_size=self.cell_size,
            is_transparent=self.is_transparent,
            is_bw=self.is_bw,
            logger=self.logger,
        )

        copy_patch._mask = self._mask.clone()
        copy_patch.cell_values = self.cell_values.clone()

        return copy_patch

    def __repr__(self):
        return 'Patch <{0[0]:d}x{0[1]:d}>'.format(self.size_px)
