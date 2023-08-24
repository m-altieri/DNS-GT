# -*- coding: utf-8 -*-
"""Utils classes and functions for handling image.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from ..utils import check_path
from ..utils.image import resize_image, opencv_to_torch_img

IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.ppm']

DEFAULT_TEXT_PARAMS = {
    'org': (30, 20),
    'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
    'fontScale': 0.3,
    'color': (0, 0, 0),
    'thickness': 1,
    'lineType': cv2.LINE_AA,
}


def load_image(img_path, color_repr='rgb', normalization=False, out_fmt='numpy'):
    """load an image from a given path.

    Parameters
    ----------
    img_path : pathlib.Path or str
        Path to the image.
    color_repr : {'rgb', 'hsl'} or None, optional
        Color representation (default: 'rgb'). If None, default color representation is
        used.
    normalization : bool, optional
        Indicates if the image is normalized between 0 and 1 (default: False).
    out_fmt : {'opencv', 'torch'}, optional
        Indicates the output format (default: 'numpy').

    Returns
    -------
    array-like
        Image with 3 channels in format HWC with values encoded as integers in {0, ,
        ..., 255} if `normalization` is False else as float in [0, 1].
    """
    check_path(img_path)
    img = cv2.imread(str(img_path))

    if color_repr is not None:
        if color_repr == 'rgb':
            convert_color = cv2.COLOR_BGR2RGB
        elif color_repr == 'hsl':
            convert_color = cv2.COLOR_BGR2HSL
        else:
            err_msg = 'Unknown color representation: {:s}.'
            raise ValueError(err_msg.format(color_repr))

        img = cv2.cvtColor(img, convert_color)

    if normalization:
        img /= 255

    if out_fmt == 'torch':
        img = opencv_to_torch_img(img)

    return img


def save_image(img, img_path, text=None, text_params=None):
    """Save an image to a given path.

    Parameters
    ----------
    img : array-like or torch.tensor
        Image to save. If array-like, should be in HWC format HWC, with values encoded
        either as float in [0, 1], or as integers in [0, 255]. If torch.tensor, should
        be in CHW forwat with values encoded as float in [0, 1].
    img_path : pathlib.Path or str
        Path to the image.
    text : str or None, optional
        Add text on the image if not None (default: None).
    text_params : dict or None, optional
        Parameters of the text (default: None).
    """

    # if as a batch, remove first dimension
    if len(img.shape) == 4:
        img = img[0, ...]

    # convert tensor image in numpy array
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)

    # convert into int values in  [0, 255]
    if img.dtype != np.uint8:
        # convert to uint8 dtype
        img = (img * 255).astype(np.uint8)

    # convert to CV format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # add text
    if text is not None:

        if text_params is None:
            text_params = DEFAULT_TEXT_PARAMS.copy()

        for idl, line in enumerate(text.split('\n')):
            text_params['org'] = (
                text_params['org'][0],
                text_params['org'][0] + idl * 10,
            )
            img = cv2.putText(img, line, **text_params)

    cv2.imwrite(str(img_path), img)


class ImageFolderDataset(torch.utils.data.Dataset):
    """Dataset made of single images without annotations.

    Path can be of:
        * a single image;
        * a folder containing images;
        * a wildcard pattern for images.

    Parameters
    ----------
    data_path : pathlib.Path or str
        Path to the samples.
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    img_size : 2-tuple
        Size of the image.
    keep_ratio : bool, optional
        Indicates if the ratio is preserved or not. If True, pad the image the achieve
        the desired size (default: True).

    Returns
    -------
    pathlib.Path
        Path to the image.
    torch.Tensor
        Tensor containing the image for inference.
    torch.Tensor
        Padding added to the image.
    torch.Tensor
        Ratio of the image compared to the original one.

    """

    def __init__(self, data_path, transform=None, img_size=None, keep_ratio=True):

        super().__init__()

        self.data_path = Path(data_path)

        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.transform = transform

        # get list of files
        if self.data_path.is_file():
            files = [self.data_path]
        elif self.data_path.is_dir():
            files = sorted(self.data_path.glob('**/*.*'))
        elif '*' in str(self.data_path):
            files = sorted(self.data_path.parent.glob(self.data_path.name))
        else:
            raise FileNotFoundError(f'Unknown data: {data_path}')

        # extract images
        self.img_paths = [item for item in files if item.suffix.lower() in IMG_FORMATS]

        # check paths
        for img_path in self.img_paths:
            check_path(img_path)

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        if isinstance(img_size, int):
            self._img_size = (img_size, img_size)
        elif (
            isinstance(img_size, tuple)
            and len(img_size) == 2
            and isinstance(img_size[0], int)
            and isinstance(img_size[1], int)
        ):
            self._img_size = img_size
        else:
            err_msg = 'Incorrect value for img_size: {}.'
            raise ValueError(err_msg.format(str(img_size)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # load the image
        img = load_image(self.img_paths[idx], normalization=False)

        # resize the image
        if self.img_size is not None:
            img, resizing_info = resize_image(
                img, self.img_size, keep_ratio=self.keep_ratio
            )
        else:
            resizing_info = {'padding': [0, 0, 0, 0], 'ratio': 1}

        # convert to torch format CRGB
        img = img.transpose(2, 0, 1)

        # convert to Tensor and to float
        img = torch.from_numpy(img).float()

        # normalize images
        img /= 255.0

        # use transform
        if self.transform is not None:
            img = self.transform(img)

        return img, str(self.img_paths[idx]), resizing_info


class TorchImageDataset(torch.utils.data.Dataset):
    """Dataset from batch of images with the support of torchvision.transforms objects.

    Parameters
    ----------
    tensors : torch.Tensor
        Batch of images in torch format.
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    """

    def __init__(self, tensors, transform=None):
        super().__init__()
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index, ...]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.tensors.size(0)
