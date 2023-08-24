# -*- coding: utf-8 -*-
"""Util functions for images.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import cv2
import numpy as np
import torch


def get_text_params_opencv(size, color=(26, 40, 210), loc=(20, 40)):
    """Get text font params for cv2.putText function.

    Parameters
    ----------
    size : ['tiny', 'small', 'medium', 'large', 'big']
        Size of the font.
    color : 3-tuple of ints
        RGB color of the text.
    loc : 2-tuple of ints
        Coordinates of the starting location of the text.

    """
    if size == 'tiny':
        font_scale = 0.5
        thickness = 2
    elif size == 'small':
        font_scale = 0.8
        thickness = 2
    elif size == 'medium':
        font_scale = 1.0
        thickness = 3
    elif size == 'large':
        font_scale = 1.4
        thickness = 4
    elif size == 'big':
        font_scale = 1.8
        thickness = 5
    elif size == 'huge':
        font_scale = 2.1
        thickness = 6
    else:
        raise ValueError(f'Unknown size: {size:s}')

    return {
        'org': loc,
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': font_scale,
        'color': color,
        'thickness': thickness,
        'lineType': cv2.LINE_AA,
    }


def permute_axes_img(img, fmt='CWH'):
    """Permute the axis of an image to get desired format.

    Parameters
    ----------
    img : array-like
        Image.
    fmt : ['CWH', 'WHC']
        Format of the image.
    """

    if len(img.shape) != 3:
        err_msg = 'Unknown number of axes for image: {:d}'
        raise ValueError(err_msg.format(len(img.shape)))

    if fmt == 'CWH':
        axis_perm = (2, 0, 1)
    elif fmt == 'WHC':
        axis_perm = (1, 2, 0)
    else:
        err_msg = 'Wrong format: {:s}'
        raise ValueError(err_msg.format(fmt))

    if isinstance(img, torch.Tensor):
        return img.permute(*axis_perm)
    elif isinstance(img, np.ndarray):
        return img.transpose(*axis_perm)
    else:
        err_msg = 'Unknown type for image: {:s}'
        raise ValueError(err_msg.format(type(img)))


def torch_to_opencv_img(torch_img):
    """Convert a torch image into an openCV image.

    Parameters
    ----------
    torch_img : torch.Tensor
        Image to convert.

    Returns
    -------
    array-like
        Image array to use in OpenCV.

    Warnings
    --------
    If the input image is not a torch.Tensor, the object is returned as such.
    """
    if not isinstance(torch_img, torch.Tensor):
        err_msg = 'Image is not a torch tensor: {:s}.'
        raise ValueError(err_msg.format(type(torch_img)))
        return torch_img

    else:
        return np.ascontiguousarray(
            permute_axes_img(torch_img, fmt='WHC').numpy() * 255
        ).astype(np.uint8)


def opencv_to_torch_img(np_img):
    """Convert an openCV image into a torch image.

    Parameters
    ----------
    np_img : torch.Tensor
        Image to convert.

    Returns
    -------
    array-like
        Image array to use in OpenCV.
    """
    if not isinstance(np_img, np.ndarray):
        err_msg = 'Image is not a numpy array: {:s}.'
        raise ValueError(err_msg.format(type(np_img)))
    else:
        img = permute_axes_img(np_img, fmt='CWH')

    if np_img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0

    return torch.from_numpy(img).float()


def resize_image(
    img, img_size, keep_ratio=False, fill_color=(114, 114, 114), best_interpolation=True
):
    """Resize an image to a given size.

    Parameters
    ----------
    img : np.ndarray or torch.tensor
        Image in numpy format (HWC) or tensor format (CWH), with values encoded either
        as float in [0, 1], or as integers in [0, 255].
    img_size : tuple of int or int
        Size of the image. If int, the same value is used for both dimensions.
    keep_ratio : bool, optional
        Indicates if the ratio of the image is preserved, by padding the image (default:
        False).
    fill_color : 3-tuple, optional
        Color to use for borders if padding occurs (default: (114, 114, 114)).
    best_interpolation : bool, optional
        Indicates whether the best interpolation is used (INTER_AREA for downsizing,
        INTER_NEAREST for upsizing), otherwise use default linear interpolation
        (default: True).

    Returns
    -------
    array-like
        Resized image in the same format.
    dict
        Information about resizing:
            padding: list of int
                Padding added, at the format [top, bottom, left, right]
            ratio : int
                Ratio of the image
    """

    is_dtype_float = img.dtype != np.uint8

    if not (isinstance(img, torch.Tensor) or isinstance(img, np.ndarray)):
        err_msg = 'Unknown type for the image: {}'
        raise ValueError(err_msg.format(type(img)))

    is_torch_tensor = isinstance(img, torch.Tensor)

    if is_torch_tensor:
        if not is_dtype_float:
            err_msg = 'Torch tensors should be float valued.'
            raise ValueError(err_msg)

        img = torch_to_opencv_img(img)
    else:
        if is_dtype_float:
            img = (img * 255).astype(np.uint8)

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    height, width = img.shape[:2]
    new_height, new_width = img_size

    old_ratio = height / width
    ratio = new_height / new_width

    r_ratio = np.round(old_ratio / ratio, 3)

    if best_interpolation:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR

    if keep_ratio and r_ratio != 1:
        ratio = min(new_height / height, new_width / width)

        # calculate the height of the image without padding
        unpad_height = int(round(height * ratio))
        unpad_width = int(round(width * ratio))

        # calculate the pad to add on each side of the image
        padding = [
            (new_height - unpad_height) // 2 + (new_height - unpad_height) % 2,
            (new_height - unpad_height) // 2,
            (new_width - unpad_width) // 2 + (new_width - unpad_width) % 2,
            (new_width - unpad_width) // 2,
        ]

        info = {'padding': padding, 'ratio': ratio}

        # resize the image without padding
        img = cv2.resize(img, (unpad_width, unpad_height), interpolation=interp)

        # add a border to get the right size
        img = cv2.copyMakeBorder(
            img,
            info['padding'][0],
            info['padding'][1],
            info['padding'][2],
            info['padding'][3],
            cv2.BORDER_CONSTANT,
            value=fill_color,
        )

    else:
        img = cv2.resize(img, (new_width, new_height), interpolation=interp)
        info = {'padding': [0, 0, 0, 0], 'ratio': new_width / width}

    if is_dtype_float:
        # convert back to float dtype
        img = img / 255.0

    if is_torch_tensor:
        img = opencv_to_torch_img(img)

    return img, info
