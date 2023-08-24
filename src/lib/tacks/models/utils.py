# -*- coding: utf-8 -*-
"""Util functions for models.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch


def get_correct_device(device, logger):
    """Return the correct device based on different possible inputs.

    The presence of the device on the machine is tested.

    Parameters
    ----------
    device : {'cpu', 'cuda', 'cuda:\\d'} or int or None or torch.device
        Input device, in various format.
    logger : logging.Logger
        Logging system.

    Returns
    -------
    torch.device
        Device.
    """

    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device is None:
        device = default_device

    elif isinstance(device, int):
        device = f'cuda:{device:d}'

    elif isinstance(device, str):
        if device == 'cuda':
            device = 'cuda:0'

    # test the presence of the device by sending a tensor on the device
    try:
        torch.tensor(0).to(device)
    except RuntimeError:
        logger.warn(
            'Selected device %s does not exist. %s is used instead.',
            device,
            default_device,
        )
        device = default_device

    return torch.device(device)


def append_one_hot_channels(imgs, labels, n_classes):
    """Append one hot channels in an image.

    Parameters
    ----------
    imgs : torch.Tensor
        Batch of images.
    labels : torch.Tensor
        Labels of images.
    n_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Image with one hot channels encoding labels.
    """
    one_hot_vector = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(
        imgs.device
    )
    one_hot_image = one_hot_vector.view(*one_hot_vector.shape, 1, 1).repeat(
        1, 1, imgs.shape[2], imgs.shape[3]
    )
    return torch.cat([imgs, one_hot_image], dim=1)


def append_one_hot_vector(x, labels, n_classes):
    """Append a one hot vector to a vector.

    Parameters
    ----------
    imgs : torch.Tensor
        Batch of images.
    labels : torch.Tensor
        Labels of images.
    n_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Image with one hot channels encoding labels.
    """
    one_hot_vector = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(
        x.device
    )
    return torch.cat([x, one_hot_vector], dim=1)
