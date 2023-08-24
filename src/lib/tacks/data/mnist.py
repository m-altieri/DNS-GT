# -*- coding: utf-8 -*-
"""MNIST dataset utils.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
from torchvision import datasets, transforms

from ..utils import get_config
from .utils import split_dataset


def load_dataset(split_name, data_path=None, transform=None, logger=None):
    """Load a split of the MNIST dataset.

    Parameters
    ----------
    split_name : {'train', 'test'}
        Name of the split.
    data_path : str or pathlib.Path, optional
        Path to the directory containing datasets. If None, the path is taken from the
        config file (default: None).
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    logger: Logger or None, optional
        Logging system (default: None).

    Returns
    -------
    torch.utils.data.Dataset
    """

    # read the data path from the config file if not provided
    if data_path is None:
        config = get_config()
        data_path = config.get_path('paths', 'data')

    if split_name not in ('train', 'test'):
        raise ValueError(f'Unknown split: {split_name}')

    if transform is None:
        transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        data_path, train=(split_name == 'train'), download=True, transform=transform
    )

    dataset.name = 'MNIST'
    dataset.in_shape = (1, 28, 28)

    return dataset


def get_loaders(batch_size, n_workers=0, reduced=False, logger=None, collate_fn=None):
    """Get data loaders for the MNIST dataset.

    Parameters
    ----------
    batch_size : int
        Size of a batch.
    n_workers : int, optional
        Number of workers (default: 0).
    logger: Logger or None, optional
        Logging system (default: None).
    reduced: bool, optional
        Inidicates if the dataset size is reduced or not (default: False).
    collate_fn: callable or None, optional
        Custom collate function for data loaders (default: None).

    Returns
    -------
    dict
        Loaders, keyed by split name.
    dict
        Number of elements in each split, keyed by split name.
    dict
        Extra information about the dataset.
    """

    # load datasets and loaders
    train_dataset = load_dataset(split_name='train', logger=logger)

    if reduced:
        train_dataset.data = train_dataset.data[0:512, ...]
        train_dataset.targets = train_dataset.targets[0:512]

    train_split, valid_split = split_dataset(train_dataset, split_ratio=4)

    test_dataset = load_dataset(split_name='test', logger=logger)

    if reduced:
        test_dataset.data = test_dataset.data[0:10, ...]
        test_dataset.targets = test_dataset.targets[0:10]

    # define loaders
    loaders = {}
    loader_sizes = {}

    loaders['train'] = torch.utils.data.DataLoader(
        train_split,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )
    loader_sizes['train'] = len(train_split) // batch_size

    loaders['valid'] = torch.utils.data.DataLoader(
        valid_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    loader_sizes['valid'] = len(valid_split) // batch_size

    loaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=n_workers,
    )
    loader_sizes['test'] = len(test_dataset) // batch_size

    extras = {
        'n_classes': len(train_dataset.classes),
        'in_shape': train_dataset.in_shape,
        'clip_values': (0, 1)
    }

    return loaders, loader_sizes, extras
