# -*- coding: utf-8 -*-
"""CIFAR dataset utils.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
from torchvision import datasets, transforms

from ..utils import get_config
from .utils import split_dataset


def load_dataset(split_name, n_classes=10, data_path=None, transform=None, logger=None):
    """Load a split of the CIFAR dataset.

    Parameters
    ----------
    split_name : {'train', 'test'}
        Name of the split.
    n_classes : {10, 100}, optional
        Number of classes of the dataset (default: 10).
    data_path : str or pathlib.Path, optional
        Path to the directory containing datasets. If None, the path is taken from the
        config file (default: None).
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    logger: Logger or None, optional
        Logging system (default: None).

    Returns
    -------
    torch.utils.data.Dataset
    """

    dataset_name = f'CIFAR{n_classes}'
    if logger:
        logger.info('Loading %s split of the %s dataset...', split_name, dataset_name)

    # read the data path from the config file if not provided
    if data_path is None:
        config = get_config()
        data_path = config.get_path('paths', 'data')

    if split_name not in ('train', 'test'):
        raise ValueError(f'Unknown split: {split_name}')

    if n_classes not in (10, 100):
        raise ValueError(f'Unknown number of classes: {n_classes}')

    if transform is None:
        transform = transforms.ToTensor()

    if n_classes == 10:
        dataset = datasets.CIFAR10(
            data_path, train=(split_name == 'train'), download=True, transform=transform
        )
    elif n_classes == 100:
        dataset = datasets.CIFAR100(
            data_path, train=(split_name == 'train'), download=True, transform=transform
        )

    dataset.name = dataset_name
    dataset.in_shape = (3, 32, 32)

    return dataset


def get_loaders(
    batch_size,
    n_workers=0,
    transform_train=None,
    reduced=False,
    logger=None,
    collate_fn=None,
):
    """Get data loaders for the CIFAR10 dataset.

    Parameters
    ----------
    batch_size : int
        Size of a batch.
    n_workers : int, optional
        Number of workers (default: 0).
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    reduced : bool, optional
        Indicates if a reduced version of the dataset (10%) is returned or not (default:
        False).
    logger: Logger or None, optional
        Logging system (default: None).
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

    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=15),
                # transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.1),
                transforms.ToTensor(),
            ]
        )

    # transform_test = transforms.Compose(
    #     [transforms.Resize(img_size), transforms.ToTensor()]
    # )

    # load datasets and loaders
    train_dataset = load_dataset(
        split_name='train', transform=transform_train, logger=logger
    )
    if reduced:
        train_dataset.data = train_dataset.data[0:5000, ...]
    train_split, valid_split = split_dataset(train_dataset, split_ratio=4)

    test_dataset = load_dataset(split_name='test', logger=logger)

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
        valid_split, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )
    loader_sizes['valid'] = len(valid_split) // batch_size

    loaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )
    loader_sizes['test'] = len(test_dataset) // batch_size

    extras = {
        'n_classes': len(train_dataset.classes),
        'in_shape': train_dataset.in_shape,
    }

    return loaders, loader_sizes, extras
