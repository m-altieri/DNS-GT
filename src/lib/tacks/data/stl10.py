# -*- coding: utf-8 -*-
"""STL-10 dataset utils.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

from torchvision import datasets, transforms
from tacks.utils import get_config


def load_dataset(split_name, data_path=None, transform=None, logger=None):
    """Load a split of the STL-10 dataset.

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
        Logging system (default: None).

    Returns
    -------
    torch.utils.data.Dataset
    """

    # read the data path from the config file if not provided
    if data_path is None:
        config = get_config()
        data_path = config.get_path('paths', 'data')

    if split_name not in ('train', 'test', 'unlabeled', 'train+unlabeled'):
        raise ValueError(f'Unknown split: {split_name}')

    if transform is None:
        transform = transforms.ToTensor()

    dataset = datasets.STL10(data_path, split=split_name, transform=transform)

    dataset.name = 'STL10'
    dataset.in_shape = (3, 96, 96)

    return dataset
