# -*- coding: utf-8 -*-
"""GTSRB dataset.

References
----------
Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, ‘Man vs. computer: Benchmarking
machine learning algorithms for traffic sign recognition’, Neural Networks, vol. 32, pp.
323–332, 2012, doi: 10.1016/j.neunet.2012.02.016.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import csv
import shutil
import tqdm
import warnings

import torch
from torchvision import transforms

from .detection import DetectionDataset
from .utils import split_dataset
from ..utils import get_config

ID_TO_LABEL = {
    0: 'speed_20',
    1: 'speed_30',
    2: 'speed_50',
    3: 'speed_60',
    4: 'speed_70',
    5: 'speed_80',
    6: 'lifted_80',
    7: 'speed_100',
    8: 'speed_120',
    9: 'no_overtaking_general',
    10: 'no_overtaking_trucks',
    11: 'right_of_way_crossing',
    12: 'right_of_way_general',
    13: 'give_way',
    14: 'stop',
    15: 'no_way_general',
    16: 'no_way_trucks',
    17: 'no_way_one_way',
    18: 'attention_general',
    19: 'attention_left_turn',
    20: 'attention_right_turn',
    21: 'attention_curvy',
    22: 'attention_bumpers',
    23: 'attention_slippery',
    24: 'attention_bottleneck',
    25: 'attention_construction',
    26: 'attention_traffic_light',
    27: 'attention_pedestrian',
    28: 'attention_children',
    29: 'attention_bikes',
    30: 'attention_snowflake',
    31: 'attention_deer',
    32: 'lifted_general',
    33: 'turn_right',
    34: 'turn_left',
    35: 'go_straight',
    36: 'turn_straight_right',
    37: 'turn_straight_left',
    38: 'turn_right_down',
    39: 'turn_left_down',
    40: 'turn_circle',
    41: 'lifted_no_overtaking_general',
    42: 'lifted_no_overtaking_trucks',
}


def convert_gtsrb_dataset():
    """Convert original GTSRB dataset into a suitable format for
    :class:DetectionDataset`.
    """
    data_path = get_config().get_path('paths', 'data') / 'gtsrb'

    # Train split

    # create directories inside dataset folder
    for split_name in ['train', 'test']:
        img_paths = data_path / 'images' / split_name
        label_paths = data_path / 'labels' / split_name

        img_paths.mkdir(exist_ok=True, parents=True)
        label_paths.mkdir(exist_ok=True, parents=True)

        list_folders = [data_path / split_name]

        if split_name == 'train':
            list_folders = list(list_folders[0].glob('0*'))

        # get list of classes
        for folder_path in list_folders:

            if split_name == 'train':
                annotation_path = folder_path / f'GT-{folder_path.stem}.csv'
            elif split_name == 'test':
                annotation_path = folder_path / 'GT-final_test.csv'

            with open(annotation_path, 'r') as infile:
                csv_reader = csv.reader(infile, delimiter=';')

                # skip header
                next(csv_reader)

                for row in tqdm.tqdm((csv_reader)):

                    filename = row[0]

                    # check if file exists
                    img_path = folder_path / filename
                    if not img_path.exists():
                        warn_msg = 'File {} not found. Skipped.'
                        warnings.warn(warn_msg.format(img_path))
                        continue

                    x = int(row[3])
                    y = int(row[4])
                    width = int(row[5]) - x
                    height = int(row[6]) - y
                    class_id = row[7]

                    bbox = [x, y, width, height]

                    # copy the image
                    shutil.copy(img_path, img_paths / f'{class_id}_{img_path.name}')

                    # save labels
                    with open(
                        label_paths / f'{class_id}_{img_path.stem}.txt', 'w'
                    ) as outfile:

                        ann = bbox + [class_id]
                        outfile.write(' '.join([str(item) for item in ann]))
                        outfile.write('\n')


def get_loaders(
    batch_size,
    n_workers=0,
    img_size=256,
    img_transform_train=None,
    logger=None,
):
    """Get data loaders for the GTSRB dataset.

    Parameter
    ----------
    batch_size : int
        Size of a batch.
    n_workers : int, optional
        Number of workers (default: 0).
    img_size : int, optional
        Size of the output image (default: 256).
    img_transform_train : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    logger: Logger or None, optional
        Logging system (default: None).

    Returns
    -------
    dict
        Loaders, keyed by split name.
    dict
        Number of elements in each split, keyed by split name.
    dict
        Extra information about the dataset.
    """

    if img_transform_train is None:

        img_transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.1)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.1),
            ]
        )

    train_dataset = GTSRBDataset(
        'train',
        img_size=img_size,
        img_transform=img_transform_train,
        logger=logger,
        with_cache=False,
    )

    train_split, valid_split = split_dataset(train_dataset, split_ratio=4)

    test_dataset = GTSRBDataset(
        'test', img_size=img_size, logger=logger, with_cache=False
    )

    # define loaders
    loaders = {}
    loader_sizes = {}

    loaders['train'] = torch.utils.data.DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
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
        shuffle=False,
        num_workers=n_workers,
    )
    loader_sizes['test'] = len(test_dataset) // batch_size

    extras = {
        'n_classes': len(train_dataset.classes),
    }

    return loaders, loader_sizes, extras


class GTSRBDataset(DetectionDataset):
    """GTSRB dataset.

    Parameters
    ----------
    split_name : str
        Name of the split.
    img_size : int
        Size of the output image.
    img_transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    data_path : str or pathlib.Path, optional
        Path to the directory containing datasets. If None, the path is taken from the
        config file (default: None).
    with_cache : bool
        Indicates whether the cache, if existing, is used or not (default: True).
    logger: Logger or None, optional
        Logging system (default: None).
    """

    classes = list(ID_TO_LABEL.values())

    def __init__(
        self,
        split_name,
        img_size=128,
        img_transform=None,
        data_path=None,
        with_cache=True,
        logger=None,
    ):
        super().__init__(
            name='gtsrb',
            split_name=split_name,
            img_size=img_size,
            img_transform=img_transform,
            data_path=data_path,
            with_cache=with_cache,
            logger=logger,
        )

        self.classes = [ID_TO_LABEL[idc] for idc in range(len(ID_TO_LABEL))]
        self.n_classes = len(self.classes)

    def split_dataset(self, split_ratio):
        """Split the dataset in two, taking into account the scene in which images belong.

        Parameters
        ----------
        split_ratio : float, optional
            Ratio between the size of the two splits (default: 1).
        """
        raise NotImplementedError

    def __getitem__(self, idx):

        t_img, t_label, resizing_info, img_name = super().__getitem__(idx)
        return t_img, t_label[0, 5].long()

    def collate_fn(batch):
        return batch
