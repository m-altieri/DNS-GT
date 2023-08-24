# -*- coding: utf-8 -*-
"""Detection dataset class.

In this context, 'Detection' should be understood as a synonym of 'Recognition', i.e.,
detecting if an object is present or not, and recognizing what type of object it is.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from .image import load_image, IMG_FORMATS, resize_image
from ..utils import check_path, compute_hash, get_config, get_logger


class DetectionDataset(torch.utils.data.Dataset):
    """Dataset for detection tasks.

    Folder structure
    ================
    dataset/
        images/
            split1/
                image1.*
                image2.*
                ...
            split2/
                ...
            ...
                ...
        labels/
            split1/
                image1.txt
                image2.txt
                ...
            split2/
                ...
            ...

    Images
    ======
    * Name of images does not matter.
    * Images can be of any supported format (see :mod:`tacks.image` for list of
    supported image formats).

    Labels
    ======
    * Labels are text files with extension '.txt'
    * Each line corresponds to a bounding box around an object.
    * Each line is made of 5 columns
        1-4. Coordinates at the 'xywh' format.
        5.   Class of the object.

    Sample format
    =============
    A sample is composed of three elements.

    **Image**
        * torch.Tensor.
        * CHW format.
        * Resized to match the desired image size.
        * Float values (0-1)

    **Label**
        * torch.Tensor
        * Coordinates are adjusted to the size of the image.
        * Each row corresponds to a bounding box.
        * Coordinates are normalized with respect to the size of the image.
        * Each row is made of 6 columns:
            1. Number of the sample in the batch.
            2. Class of the object.
            3-6. Coordinates are at the format 'xywh'

    **Path**
        * Path of the original image.

    Parameters
    ----------
    name : str
        Name of the dataset.
    split_name : str
        Name of the split.
    img_size : int or 2-tuple of ints
        Size of the output image.
    transform : torchvision.transforms.Compose, optional
        Transform object for the image (default: None).
    data_path : pathlib.Path or str
        Path to the samples.
    with_cache : bool
        Indicates whether the cache, if existing, is used or not (default: True).
    logger: Logger or None, optional
        Logging system (default: None).
    """


    def __init__(
        self,
        name,
        split_name,
        img_size=640,
        img_transform=None,
        data_path=None,
        with_cache=True,
        logger=None,
    ):

        self.name = name
        self.classname = self.__class__.__name__
        self.logger = logger
        self.with_cache = with_cache
        self.img_transform = img_transform
        self.img_size = img_size
        self.split_name = split_name

        if self.logger:
            self.logger.info(
                '%s| Loading split %s of dataset %s.',
                self.classname,
                self.split_name,
                self.name,
            )

        # Get the path where data are stored if not provided
        if data_path is None:
            data_path = get_config().get_path('paths', 'data')

        self.data_path = Path(data_path) / self.name
        check_path(data_path)

        # get the paths of all the images in the split folder
        self.img_paths = sorted(
            [
                item
                for item in (self.data_path / 'images' / split_name).glob('*')
                if item.suffix in IMG_FORMATS
            ]
        )

        if len(self.img_paths) == 0:
            err_msg = 'No image found in {}.'
            raise FileNotFoundError(
                err_msg.format(self.data_path / 'images' / split_name)
            )

        if self.logger:
            self.logger.info('%s| Found %d images.', self.classname, len(self.img_paths))

        # get the paths of labels of images
        self.label_paths = [
            self.data_path / 'labels' / split_name / (item.stem + '.txt')
            for item in self.img_paths
        ]

        # container for labels and image sizes
        self.labels = None
        self.sizes = None

        # if the cache file exists, load labels and sizes from cache
        if self.with_cache:
            self.load_cache()

        # if containers are still None, get labels and sizes
        if self.labels is None or self.sizes is None:
            self.get_labels()
            self.get_sizes()

        # save labels and sizes in cache
        if self.with_cache:
            self.save_cache()

        # set attributes
        self.input_shape = (3, img_size, img_size)

    def load_cache(self):
        """Load cache.

        The validity of the cache is checked by computing a hash over the labels and
        images paths.
        """

        cache_path = self.data_path / f'{self.split_name}.cache'

        if cache_path.exists():
            if self.logger:
                self.logger.info('Loading cache...')

            cache = torch.load(str(cache_path))

            # check if the dataset has changes since last caching
            current_hash = compute_hash(str(sorted(self.label_paths + self.img_paths)))

            if cache['hash'] != current_hash:
                # do nothing
                if self.logger:
                    self.logger.info('Invalid hash of the cache.')
            else:
                self.labels, self.sizes = zip(
                    *[cache[str(img_path)] for img_path in self.img_paths]
                )
                if self.logger:
                    self.logger.info('Cache loaded.')

    def save_cache(self):
        """Save labels and sizes in the cache."""

        cache = {
            str(self.img_paths[idi]): (self.labels[idi], self.sizes[idi])
            for idi in range(len(self.img_paths))
        }

        # compute hash to check integrity of database at loading
        cache['hash'] = compute_hash(str(sorted(self.label_paths + self.img_paths)))

        cache_path = self.data_path / f'{self.split_name}.cache'
        torch.save(cache, cache_path)

        if self.logger:
            self.logger.info('Cache saved at %s', cache_path)

    def get_labels(self):
        """Get labels corresponding to images.

        If a label file does not exist, an empty label is used.
        """

        if self.logger:
            self.logger.debug('%s| Getting labels of images...', self.classname)

        self.labels = []

        for label_path in self.label_paths:

            if label_path.exists():
                # open label file
                with open(label_path, 'r') as infile:
                    label = np.array(
                        [item.split() for item in infile.read().splitlines()],
                        dtype=np.int32,
                    )

                if label.shape[0] == 0:
                    err_msg = 'Missing label for {}.'
                    raise ValueError(err_msg.format(label_path.name))
                else:

                    if label.shape[1] != 5:
                        err_msg = 'Wrong label: {} columns found instead of 5.'
                        raise ValueError(err_msg.format(label.shape[1]))

                    if (label < 0).any():
                        err_msg = 'Found negative values in label.'
                        raise ValueError(err_msg)

            else:
                err_msg = 'Missing label for {}.'
                raise ValueError(err_msg.format(label_path.name))

            # add label in list of labels
            self.labels.append(label)

    def get_sizes(self):
        """Get sizes of images."""
        if self.logger:
            self.logger.debug('%s| Getting sizes of images...', self.classname)

        self.sizes = []

        to_be_removed = []
        for img_path in self.img_paths:
            img = cv2.imread(str(img_path))

            if img is None:
                warn_msg = 'Unable to load image {}. To be removed.'
                warnings.warn(warn_msg.format(img_path))
                to_be_removed.append(img_path)
                continue

            self.sizes.append(img.shape[0:2])

        # remove skipped images
        if len(to_be_removed) > 0:
            self.logger.info('Removing %d images.', len(to_be_removed))
            for img_path in to_be_removed:
                self.img_paths.remove(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # open image
        orig_img = load_image(self.img_paths[idx])

        # get image path
        img_path = self.img_paths[idx]
        orig_height, orig_width, n_channels = orig_img.shape

        if n_channels == 1:
            raise NotImplementedError('Single-channel images are not supported yet.')

        if self.logger:
            self.logger.debug(
                'Load image: %s - (%d, %d)', img_path.name, orig_height, orig_width
            )

        # resize image
        img, resizing_info = resize_image(
            orig_img, img_size=(self.img_size, self.img_size), keep_ratio=True
        )

        if self.logger:
            self.logger.debug(
                'Resize image: %s to (%d, %d). Pad: %s, Ratio: %.2f',
                img_path.name,
                img.shape[0],
                img.shape[1],
                resizing_info['padding'],
                resizing_info['ratio'],
            )

        # transpose image, convert as torch tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # normalize values between 0 and 1
        img /= 255.0

        if self.img_transform is not None:
            img = self.img_transform(img)

        # load label
        label = self.labels[idx].astype(np.float32)

        # scale and translate coordinates
        label[:, [0, 2]] *= resizing_info['ratio']
        label[:, 0] += resizing_info['padding'][2]

        label[:, [1, 3]] *= resizing_info['ratio']
        label[:, 1] += resizing_info['padding'][0]

        # rescale according to the size of the new image
        label[:, 0:4] /= self.img_size

        # define a torch tensor to store label, with an extra column to allow stacking
        # the first column contains the id of the image in the batch
        t_label = torch.zeros((label.shape[0], 6))

        # add label in torch tensor
        t_label[:, 1::] = torch.from_numpy(label)

        return (img, t_label, resizing_info, self.img_paths[idx].name)

    @staticmethod
    def collate_fn(batch):
        """Override :func:`collate_fn` to add the id of the sample in the first
        column."""
        imgs, labels, infos, paths = zip(*batch)

        # add image index for batch stacking
        for idl, label in enumerate(labels):
            label[:, 0] = idl

        return (torch.stack(imgs, 0), torch.cat(labels, 0), infos, paths)
