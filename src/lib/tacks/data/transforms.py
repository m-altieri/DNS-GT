# -*- coding: utf-8 -*-
"""Definition of custom transformations.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import kornia
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from torchvision import transforms


class Normalize1D:
    """Normalize a 1D vector by subtracting the mean and dividing by the
    standard deviation.

    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def get_inverse_transform(self):
        return Unnormalize1D(self.mean, self.std)

    def __call__(self, sample):
        mean = sample.mean() if self.mean is None else self.mean
        std = sample.std() if self.std is None else self.std
        return (sample - mean) / std

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class Unnormalize1D:
    """Unnormalize a 1D vector by subtracting the mean and dividing by the
    standard deviation.

    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample * self.std) + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class Unnormalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        self.imean = mean
        self.istd = std
        super().__init__(
            mean=[-m / s for m, s in zip(self.imean, self.istd)],
            std=[1 / s for s in self.istd],
            inplace=inplace,
        )


class ColorJiggle(IntensityAugmentationBase2D):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        super().__init__(p=1.0, same_on_batch=True, keepdim=False)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply_transform(self, x, params, flags, transform=None):
        transforms = [
            lambda img: kornia.enhance.adjust_brightness(img, self.brightness),
            lambda img: kornia.enhance.adjust_gamma(img, self.contrast + 1),
            lambda img: kornia.enhance.adjust_saturation(img, self.saturation + 1),
            lambda img: kornia.enhance.adjust_hue(img, self.hue),
        ]

        jittered = x
        for idx in range(len(transforms)):
            t = transforms[idx]
            jittered = t(jittered)

        return jittered
