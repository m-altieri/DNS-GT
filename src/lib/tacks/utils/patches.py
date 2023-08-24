# -*- coding: utf-8 -*-
"""Utils for patch handling.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

from ..data.image import resize_image


def resize_patch_to_print(patch, resize_factor, img_size):
    """Save patch in correct size for a specific printer.

    Parameters
    ----------
    patch : array-like
        Numpy array containing the mask that describes the patch region.
    resize_factor : float
        Empirical printer factor that determines the effective scaling needed for a
        patch to have the right physical size when printed. Needs to be determined from
        printer tests.
    img_size : int
        original image size from internal processing that is resized to adjust for the
        printer.

    Returns
    -------
    None.
    """
    printing_size = (int(img_size * resize_factor), int(img_size * resize_factor))
    resized_patch, _ = resize_image(patch, printing_size, keep_ratio=True)

    return resized_patch
