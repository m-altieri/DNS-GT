# -*- coding: utf-8 -*-
"""Utils for color scheme handling.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_color_scheme(colormap=None, n_colors=32, with_bw=False):
    """Generate a color scheme.


    The number of colors is given by `n_values`^3.

    Parameters
    ----------
    colormap : matplotlib.colors.LinearSegmentedColormap or None, optional
        Name of the color scheme. If None, generate fixed permutations of values per
        channel (default: None).
    n_colors : int, optional
        Number of colors (default: 32)
    with_bw : bool, optional
        Indicates if black and white are added to the scheme (default: False).

    Returns
    -------
    torch.tensor
        color scheme.
    """

    if isinstance(colormap, str):
        colormap_list = {
            'tab20c': plt.cm.tab20c,
            'dark': plt.cm.Dark2,
            'gray': plt.cm.gray,
            'viridis': plt.cm.viridis,
        }
        colormap = colormap_list.get(colormap, None)

    if isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
        color_scheme = torch.tensor(colormap(np.linspace(0, 1, n_colors)))[:, 0:3].T
    else:
        n_values = int(np.round(n_colors ** (1 / 3)))
        grid_red, grid_green, grid_blue = torch.meshgrid(
            torch.linspace(0, 255, n_values),
            torch.linspace(0, 255, n_values),
            torch.linspace(0, 255, n_values),
            indexing='ij'
        )
        color_scheme = torch.cat(
            [grid_red.flatten(), grid_green.flatten(), grid_blue.flatten()]
        )
        color_scheme = color_scheme.view(3, -1) / 255.0

    if with_bw:

        # add black and white colors in colors if missing
        black_color = torch.zeros((3, 1))
        white_color = torch.ones((3, 1))

        if not (color_scheme == black_color).prod(0).any():
            color_scheme = torch.cat([color_scheme, black_color], axis=1)

        if not (color_scheme == white_color).prod(0).any():
            color_scheme = torch.cat([color_scheme, white_color], axis=1)

    return color_scheme


def plot_color_scheme(color_scheme):
    """Plot a color scheme.

    Parameters
    ----------
    color_scheme: torch.tensor
        Tensors containing a list of RGB colors.
    """

    # test the dimensions of the array
    if color_scheme.ndim != 2:
        err_msg = 'Number of dimensions should be 2 (given: {:d}).'
        raise ValueError(err_msg.format(color_scheme.ndim))

    n_channels, n_colors = color_scheme.shape

    if n_channels != 3:
        err_msg = 'Shape of dimension 0 (number of channels) should be 3 (given: {:d}).'
        raise ValueError(err_msg.format(n_channels))

    plt.figure()
    plt.imshow(color_scheme.unsqueeze(-1).numpy().transpose(2, 1, 0))
    plt.title(f'{n_colors:d} colors')
    plt.xticks([])
    plt.yticks([])
