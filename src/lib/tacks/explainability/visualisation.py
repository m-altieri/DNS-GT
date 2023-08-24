"""Visualisation of features.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

CMAP_RED_GREEN = LinearSegmentedColormap.from_list("RdWhGn",
                                                   ["red", "white", "green"])


def plot_attribute_maps(attribute_map):
    """Plot the attribute maps as heatmap on top of the instance.

    If the attribute maps is a RGB image, mean over channels is computed to
    obtain a single channel image. Values are colored from red (negative
    values) to green (positive values).

    Parameters
    ----------
    attribute_map : array-like
        Attribute map.
    """

    if attribute_map.ndim == 3:
        attribute_map = attribute_map.mean(-1)

    # get the maximal value
    vmax = np.abs(attribute_map).max()
    alpha = 0.85 * np.abs(attribute_map) / vmax

    plt.imshow(attribute_map,
               alpha=alpha,
               cmap=CMAP_RED_GREEN,
               vmin=-vmax,
               vmax=vmax)
    plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax, vmax),
                                       cmap=CMAP_RED_GREEN),
                 orientation='horizontal')
