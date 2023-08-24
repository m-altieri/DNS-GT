# -*- coding: utf-8 -*-
"""Handling metrics.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import numpy as np


class Meter:
    """Meter to store numerical values

    A Meter stores scalar items and allows for various operations on them.

    Parameters
    ----------
    store : bool, optional
        Indicates if all items are sorted or only the last one.
    """

    def __init__(self, store=False):
        self.store = store
        self.items = None
        self.n_items = None
        self.sum = None
        self.avg = None
        self.min = None
        self.max = None

        self.reset()

    def reset(self):
        """Reset the list of items."""

        self.items = []
        self.n_items = 0
        self.sum = 0
        self.avg = 0
        self.min = np.inf
        self.max = -np.inf

    @property
    def last(self):
        """Return the last item."""
        return self.items[-1]

    def length(self):
        """Return the length of the meter."""
        return len(self)

    def __call__(self, items):
        """Update the meters with a list of items.

        Parameters
        ----------
        items : list or scalar
            Items to add in the meter.
        """

        if not isinstance(items, list):
            items = [items]

        if len(items) > 0:

            # store items
            if self.store:
                self.items += items

            # compute moving properties
            self.n_items += len(items)
            self.sum += sum(items)
            self.avg = self.sum / self.n_items
            self.min = np.minimum(self.min, np.min(items))
            self.max = np.maximum(self.min, np.max(items))

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):

        if hasattr(self, index):
            return getattr(self, index)
