# -*- coding: utf-8 -*-
"""Utils function for datasets.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly to achieve a given distribution over labels.

    Parameters
    ----------
    indices : list of ints, optional
        List of indices. If None, a list based on the length of `labels' is
        used.
    labels : list of ints
        List of labels for all samples.
    label_dist : list of floats
        Desired distribution over labels.


    """

    def __init__(self, labels, indices=None, label_dist=None):

        if indices is None:
            indices = range(len(labels))

        self.n_samples = len(indices)

        # weight for each sample
        weights = [label_dist[labels[idx]] for idx in indices]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            idi
            for idi in torch.multinomial(self.weights, self.n_samples, replacement=True)
        )

    def __len__(self):
        return self.n_samples


def split_dataset(dataset, split_ratio=1):
    """Split a dataset into two subsets, according to a given ratio.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or torch.utils.data.Subset
        Dataset or subset of a dataset to split.
    split_ratio : float, optional
        Ratio between the size of the two splits (default: 1).

    Returns
    -------
    tuple of torch.utils.data.Dataset
        Two splits of the dataset.
    """
    n_examples = len(dataset)
    n_split1_examples = int(n_examples * split_ratio / (1 + split_ratio))

    return torch.utils.data.random_split(
        dataset, (n_split1_examples, len(dataset) - n_split1_examples)
    )


def get_loaders(load_dataset, split_names, batch_size, n_workers):
    """Return loaders from a load_dataset function.

    Parameters
    ----------
    load_dataset : func
        Function taking as input a split name and outputting a dataset.
    split_names : list of str
        List of split names.
    batch_size : int
        Size of the batch.
    n_workers : int
        Number of workers

    Returns
    -------
    dict
        Dict keyed by split names and valued by torch.utils.data.DataLoader
    """

    loaders = {}
    for split_name in split_names:
        shuffle = split_name == 'train'
        loaders[split_name] = torch.utils.data.DataLoader(
            load_dataset(split_name=split_name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
        )

    return loaders


