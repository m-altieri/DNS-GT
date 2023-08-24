# -*- coding: utf-8 -*-
"""Creation of a pytorch dataset for MLSEC

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""

import numpy as np
import torch


def create_PEfile_from_instance(instance):
    """Create a PE file from an instance.

    Parameters
    ----------
    instance : torch.Tensor
        Instance.

    Returns
    -------
    list of bytes
    """
    return np.array(instance, dtype=np.uint8).tobytes()


class MLSEC(torch.utils.data.Dataset):
    """MLSEC dataset.

    Load MLSEC examples from a given path. Samples should be named between
    '0XX' with XX ranging from '01' to '50'.

    Parameters
    ----------
    data_path : pathlib.Path or str
        Path to the samples.
    """

    classes = ['Goodware', 'Malware']

    def __init__(self, data_path, raw=False, transform=None):
        self.data_path = data_path / 'MLSEC'
        self.n_samples = 50
        self.raw = raw

        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        if idx < self.n_samples:

            sample_path = self.data_path / '{:03d}'.format(idx + 1)

            with open(sample_path, 'rb') as infile:
                instance = infile.read()

            if not self.raw:
                instance = torch.from_numpy(
                    np.frombuffer(instance, dtype=np.uint8)).long()
                if self.transform:
                    instance = self.transform(instance)

        else:
            err_msg = 'Sample {:03d} does not exist.'
            raise IndexError(err_msg.format(idx + 1))

        return instance
