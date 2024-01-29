"""Visualisation of the sequencing strategies."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.data_loading import (
    SequenceGenerator,
    TimeWindowStrategy,
    FixedSequencingStrategy,
    ClusterSequencingStrategy,
)

queries_folder = Path("/mnt/storage15/rhTI2016/npy/train")

strategy = ClusterSequencingStrategy()
# strategy = FixedSequencingStrategy()

sequencing_strategies = {
    "fixed": FixedSequencingStrategy(),
    "time": TimeWindowStrategy(),
    "cluster": ClusterSequencingStrategy(),
}

sequencing_strategy = sequencing_strategies["fixed"]

generator = SequenceGenerator(input_folder=queries_folder, sequencing_strategy=sequencing_strategy, 
        task=None, seqlen=32, include_class=False, group_by_host=True)

queries = np.load(queries_folder / "20160424_135417.npy", allow_pickle=True)
seqs = strategy.make_sequences(
    queries, seqlen=32, include_class=False, group_by_host=True, with_timestamps=True
)

timestamps = [[float(qry[2]) for qry in seq if qry[2] != '0'] for seq in seqs]
