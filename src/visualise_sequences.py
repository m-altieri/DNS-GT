"""Visualisation of the sequencing strategies."""

from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

from utils.data_loading import (
    SequenceGenerator,
    TimeWindowStrategy,
    FixedSequencingStrategy,
    ClusterSequencingStrategy,
)


# data_path = Path("/mnt/storage15/rhTI2016/")
data_path = Path("/mnt/storage15/TI-2016/")

if __name__ == "__main__":

    sequencing_strategies = {
        "fixed": FixedSequencingStrategy(),
        "time": TimeWindowStrategy(),
        "cluster": ClusterSequencingStrategy(),
    }

    sequencing_strategy = sequencing_strategies["fixed"]

    train = tf.data.Dataset.from_generator(
        SequenceGenerator(
            data_path / "npy" / "train",
            sequencing_strategy,
            seqlen=32,
            task=None,
            include_class=False,
            group_by_host=True,
            stride=1,
            include_start=False,
        ),
        output_signature=tf.TensorSpec(
            shape=[32, 2],
            dtype=tf.string,
        ),
    )

    train = train.batch(8).prefetch(tf.data.AUTOTUNE)


    pbar = tqdm(train, total=10)
    for x in pbar:
        pass
