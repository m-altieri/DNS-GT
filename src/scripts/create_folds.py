import os
import numpy as np

# Set constants
PARTITIONS = 10  # number of different fold partitions
FOLDS_PER_PARTITIONS = 5  # number of folds for each partition

# Load domain
with open(
    "/mnt/storage15/TI-2016/npy/tokenized/trivial/domain_vocab.txt", "r"
) as f:
    domains = f.read().splitlines()

# Remove special domains
domains = domains[:-3]


for partition in range(PARTITIONS):

    # Create folder for the current PARTITIONS
    if not os.path.exists(
        f"/mnt/storage15/TI-2016/npy/tokenized/trivial/folds/partition-{partition}"
    ):
        os.makedirs(
            f"/mnt/storage15/TI-2016/npy/tokenized/trivial/folds/partition-{partition}"
        )

    # Randomly rearrange the domains to partition them in equally sized folds
    shuffled_domains = np.random.permutation(domains)

    for fold in range(FOLDS_PER_PARTITIONS):

        # Slice the domains for the current fold
        in_fold = shuffled_domains[
            len(domains)
            // FOLDS_PER_PARTITIONS
            * fold : len(domains)
            // FOLDS_PER_PARTITIONS
            * (fold + 1)
        ]

        # Save current fold
        np.save(
            f"/mnt/storage15/TI-2016/npy/tokenized/trivial/folds/partition-{partition}/fold-{fold}.npy",
            in_fold,
        )
