#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create test folds."""

import os
import argparse
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument("path", help="Path to the main data folder.")
argparser.add_argument(
    "-p",
    "--partitions",
    default=10,
    type=int,
    help="Number of different fold partitions.",
)
argparser.add_argument(
    "-f",
    "--folds",
    default=5,
    type=int,
    help="Number of test folds for each partition.",
)
args = argparser.parse_args()

# Load domain
with open(os.path.join(args.path, "vocab", "domains_vocab.txt"), "r") as f:
    domains = f.read().splitlines()

# Remove special domains
domains = domains[:-3]

rng = np.random.default_rng(seed=42)

for partition in range(args.partitions):
    # Create folder for the current partition
    partition_path = os.path.join(args.path, "test_folds", f"partition-{partition}")
    if not os.path.exists(partition_path):
        os.makedirs(partition_path)

    # Randomly rearrange the domains to partition them in equally sized folds
    shuffled_domains = rng.permutation(domains)

    for fold in range(args.folds):
        # Slice the domains for the current fold
        in_fold = shuffled_domains[
            len(domains) // args.folds * fold : len(domains) // args.folds * (fold + 1)
        ]

        # Save current fold
        np.save(os.path.join(partition_path, f"fold-{fold}.npy"), in_fold)
