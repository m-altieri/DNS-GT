#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversion of preprocessed CSV files into NPY arrays ready for 
the training and evaluation pipeline.
"""


import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from tokenizers import TrivialTokenizer, SubdomainTokenizer

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "path",
    help="Path to the folder containing the preprocessed CSV files.",
)
argparser.add_argument(
    "-v",
    "--vocab",
    action="store",
    nargs="?",
    const="vocab.txt",
    help="Vocabulary path. If unspecified, it will use the default path. If omitted, a new vocabulary will be created.",
)
argparser.add_argument(
    "--tokenizer",
    choices=["trivial", "subdomain"],
    default="trivial",
    help="Choose the tokenizer to use to tokenize domain names. The default is TrivialTokenizer (does not actually tokenize).",
)
args = argparser.parse_args()

# -- Data
pcsv_path = os.path.join(args.path, "pcsv")
output_path = os.path.join(args.path, "npy")
if not os.path.exists(output_path):
    os.makedirs(output_path)


tokenizer = {
    "trivial": TrivialTokenizer(),
    "subdomain": SubdomainTokenizer(max_tokens=30000),
}[args.tokenizer]

if args.vocab:
    print("Loading vocabulary...")
    tokenizer.load_domain_vocabulary(os.path.join(args.vocab, "domains_vocab.txt"))
    tokenizer.load_host_vocabulary(os.path.join(args.vocab, "hosts_vocab.txt"))
else:
    # If the vocabulary is not loaded, first I have to create it by looping over the whole data
    print("Creating vocabulary from scratch...")
    for filename in tqdm(os.listdir(pcsv_path)):
        df = pd.read_csv(os.path.join(pcsv_path, filename), sep=";")
        df = df[["ip_src", "qry_name"]]
        queries = df.to_numpy()
        tokenizer.fit(queries)
    tokenizer.save_domain_vocabulary(
        os.path.join(args.path, "vocab", "domains_vocab.txt")
    )
    tokenizer.save_host_vocabulary(os.path.join(args.path, "vocab", "hosts_vocab.txt"))

# Then I have to loop again and process one file at a time
print("Processing queries...")
for filename in tqdm(os.listdir(pcsv_path)):
    df = pd.read_csv(os.path.join(pcsv_path, filename), sep=";")
    df = df[["ip_src", "qry_name", "timestamp"]]
    queries = df.to_numpy()
    for q, query in enumerate(queries):
        queries[q, 1] = np.array(tokenizer.tokenize(query[1]))

    np.save(os.path.join(output_path, f"{os.path.splitext(filename)[0]}.npy"), queries)

# Split query files into train and test
if not os.path.exists(os.path.join(output_path, "train")):
    os.makedirs(os.path.join(output_path, "train"))

if not os.path.exists(os.path.join(output_path, "test")):
    os.makedirs(os.path.join(output_path, "test"))

# Take 70% of files for train and 30% for test
npy_files = [
    f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))
]
npy_files.sort()
train_split = int(len(npy_files) * 0.7)

for f in npy_files[:train_split]:
    os.rename(os.path.join(output_path, f), os.path.join(output_path, "train", f))

for f in npy_files[train_split:]:
    os.rename(os.path.join(output_path, f), os.path.join(output_path, "test", f))
