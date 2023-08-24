import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../utils")
from tokenizers import TrivialTokenizer, SubdomainTokenizer


argparser = argparse.ArgumentParser()
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
argparser.add_argument(
    "--save-timestamp",
    action="store_true",
    help="Add a timestamp column in addition to host and domain, to the final npy. For now, only has effect if not --tokenize.",
)
args = argparser.parse_args()

# -- Data
data_path = "/mnt/storage15/TI-2016/pcsv"
output_path = "/mnt/storage15/TI-2016/npy/tokenized/" + args.tokenizer
if not os.path.exists(output_path):
    os.makedirs(output_path)


tokenizer = {
    "trivial": TrivialTokenizer(),
    "subdomain": SubdomainTokenizer(max_tokens=30000),
}[args.tokenizer]

if args.vocab:
    print("Loading vocabulary...")
    tokenizer.load_domain_vocabulary(os.path.join(output_path, "domain_vocab.txt"))
    tokenizer.load_host_vocabulary(os.path.join(output_path, "host_vocab.txt"))
else:
    # If the vocabulary is not loaded, first I have to create it by looping over the whole data
    print("Creating vocabulary from scratch...")
    for filename in tqdm(os.listdir(data_path)):
        df = pd.read_csv(os.path.join(data_path, filename), sep=";")
        df = df[["ip_src", "qry_name"]]
        queries = df.to_numpy()
        tokenizer.fit(queries)
    tokenizer.save_domain_vocabulary(os.path.join(output_path, "domain_vocab.txt"))
    tokenizer.save_host_vocabulary(os.path.join(output_path, "host_vocab.txt"))

# Then I have to loop again and process one file at a time
print("Processing queries...")
for filename in tqdm(os.listdir(data_path)):
    df = pd.read_csv(os.path.join(data_path, filename), sep=";")
    columns = (
        ["ip_src", "qry_name", "timestamp"]
        if args.save_timestamp
        else ["ip_src", "qry_name"]
    )
    df = df[columns]
    queries = df.to_numpy()
    for q, query in enumerate(queries):
        queries[q, 1] = np.array(tokenizer.tokenize(query[1]))
    print(queries)
    np.save(os.path.join(output_path, f"{os.path.splitext(filename)[0]}.npy"), queries)
