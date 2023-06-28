import numpy as np
import os
import argparse
import re
from tqdm import tqdm
from colorama import Fore, Style


# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--queries-folder",
    default="/mnt/storage15/TI-2016-preprocessed/arrays/all/queries/train",
)
argparser.add_argument("--output-folder", default="vocabs/all/exp/")
args = argparser.parse_args()

# initialize hosts and domains
hosts, domains = ([], []), ([], [])

# for each file in the queries folder
c = 0
for f in (pbar := tqdm(os.listdir(args.queries_folder))):
    c += 1

    # skip irrelevant files
    if not f.startswith("queries-"):
        continue

    # load the file
    q = np.load(os.path.join(args.queries_folder, f), allow_pickle=True)
    pbar.set_description(
        f"[File {c}] ({len(q):,} new queries, {np.sum(hosts[1]):,.0f} processed), "
        + f"total unique: ({len(hosts[0]):,} hosts / {len(domains[0]):,} domains)"
    )

    # get hosts and domains
    new_hosts = q[:, 0]
    new_domains = q[:, 1]

    # remove non-ascii characters
    new_hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), new_hosts))
    new_domains = list(map(lambda d: re.sub("[^!-~\\n]+", "", d), new_domains))

    # get counts of hosts and domains
    new_hosts = np.unique(new_hosts, return_counts=True)
    new_domains = np.unique(new_domains, return_counts=True)

    # add new hosts and domains to the global list, or update their count if the host/domain already exists
    for h in np.argwhere(np.isin(hosts[0], new_hosts[0])):
        in_idx = np.argwhere(new_hosts[0] == hosts[0][h]).flatten()
        hosts[1][h] += new_hosts[1][in_idx]
    out_idx = np.argwhere(~np.isin(new_hosts[0], hosts[0])).flatten()
    hosts = (
        np.concatenate((hosts[0], new_hosts[0][out_idx])),
        np.concatenate((hosts[1], new_hosts[1][out_idx])),
    )
    for d in np.argwhere(np.isin(domains[0], new_domains[0])):
        in_idx = np.argwhere(new_domains[0] == domains[0][d]).flatten()
        domains[1][d] += new_domains[1][in_idx]
    out_idx = np.argwhere(~np.isin(new_domains[0], domains[0])).flatten()
    domains = (
        np.concatenate((domains[0], new_domains[0][out_idx])),
        np.concatenate((domains[1], new_domains[1][out_idx])),
    )

# sort hosts and domains by count
sorter = np.argsort(hosts[1])[::-1]
hosts = (hosts[0][sorter], hosts[1][sorter])
sorter = np.argsort(domains[1])[::-1]
domains = (domains[0][sorter], domains[1][sorter])

# create the vocabularies
hosts_vocab = "\n".join(hosts[0])
domains_vocab = "\n".join(domains[0])

# save the vocabularies
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
hosts_vocab_path = os.path.join(args.output_folder, "hosts_vocab.txt")
with open(
    hosts_vocab_path, "w"
) as f:  # not adding <UNK> as it will be added by the model
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write(hosts_vocab)
print("Hosts vocab saved in", hosts_vocab_path)
domains_vocab_path = os.path.join(args.output_folder, "domains_vocab.txt")
with open(
    domains_vocab_path, "w"
) as f:  # not adding <UNK> as it will be added by the model
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write(domains_vocab)
print("Domains vocab saved in", domains_vocab_path)
