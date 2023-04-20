import numpy as np
import os
import argparse
import re


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--queries_folder",
    default="/mnt/storage/TI-2016-preprocessed/arrays/all/queries/train",
)
argparser.add_argument("--output_folder", default="vocabs/all/exp/")
args = argparser.parse_args()

hosts, domains = [], []

for f in os.listdir(args.queries_folder):
    if f == "count.py":
        continue
    q = np.load(os.path.join(args.queries_folder, f), allow_pickle=True)
    hosts = np.concatenate([hosts, q[:, 0]])
    domains = np.concatenate([domains, q[:, 1]])
    print(f"{f} loaded")

hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), hosts))
hosts = np.unique(hosts, return_counts=True)
sorter = np.argsort(hosts[1])[::-1]
hosts[0][:] = hosts[0][sorter]
hosts[1][:] = hosts[1][sorter]
print("Hosts processed")

domains = list(map(lambda d: re.sub("[^!-~\\n]+", "", d), domains))
domains = np.unique(domains, return_counts=True)
sorter = np.argsort(domains[1])[::-1]
domains[0][:] = domains[0][sorter]
domains[1][:] = domains[1][sorter]
print("Domains processed")

hosts_vocab = "\n".join(hosts[0])
domains_vocab = "\n".join(domains[0])

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
