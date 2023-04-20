import numpy as np
import os
import argparse
import re
from tqdm import tqdm


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--queries_folder",
    default="/mnt/storage/TI-2016-preprocessed/arrays/all/queries/train",
)
argparser.add_argument("--output_folder", default="vocabs/all/exp/")
args = argparser.parse_args()

hosts, domains = [[], []], [[], []]

count = 0
for f in (pbar := tqdm(os.listdir(args.queries_folder))):
    count += 1
    if count > 2:
        break

    if not f.startswith("queries-"):  # skip irrelevant files
        continue
    q = np.load(os.path.join(args.queries_folder, f), allow_pickle=True)
    pbar.set_description(f"{f} loaded")

    new_hosts = q[:, 0]
    new_domains = q[:, 1]
    new_hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), new_hosts))
    new_domains = list(map(lambda d: re.sub("[^!-~\\n]+", "", d), new_domains))

    new_hosts = np.unique(new_hosts, return_counts=True)
    new_domains = np.unique(new_domains, return_counts=True)
    print(new_hosts)
    for h in new_hosts[0]:
        if h in hosts[0]:
            hosts[1][np.argwhere(hosts[0] == h)] += new_hosts[1][
                np.argwhere(new_hosts[0] == h)
            ]
            # hosts[1][hosts[0].index(h)] += new_hosts[1][new_hosts[0].index(h)]
        else:
            hosts[0].append(h)
            hosts[1].append(new_hosts[1][np.argwhere(new_hosts[0] == h)])
            # hosts[1].append(
            #     new_hosts[1][new_hosts[0].index(h)]
            # )
    for d in new_domains[0]:
        if d in domains[0]:
            domains[1][np.argwhere(domains[0] == d)] += new_domains[1][
                np.argwhere(new_domains[0] == d)
            ]
            # domains[1][domains[0].index(d)] += new_domains[1][new_domains[0].index(d)]
        else:
            domains[0].append(d)
            domains[1].append(new_domains[1][np.argwhere(new_domains[0] == d)])
            # domains[1].append(new_domains[1][new_domains[0].index(d)])

print(hosts[:, :10])
print(domains[:, :10])
sorter = np.argsort(hosts[1])[::-1]
hosts[0] = hosts[0][sorter]
hosts[1] = hosts[1][sorter]
print("Hosts processed")

sorter = np.argsort(domains[1])[::-1]
domains[0] = domains[0][sorter]
domains[1] = domains[1][sorter]
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
