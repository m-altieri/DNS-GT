import numpy as np
import os
import argparse
import re
import sys


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--hosts_folder",
    default="arrays/all/hosts/",
    help="Folder containing the hosts arrays",
)
argparser.add_argument(
    "--domains_folder",
    default="arrays/all/domains/",
    help="Folder containing the domains arrays",
)
argparser.add_argument("--queries_folder", default="arrays/all/queries/train")
argparser.add_argument(
    "--output_folder",
    default="vocabs/all/exp/",
    help="Folder to output the vocabularies to",
)
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

hosts_vocab = "\n".join(
    hosts[0]
)  # .replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()
domains_vocab = "\n".join(
    domains[0]
)  # .replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

hosts_vocab_path = os.path.join(args.output_folder, "hosts_vocab.txt")
with open(hosts_vocab_path, "w") as f:
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write("<UNK>\n")
    f.write(hosts_vocab)
print("Hosts vocab saved in", hosts_vocab_path)

domains_vocab_path = os.path.join(args.output_folder, "domains_vocab.txt")
with open(domains_vocab_path, "w") as f:
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write("<UNK>\n")
    f.write(domains_vocab)
print("Domains vocab saved in", domains_vocab_path)

sys.exit(0)

for f in os.listdir(args.hosts_folder):
    new_hosts = np.load(
        os.path.join(args.hosts_folder, f), allow_pickle=True, encoding="ASCII"
    )
    hosts = np.concatenate([hosts, new_hosts])
    print(f"{f} loaded")

hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), hosts))

hosts = np.unique(hosts, return_counts=True)
sorter = np.argsort(hosts[1])[::-1]
hosts[0][:] = hosts[0][sorter]
hosts[1][:] = hosts[1][sorter]
print("Hosts processed")

# for f in os.listdir(args.hosts_folder):
#     fpath = os.path.join(args.hosts_folder, f)
#     new_hosts = np.load(fpath, allow_pickle=True, encoding="ASCII")
#     print(len(new_hosts))
#     new_hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), new_hosts))
#     new_hosts = np.unique(new_hosts, return_counts=True)
#     sorter = np.argsort(new_hosts[1][:])
#     new_hosts[0][:] = new_hosts[0][sorter]
#     new_hosts[1][:] = new_hosts[1][sorter]
#     print(new_hosts)
#     print(np.max(new_hosts[1]))
#     print(len(new_hosts[0]))

#     continue
#     sys.exit()

#     new_hosts = list(map(lambda h: re.sub("[^!-~\\n]+", "", h), new_hosts))
#     hosts = np.unique(np.concatenate((hosts, new_hosts)))
# hosts = np.unique(hosts)

for f in os.listdir(args.domains_folder):
    new_domains = np.load(
        os.path.join(args.domains_folder, f), allow_pickle=True, encoding="ASCII"
    )
    domains = np.concatenate([domains, new_domains])
    print(f"{f} loaded")

domains = list(map(lambda d: re.sub("[^!-~\\n]+", "", d), domains))

domains = np.unique(domains, return_counts=True)
sorter = np.argsort(domains[1])[::-1]
domains[0][:] = domains[0][sorter]
domains[1][:] = domains[1][sorter]
print("Domains processed")

hosts_vocab = "\n".join(
    hosts[0]
)  # .replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()
domains_vocab = "\n".join(
    domains[0]
)  # .replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()

# hosts_vocab = re.sub('[^!-~\\n]+', '', hosts_vocab)
# domains_vocab = re.sub('[^!-~\\n]+', '', domains_vocab)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

with open(os.path.join(args.output_folder, "hosts_vocab.txt"), "w") as f:
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write("<UNK>\n")
    f.write(hosts_vocab)
print("Hosts vocab saved")

with open(os.path.join(args.output_folder, "domains_vocab.txt"), "w") as f:
    f.write("<START>\n")
    f.write("<PAD>\n")
    f.write("<MASK>\n")
    f.write("<UNK>\n")
    f.write(domains_vocab)
print("Domains vocab saved")
