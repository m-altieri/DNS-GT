import numpy as np
import os
import matplotlib.pyplot as plt

# Initialize
domains_file = "../preprocessing/vocabs/all/domains_vocab.txt"
with open(domains_file, "r") as f:
    domains = f.read().split("\n")
domains = {d: 0 for d in domains}
print("Domains loaded.")

# Count
queries_folder_train = "../preprocessing/arrays/all/queries/train"
for query_file in os.listdir(queries_folder_train):
    queries = np.load(os.path.join(queries_folder_train, query_file))
    for q in queries:
        if q[1] in domains:
            domains[q[1]] += 1
    print(f"Finished analyzing {query_file}")
queries_folder_test = "../preprocessing/arrays/all/queries/test"
for query_file in os.listdir(queries_folder_test):
    queries = np.load(os.path.join(queries_folder_test, query_file))
    for q in queries:
        if q[1] in domains:
            domains[q[1]] += 1
    print(f"Finished analyzing {query_file}")

# Sort
domains = sorted(domains.items(), key=lambda item: item[1], reverse=True)

# Save
with open("domains_occurrences.txt", "w") as f:
    f.write("\n".join([f"{domains[d[0]]} {d[0]}" for d in domains]))
