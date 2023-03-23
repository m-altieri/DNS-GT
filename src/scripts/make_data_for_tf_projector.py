import numpy as np


# Load
embeddings_path = "../embeddings/embeddings-increasingmasking.npy"
embeddings = np.load(embeddings_path)
domains_path = "../preprocessing/vocabs/small/domains_vocab.txt"
domains = None
with open(domains_path, "r") as f:
    domains = np.array(f.read().split("\n"))

# embeddings = embeddings[:2]

tsv_embeddings = ""
tsv_metadata = ""
for index, emb in enumerate(embeddings):
    tsv_embeddings += "\t".join(emb.astype(str)) + "\n"
    # for feature in emb:
    #     tsv_embeddings += str(feature) + "\t"
    # tsv_embeddings = tsv_embeddings[:-2] + "\n"
    tsv_metadata += domains[index] + "\n"
    if index % 1000 == 0:
        print(index)
# print(tsv_embeddings)
# print(tsv_metadata)

with open("output_embeddings_for_projector.tsv", "w") as f:
    f.write(tsv_embeddings)

with open("output_metadata_for_projector.tsv", "w") as f:
    f.write(tsv_metadata)
