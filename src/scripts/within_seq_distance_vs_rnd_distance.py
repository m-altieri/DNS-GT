import sys

sys.path.append("..")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_loader import SequenceGenerator, ClusterSequencingStrategy


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Load
embeddings_path = "../../runs/DELM/DELM-L32/embeddings.npy"
embeddings = np.load(embeddings_path)
domains_path = "/mnt/storage15/TI-2016/npy/tokenized/trivial/domain_vocab.txt"
domains = None
with open(domains_path, "r") as f:
    domains = np.array(f.read().split("\n"))


queries_path = f"/mnt/storage15/TI-2016/npy/tokenized/trivial/train"
train = tf.data.Dataset.from_generator(
    SequenceGenerator(
        queries_path, ClusterSequencingStrategy(), 32, False, True
    ),
    output_signature=tf.TensorSpec(shape=(32, 2), dtype=tf.string),
)
train = train.batch(256).prefetch(tf.data.AUTOTUNE)

n_seqs = 200
unique = False

# Real sequences
seqs = (
    train.unbatch().take(n_seqs).as_numpy_iterator()
)  # .skip(np.random.randint(0, 1000)) prima di take
seqs = np.array([s for s in seqs], dtype=object)
seqs = seqs[..., 1]
seqs = seqs.astype(str)

# Random sequences
random_seqs = np.random.choice(domains, (n_seqs, 32))
avg_dists_real = np.zeros((len(seqs)))
avg_dists_random = np.zeros(len(random_seqs))

# Avg distance for real seqs
for s, seq in enumerate(seqs):
    dist = 0.0

    # Remove [UNK] tokens because they have no associated embedding
    seq = [d for d in seq if d != "[UNK]" and d != "<UNK>"]

    # If unique, remove duplicated domains. If there are duplicated domains, it's obviously easier to have a lower average distance
    if unique:
        seq = list(set(seq))

    seq_embeddings = embeddings[[np.where(domains == s)[0][0] for s in seq]]
    for _, emb_i in enumerate(seq_embeddings):
        for _, emb_j in enumerate(seq_embeddings):
            dist += eucl_dist(emb_i, emb_j)
    avg_dists_real[s] = dist / (len(seq) ** 2)
    print(f"Seq {s}: {avg_dists_real[s]}")
print(f"Avg: {np.mean(avg_dists_real)}")

# Avg distance for random seqs
for s, seq in enumerate(random_seqs):
    dist = 0.0
    seq_embeddings = embeddings[[np.where(domains == s)[0][0] for s in seq]]
    for _, emb_i in enumerate(seq_embeddings):
        for _, emb_j in enumerate(seq_embeddings):
            dist += eucl_dist(emb_i, emb_j)
    avg_dists_random[s] = dist / (len(seq) ** 2)
    print(f"Seq {s}: {avg_dists_random[s]}")
print(f"Avg: {np.mean(avg_dists_random)}")

if True:
    avg_dists_real = np.sort(avg_dists_real)
    avg_dists_random = np.sort(avg_dists_random)

# Plot
fig = plt.figure(
    figsize=(
        5,
        2.5,
    )
)
# fig.suptitle("Euclidean distance")
plt.plot(avg_dists_real, label="Real sequences")
plt.plot(avg_dists_random, label="Random sequences")
plt.xlabel("Sequences")
plt.ylabel("Euclidean distance")
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.legend(loc="lower right", fontsize="small")

plt.savefig("../../runs/_extras/DELM/avg_dists.png")
