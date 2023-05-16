import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

n_seqs = 250


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Load
embeddings_path = "../embeddings/embeddings-increasingmasking.npy"
embeddings = np.load(embeddings_path)
domains_path = "../preprocessing/vocabs/small/domains_vocab.txt"
domains = None
with open(domains_path, "r") as f:
    domains = np.array(f.read().split("\n"))


def seq_generator_from_folder(
    input_folder, seqlen, stride=1, include_start=False, group_hosts=True
):
    """Folder containing .npy files, each representing a matrix of shape (n_queries, 2)."""
    for f in os.listdir(input_folder):
        seqs = create_sequences(
            os.path.join(input_folder, f), seqlen, stride, include_start, group_hosts
        )
        for seq in seqs:
            yield seq


def create_sequences(
    input_file, seqlen, stride=1, include_start=False, group_hosts=True
):
    # input [queries, 2]
    # output [queries - stride, seqlen, 2]

    actual_seqlen = seqlen - include_start

    queries = np.load(input_file)
    if group_hosts:
        queries = queries[
            np.argsort(queries[:, 0])
        ]  # Sort queries by host, preserving row structure

    seqs = np.empty(
        shape=((len(queries) - actual_seqlen) // stride + 1, seqlen, 2), dtype=object
    )

    for i in range(len(seqs)):
        if include_start:
            seqs[i][0] = ["<START>", "<START>"]
        seqs[i][include_start:] = queries[i * stride : i * stride + actual_seqlen]

    return seqs


queries_path = f"../preprocessing/arrays/small/queries/"
domains_vocab_path = f"../preprocessing/vocabs/small/domains_vocab.txt"
train = tf.data.Dataset.from_generator(
    lambda: seq_generator_from_folder(
        os.path.join(queries_path, "train"),
        stride=1,
        seqlen=32,
        include_start=False,
        group_hosts=True,
    ),
    output_signature=tf.TensorSpec(shape=(32, 2), dtype=tf.string),
)

train = train.batch(256).prefetch(tf.data.AUTOTUNE)
with open(domains_vocab_path, "r") as f:
    domains_vocab = [l.strip() for l in f.readlines()]

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
plt.legend()
# plt.xticks([])
# plt.yticks([0.95, 1.00, 1.05, 1.10])

plt.savefig("avg_dists.png")
