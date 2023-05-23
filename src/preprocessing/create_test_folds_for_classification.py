import numpy as np
import os

vocab_path = "../../data/vocabs/small/exp/domains_vocab.txt"
output_path = "../../data/vocabs/small/exp/test_folds"

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(vocab_path, "r") as f:
    vocab = [l.strip() for l in f.readlines()]

np.random.shuffle(vocab)  # in-place

folds = 20
for k in range(folds):
    fold = vocab[len(vocab) // folds * k : len(vocab) // folds * (k + 1)]
    np.save(os.path.join(output_path, f"fold-{k}.npy"), fold)
