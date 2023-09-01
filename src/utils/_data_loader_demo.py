import sys

sys.path.append("..")
import numpy as np
from data_loader import *

# set the queries folder
queries_folder = "/mnt/storage15/TI-2016/npy/tokenized/trivial/train"

# choose a strategy. for now, fixed and cluster are supported
strategy = ClusterSequencingStrategy()
# strategy = FixedSequencingStrategy()

# create a generator using that strategy
generator = SequenceGenerator(queries_folder, strategy, 32, False, True)

# the strategy can also be changed later with set_strategy()
# generator.set_strategy(FixedSequencingStrategy())


# ~~~~~ Ways to use this functionality ~~~~~

# First way: directly create sequences from a specific array of queries using a strategy. Useful for debugging.
# queries = np.load(
#     os.path.join(queries_folder, "20160423_235403.npy"), allow_pickle=True
# )
# seqs = strategy.make_sequences(
#     queries, seqlen=32, include_class=False, group_by_host=True
# )
# print(seqs)

# Second way: call the generator object (it is a callable) to obtain an iterator. Useful for the tf.Dataset API.
# g = generator()
# print(next(g))
# print(next(g))
# print(next(g))

# Third way: use the generator as a normal iterator
# for sequence in generator:
#     print(sequence)
