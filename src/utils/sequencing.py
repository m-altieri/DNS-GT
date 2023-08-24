import sys
import numpy as np
from sklearn.cluster import DBSCAN


def get_clusters_from_timestamp(queries, eps=lambda deltas: np.percentile(deltas, 50)):
    """Get a list of labels mapping each item in the input array to a cluster.
    Uses the DBSCAN density-based clustering algorithm.

    Args:
        queries (numpy array): A numpy 2D array containing timestamped items.
        The timestamp is assumed to be in the last column.
        eps (func): Used to change the method eps is calculated in DBSCAN
        (which determines the maximum distance for two points to be neighbors).
        By default it is the 50-percentile of the subsequent timestamp differences.
        The previous default was `lambda deltas: np.mean(deltas) + 3 * np.std(deltas)`.
    """
    # queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]

    deltas = queries[1:, -1] - queries[:-1, -1]
    avg_delta = np.mean(deltas)
    try:  # TODO remove the try-except block once you figure out the bug
        std_delta = np.std(deltas)
    except ZeroDivisionError as e:
        print(e)
        print(deltas)
        print(queries)
        if np.isnan(avg_delta):
            avg_delta = 0.0
        std_delta = 0.0
        return [-1]
    # print("Avg:", avg_delta)
    # print("Std:", std_delta)

    dbscan = DBSCAN(eps=eps(deltas)).fit(np.expand_dims(queries[:, -1], -1))
    # dbscan = DBSCAN(eps=avg_delta + 2 * std_delta).fit(
    #     np.expand_dims(queries[:, -1], -1)
    # )
    # print(dbscan.labels_)
    return dbscan.labels_


def pad(sequence, to_len, token="<PAD>"):
    return np.pad(
        sequence,
        ((0, to_len - len(sequence)), (0, 0)),
        constant_values=token,
    )


if __name__ == "__main__":
    queries = np.load(
        "/mnt/storage15/TI-2016/npy/tokenized/trivial/train/20160424_075411.npy",
        allow_pickle=True,
    )
    queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]

    deltas = queries[1:, -1] - queries[:-1, -1]
    avg_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    print("Avg:", avg_delta)
    print("Std:", std_delta)
    for q in range(10, 100, 10):
        print(f"Delta {q}-percentile: {np.percentile(deltas, q)}")
    dbscan = DBSCAN(eps=np.percentile(deltas, 50)).fit(
        np.expand_dims(queries[:, -1], -1)
    )

    # dbscan = DBSCAN(eps=avg_delta + 1 * std_delta).fit(
    #     np.expand_dims(queries[:, -1], -1)
    # )
    print(dbscan.labels_)
    print(deltas)
    print("Max delta:", np.max(deltas))
    print("Avg:", avg_delta)
    print("Std:", std_delta)

    last_ts = queries[0, -1]
    block_size, block_sizes = 0, []
    for q, query in enumerate(queries):
        if query[-1] - last_ts > avg_delta + 3 * std_delta:
            print("Block size:", block_size)
            print("\n")
            block_sizes.append(block_size)
            block_size = 0

        print(f"{query[2]} [{dbscan.labels_[q]}]: {query[1]} ({query[0]})")
        block_size += 1
        last_ts = query[-1]
    print(dbscan.labels_)

    print("Block size:", block_size)
    print("\n")

    print("# blocks:", len(block_sizes))
    print("Max block size:", np.max(block_sizes))
    print("Avg block size:", np.mean(block_sizes))
    print("Std block size:", np.std(block_sizes))
