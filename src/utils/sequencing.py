import sys
import numpy as np
from sklearn.cluster import DBSCAN


def get_clusters_from_timestamp(
    queries, eps=lambda deltas: np.percentile(deltas, 50)
):
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
    deltas = queries[1:, -1] - queries[:-1, -1]
    avg_delta = np.mean(deltas)
    try:
        std_delta = np.std(deltas)
    # TODO is this a legit use case? a host with a single query? is it supposed to happen?
    except ZeroDivisionError as e:
        print(
            f"[WARN] Found a host with a single query: {queries}. Returning [-1]."
        )
        return [-1]

    dbscan = DBSCAN(eps=eps(deltas)).fit(np.expand_dims(queries[:, -1], -1))
    return dbscan.labels_


def pad(sequence, to_len, token="<PAD>"):

    pad_width = to_len - len(sequence)
    sequence = np.pad(
        sequence,
        ((0, pad_width), (0, 0)),
        constant_values=token,
    )

    # If sequence has 3 columns (includes class), then the class for <PAD> is 0
    if np.shape(sequence)[1] == 3:
        sequence[-pad_width:] = [token, token, 0]

    return sequence


if __name__ == "__main__":
    seqlen = 32
    queries = np.load(
        "/mnt/storage15/TI-2016/npy/tokenized/trivial/train/20160423_235403.npy",
        allow_pickle=True,
    )

    # deltas = queries[1:, -1] - queries[:-1, -1]
    # avg_delta = np.mean(deltas)
    # std_delta = np.std(deltas)
    # print("Avg:", avg_delta)
    # print("Std:", std_delta)
    # for q in range(10, 100, 10):
    #     print(f"Delta {q}-percentile: {np.percentile(deltas, q)}")

    # dbscan = DBSCAN(eps=np.percentile(deltas, 50)).fit(
    #     np.expand_dims(queries[:, -1], -1)
    # )

    # print(dbscan.labels_)
    # print(deltas)
    # print("Max delta:", np.max(deltas))
    # print("Avg:", avg_delta)
    # print("Std:", std_delta)

    # clusters = get_clusters_from_timestamp(queries)

    # for q, query in enumerate(queries):
    #     print(f"{query[2]} [{clusters[q]}]: {query[1]} ({query[0]})")

    queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]
    seqs = []

    for host in np.unique(queries[:, 0]):  # for each unique host

        # get queries made by the current host
        host_queries = queries[np.where(queries[:, 0] == host)[0]]

        # get cluster labels of that host's queries
        host_cluster_labels = get_clusters_from_timestamp(host_queries)

        # from each of those clusters we are going to make a sequence
        for c in range(np.max(host_cluster_labels) + 1):

            cluster = host_queries[
                np.where(host_cluster_labels == c)
            ]  # get queries associated to the current cluster label

            # cluster = cluster[:, :-1]  # remove timestamp once it's no longer needed

            # take first element of the domain (the domain is a list with always one token)
            # TODO this is only ok for the trivial tokenizer
            cluster = [[q[0], q[1][0]] for q in cluster]

            # truncate clusters (sequences) longer than seqlen, moving the excess to a new sequence
            while len(cluster) > seqlen:
                print(
                    f"[INFO] Truncating sequence with cluster ID {c} for host {host}: length of {len(cluster)} exceeds seqlen of {seqlen}."
                )
                truncated_cluster = cluster[:seqlen]

                seqs.append(truncated_cluster)
                cluster = cluster[seqlen:]

            # this sequence will not be full, so we have to pad it to seqlen
            seqs.append(pad(cluster, seqlen))

    for s in seqs:
        print(s)
