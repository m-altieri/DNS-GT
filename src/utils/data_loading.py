import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from sklearn.cluster import DBSCAN
from typing import Any, Iterator, Protocol


class SequencingStrategy(Protocol):
    @abstractmethod
    def make_sequences(
        self,
        queries: np.ndarray[Any, np.dtype[object]],
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
    ) -> np.ndarray[Any, object]:
        raise NotImplementedError()


class SequenceGenerator:
    """Create a sequence generator.
    The main use for this class is with the tf.data API.
    Specifically, a `SequenceGenerator` object is a callable which returns an iterator that provides
    input sequence during model training.

    >>> generator = SequenceGenerator(...)
    >>> g = generator()
    >>> print(next(g))

    Alternatively, it's possible to use the generator object as a normal iterator:

    >>> generator = SequenceGenerator(...)
    >>> for sequence in generator:
    ...   print(sequence)
    """

    # TODO some kwargs are not fully supported.
    def __init__(
        self,
        input_folder: str,
        sequencing_strategy: SequencingStrategy,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any,
    ) -> None:
        """Initialize a SequenceGenerator object.

        Args:
            input_folder (str): folder containing .npy files, each representing a matrix of shape `(n_queries, 2)` or `(n_queries, 3)` ,
        where columns are host, domain, and optionally timestamp.
            sequencing_strategy (SequencingStrategy): a SequencingStrategy object. Used to determine the way sequences should be created.
            seqlen (int): maximum sequence length.
            include_class (bool): used in finetuning to include the class label of each query.
            group_by_host (bool): whether each sequence should have only queries from the same host.

        Keyword args:
            stride (int): if strategy is "fixed", by how many queries to shift after each sequence.
            include_start (bool): Deprecated. whether to include a <START> token at the beginning of the sequence.
            model (str): model to use. Either "dns_gt" or "w2v".
            vocab (str): domains vocabulary. Only used in finetuning.
            tiny_amount (bool): Deprecated. Whether to use a small number of queries for debugging purposes.
            verbose (bool): whether to print additional debugging info.
        """
        self.input_folder = input_folder
        self.sequencing_strategy = sequencing_strategy
        self.seqlen = seqlen
        self.include_class = include_class
        self.group_by_host = group_by_host
        self.kwargs = kwargs

    def set_strategy(self, strategy: SequencingStrategy) -> None:
        self.sequencing_strategy = strategy

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.__next__()

    def __call__(self):
        return self.__next__()

    def __next__(self) -> Iterator[np.ndarray[Any, Any]]:
        # For each query file
        for f in os.listdir(self.input_folder):
            # Ignore file if it's not a .npy
            if os.path.splitext(os.path.join(self.input_folder, f))[-1] != ".npy":
                continue

            # Load queries from file
            queries = np.load(os.path.join(self.input_folder, f), allow_pickle=True)

            # Pair up queries with domain class if necessary
            if self.include_class:
                queries = self._add_class_to_queries(
                    queries, os.path.join(self.input_folder, "..", "..")
                )

            # Make sequences from queries with given strategy
            seqs = self.sequencing_strategy.make_sequences(
                queries,
                self.seqlen,
                self.include_class,
                self.group_by_host,
                **self.kwargs,
            )

            # Yield each sequence
            for seq in seqs:
                # print(seq)
                # print(seq.shape)
                # print(seq.dtype)
                yield seq

    @staticmethod
    def _add_class_to_queries(queries, data_path):
        # Load labels from blacklist
        labels = pd.read_csv(
            os.path.join(data_path, "labels.csv"),
            index_col=0,
            header=[0, 1],
        )
        labels.columns = pd.MultiIndex.from_tuples(
            [
                ("domain", ""),
                ("advertising", "good"),
                ("advertising", "ok"),
                ("malicious", "good"),
                ("malicious", "ok"),
                ("suspicious", "good"),
                ("suspicious", "ok"),
                ("tracking", "good"),
                ("tracking", "ok"),
                ("other", "good"),
                ("other", "ok"),
                ("any", "good"),
                ("any", "ok"),
            ]
        )

        # Only use labels for domains in embs (i.e. in the vocabulary)
        # TODO this only works for TrivialTokenizer
        with open(os.path.join(data_path, "vocab", "domains_vocab.txt"), "r") as f:
            vocab = f.read().splitlines()

        labels = labels[labels["domain"].isin(vocab)]
        labels = labels.reset_index()

        labels = labels.sort_values(
            by="domain",
            key=lambda domains: domains.map(lambda domain: vocab.index(domain)),
        )

        # take (any, good) column
        labels = labels[["domain", "any"]]
        labels = labels.to_numpy()[:, [0, 1]]  # [0, 2] for 'ok'

        # add class to each query
        sorter = np.argsort(labels[:, 0])
        idx = sorter[np.searchsorted(labels[:, 0], queries[:, 1], sorter=sorter)]
        classes = labels[idx, 1]
        queries = np.concatenate(
            [queries, np.expand_dims(classes, -1).astype(str)], axis=-1
        )

        return queries


def pad(sequence, to_len, token="<PAD>"):
    """Pad the input sequence with the specified token along axis 0 until it reaches a length of to_len.

    Args:
        sequence (array): the input sequence to pad. Must have shape (n_queries, 3) or (n_queries, 4).
        to_len (int): the total length along axis 0 of the returned array.
        token (str, optional): token to pad with. Defaults to "<PAD>". The last column is always padded with value 0.

    Returns:
        array: the padded sequence of length to_len.
    """
    pad_width = to_len - len(sequence)
    sequence = np.pad(
        sequence,
        ((0, pad_width), (0, 0)),
        constant_values=token,
    )

    # The class for the padding token is 0
    if np.shape(sequence)[1] == 3:
        sequence[-pad_width:] = [token, token, 0]
    if np.shape(sequence)[1] == 4:
        sequence[-pad_width:] = [token, token, token, 0]

    return sequence


class FixedSequencingStrategy:
    """Sequences are cut exactly every `seqlen` queries, ignoring the timestamp.
    Each query in the dataset will appear in multiple sequences, depending on the stride value.
    """

    def make_sequences(
        self,
        queries: np.ndarray,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        # get kwargs
        stride: bool = kwargs.get("stride", 1)
        include_start: bool = kwargs.get("include_start", False)

        # sort queries by host, and within each host by timestamp
        if group_by_host:
            queries = queries[np.lexsort((queries[:, 2], queries[:, 0]))]

        # if the timestamp is present, remove it
        if np.shape(queries)[-1] == 3 + include_class:
            queries = np.delete(queries, 2, axis=1)

        # if the domain is a list (tokenized), take the first element
        # TODO this only works for TrivialTokenizer
        if queries.shape[-1] == 2:
            queries = np.array([[q[0], q[1][0]] for q in queries])
        else:  # if it includes the class
            queries = np.array([[q[0], q[1][0], q[2]] for q in queries])

        # initialize sequence
        actual_seqlen = seqlen - include_start
        seqs = np.empty(
            shape=(
                (len(queries) - actual_seqlen) // stride + 1,
                seqlen,
                2 + include_class,
            ),
            dtype=object,
        )

        # fill sequence
        for i, _ in enumerate(seqs):
            if include_start:
                seqs[i][0] = ["<START>", "<START>"]
            seqs[i][include_start:] = queries[i * stride : i * stride + actual_seqlen]

        return np.array(seqs, dtype=str)


class ClusterSequencingStrategy:
    """Sequences are created by splitting queries based on their timestamp, resulting in sequences of different lengths being padded;
    each query in the dataset will only appear in a single sequence.
    """

    def make_sequences(
        self,
        queries: np.ndarray,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        seqs = []

        # sort queries by host, and within each host by timestamp
        if group_by_host:
            queries = queries[np.lexsort((queries[:, 2], queries[:, 0]))]

        # for each unique host, get queries made by that host
        for host in np.unique(queries[:, 0]):
            host_queries = queries[np.where(queries[:, 0] == host)[0]]

            # get cluster labels of that host's queries
            host_cluster_labels = self.get_clusters_from_timestamp(host_queries)

            # from each of those clusters we are going to make a sequence
            for c in range(np.max(host_cluster_labels) + 1):
                cluster = host_queries[
                    np.where(host_cluster_labels == c)
                ]  # get queries associated to the current cluster label

                # remove timestamp once it's no longer needed
                cluster = np.delete(cluster, 2, axis=1)

                # take first element of the domain (the domain is a list with always one token)
                # TODO this only works for the trivial tokenizer
                if cluster.shape[-1] == 2:
                    cluster = np.array([[q[0], q[1][0]] for q in cluster])
                else:  # if it includes the class
                    cluster = np.array([[q[0], q[1][0], q[2]] for q in cluster])

                # truncate clusters (sequences) longer than seqlen, moving the excess to a new sequence
                while len(cluster) > seqlen:
                    truncated_cluster = cluster[:seqlen]

                    seqs.append(truncated_cluster)
                    cluster = cluster[seqlen:]

                # this sequence will not be full, so we have to pad it to seqlen
                seqs.append(pad(cluster, seqlen))

        return np.array(seqs, dtype=str)

    @staticmethod
    def get_clusters_from_timestamp(
        queries, eps=lambda deltas: np.percentile(deltas, 50)
    ):
        """Get a list of labels mapping each item in the input array to a cluster.
        Uses the DBSCAN density-based clustering algorithm.

        Args:
            queries (numpy array): A numpy 2D array containing timestamped items.
            The timestamp is assumed to be in the third column.
            eps (func): Used to change the method eps is calculated in DBSCAN
            (which determines the maximum distance for two points to be neighbors).
            By default it is the 50-percentile of the subsequent timestamp differences.
            The previous default was `lambda deltas: np.mean(deltas) + 3 * np.std(deltas)`.
        """
        deltas = queries[1:, 2] - queries[:-1, 2]
        try:
            dbscan = DBSCAN(eps=eps(deltas)).fit(np.expand_dims(queries[:, 2], -1))
        except IndexError:
            print(
                f"[WARN] Found a host with a single query: {queries}. Returning [-1]."
            )
            return [-1]

        return dbscan.labels_


class TimeWindowStrategy:
    """Sequences are created by shifting a time window of a certain duration
    across the queries and returning the queries that appear inside the window.
    The duration of the window is not completely fixed: it has a base duration,
    but it can be modified according to certain constraints on a per-sequence
    basis. In particular, the two main constraints are:
    - minimum inter-sequence distance (delta_min): if the last query of a
    sequence and the query after that would have a gap below a certain
    threshold, the second query is merged in the sequence,
    - maximum intra-sequence distance (delta_max): if a query in a sequence and
    another query in the same sequence would have a gap above a certain
    threshold, the second query is split into a new sequence.
    """

    def make_sequences(
        self,
        queries: np.ndarray,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any,
    ) -> np.ndarray:
        verbose = kwargs.get("verbose", False)

        # queries: [n_queries_in_file, 3] (host, domain, timestamp)

        # sort queries by host, and within each host by timestamp
        if group_by_host:
            queries = queries[np.lexsort((queries[:, 2], queries[:, 0]))]

        base_duration = kwargs.get("base_duration", 8.0)
        delta_min = kwargs.get("delta_min", 3.0)
        delta_max = kwargs.get("delta_max", 4.0)

        seqs = []

        for host in np.unique(queries[:, 0]):
            host_queries = queries[np.where(queries[:, 0] == host)]
            start_ts = host_queries[0, 2]
            last_ts = host_queries[0, 2]
            current_seq = []
            for query in host_queries:
                if query.shape[0] == 4:  # if it includes class
                    (host, domain, timestamp, c) = query
                else:
                    (host, domain, timestamp) = query
                if verbose:
                    print(
                        f"Reading new query ({host} {domain} {timestamp}); start_ts: {start_ts}, last_ts: {last_ts}"
                    )
                exceeds_base_duration = timestamp - start_ts > base_duration
                exceeds_intradistance = timestamp - last_ts > delta_max
                exceeds_interdistance = timestamp - last_ts > delta_min
                isfull = len(current_seq) >= seqlen

                if (
                    exceeds_intradistance
                    or (exceeds_base_duration and exceeds_interdistance)
                    or isfull
                ):
                    current_seq = pad(current_seq, seqlen)
                    if verbose:
                        print(
                            f"End of sequence ({'too thin' if exceeds_intradistance else 'too long' if (exceeds_base_duration and exceeds_interdistance) else 'full'})"
                        )
                        print(current_seq)
                    seqs.append(current_seq)
                    current_seq = []
                    start_ts = timestamp

                # only works for TrivialTokenizer
                if query.shape[0] == 4:  # if it includes class
                    current_seq.append([host, domain[0], timestamp, c])
                else:
                    current_seq.append([host, domain[0], timestamp])

                if verbose:
                    print(f"Appending query ({host} {domain} {timestamp})")

                last_ts = timestamp

        seqs = np.array(seqs, dtype=str)

        # remove timestamp
        seqs = np.delete(seqs, 2, axis=-1)

        return seqs
