import os
import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from typing import Any, Iterator, Protocol
from utils.sequencing import pad, get_clusters_from_timestamp


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

    def __init__(
        self,
        input_folder: str,
        sequencing_strategy: SequencingStrategy,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any
    ) -> None:
        """Initialize a SequenceGenerator object.
        TODO some kwargs are not fully supported.

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
            model (str): model to use. Either "delm" or "w2v".
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
        for f in os.listdir(self.input_folder):
            if os.path.splitext(os.path.join(self.input_folder, f))[-1] != ".npy":
                continue

            queries = np.load(os.path.join(self.input_folder, f), allow_pickle=True)
            seqs = self.sequencing_strategy.make_sequences(
                queries,
                self.seqlen,
                self.include_class,
                self.group_by_host,
                **self.kwargs
            )
            for seq in seqs:
                yield seq


class FixedSequencingStrategy:
    """Sequences are cut exactly every `seqlen` queries, disregarding the timestamp.
    Each query in the dataset will appear in multiple sequences, depending on the stride value.
    """

    def make_sequences(
        self,
        queries: np.ndarray,
        seqlen: int,
        include_class: bool,
        group_by_host: bool,
        **kwargs: Any
    ) -> np.ndarray:

        # get kwargs
        stride: bool = kwargs.get("stride", 1)
        include_start: bool = kwargs.get("include_start", False)

        # sort queries by host, and within each host by timestamp
        if group_by_host:
            queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]

        # if the timestamp is present, remove it
        if np.shape(queries)[-1] == 3 + include_class:
            queries = queries[..., :-1]

        # if the domain is a list (tokenized), take the first element
        # TODO this is only ok for TrivialTokenizer
        queries = np.array([[q[0], q[1][0]] for q in queries])

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
        **kwargs: Any
    ) -> np.ndarray:

        seqs = []

        # sort queries by host, and within each host by timestamp
        if group_by_host:
            queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]

        # for each unique host, get queries made by that host
        for host in np.unique(queries[:, 0]):
            host_queries = queries[np.where(queries[:, 0] == host)[0]]

            # get cluster labels of that host's queries
            host_cluster_labels = get_clusters_from_timestamp(host_queries)

            # from each of those clusters we are going to make a sequence
            for c in range(np.max(host_cluster_labels) + 1):

                cluster = host_queries[
                    np.where(host_cluster_labels == c)
                ]  # get queries associated to the current cluster label

                cluster = cluster[:, :-1]  # remove timestamp once it's no longer needed

                # take first element of the domain (the domain is a list with always one token)
                # TODO this is only ok for the trivial tokenizer
                cluster = [[q[0], q[1][0]] for q in cluster]

                # truncate clusters (sequences) longer than seqlen, moving the excess to a new sequence
                while len(cluster) > seqlen:
                    # if kwargs.get("verbose"):
                    #     print(
                    #         f"[INFO] Truncating sequence with cluster ID {c} for host {host}: length of {len(cluster)} exceeds seqlen of {kwargs.get('seqlen')}."
                    #     )
                    truncated_cluster = cluster[:seqlen]

                    seqs.append(truncated_cluster)
                    cluster = cluster[seqlen:]

                # this sequence will not be full, so we have to pad it to seqlen
                seqs.append(pad(cluster, seqlen))

        return np.array(seqs, dtype=str)

    # TODO Include the supervised classification data labelling into the new pipeline
    # this is the previous code:
    #
    # if kwargs.get("include_class"):
    #     labels = pd.read_csv(
    #         os.path.join("scripts", "labels.csv"), index_col=0, header=[0, 1]
    #     )
    #     labels.columns = pd.MultiIndex.from_tuples(
    #         [
    #             ("domain", ""),
    #             ("advertising", "good"),
    #             ("advertising", "ok"),
    #             ("malicious", "good"),
    #             ("malicious", "ok"),
    #             ("suspicious", "good"),
    #             ("suspicious", "ok"),
    #             ("tracking", "good"),
    #             ("tracking", "ok"),
    #             ("other", "good"),
    #             ("other", "ok"),
    #             ("any", "good"),
    #             ("any", "ok"),
    #         ]
    #     )
    #     # Only use labels for domains in embs (i.e. in the vocabulary)
    #     labels = labels[labels["domain"].isin(kwargs.get("vocab"))]
    #     labels = labels.reset_index()

    #     # take (any, ok) column
    #     labels = labels[["domain", "any"]]
    #     labels = labels.to_numpy()[:, [0, 2]]

    #     # add class to each query
    #     sorter = np.argsort(labels[:, 0])
    #     idx = sorter[np.searchsorted(labels[:, 0], queries[:, 1], sorter=sorter)]
    #     classes = labels[idx, 1]
    #     queries = np.concatenate(
    #         [queries, np.expand_dims(classes, -1).astype(str)], axis=-1
    #     )

    #     else:
    #         raise ValueError(
    #             "'strategy' kwarg must be either 'cluster' or 'fixed'."
    #         )

    #     seqs = np.array(seqs, dtype=str)

    # elif kwargs.get("model") == "w2v":  # output [queries, seqlen]
    #     seqs = np.array(Word2Vec.create_pairs(queries[:, 1:], kwargs.get("seqlen")))
    # else:
    #     raise ValueError("Specify model to create sequences.")

    # return seqs
