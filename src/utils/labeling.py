import re
import numpy as np
import pandas as pd
from typing import Protocol
from abc import abstractmethod


class LabelMatcher(Protocol):
    @abstractmethod
    def __init__(self, domains_vocab_path):
        raise NotImplementedError()

    @abstractmethod
    def match_labels(self, queries, labeled_data_path, **_):
        raise NotImplementedError()


class MaliciousDomainClassificationLabelMatcher:
    def __init__(self, domains_vocab_path):
        # TODO this only works for TrivialTokenizer
        with open(domains_vocab_path, "r") as f:
            self.vocab = f.read().splitlines()

    def match_labels(self, queries, labeled_data_path, **_):
        # Load labels
        labels = pd.read_csv(
            labeled_data_path,
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

        # Only use labels for domains in the vocabulary
        labels = labels[labels["domain"].isin(self.vocab)]
        labels = labels.reset_index()

        labels = labels.sort_values(
            by="domain",
            key=lambda domains: domains.map(lambda domain: self.vocab.index(domain)),
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


class BotnetDetectionLabelMatcher:
    _LABEL_IDS = {
        "Clean": 0,
        "Unknown": 1,
        "modpack": 2,
        "virut": 3,
        "necurs": 4,
        "conficker": 5,
        "ud3": 6,
        "suppobox": 7,
        "nymaim": 8,
        "tofsee": 9,
        "pitou": 10,
    }

    def __init__(self, domains_vocab_path):
        pass

    def match_labels(self, queries, labeled_data_path, **kwargs):
        # TODO it would be nice to clean train_classifier.py by removing
        # its labeling algorithm and calling the one in this file directly
        # (MaliciousDomainClassificationLaberMatcher).
        # TODO train_classifier.py and train_classifier_features.py
        # should be merged and it should point to the appropriate labeling
        # algorithm from this file

        # Get datetime of the current queries
        mmddhh = re.search("^\d{4}(\d{4})_(\d{2})", kwargs.get("queries_filename"))
        mmddhh = mmddhh.group(1) + mmddhh.group(2)

        # Get host labels (for botnet belonging)
        labels = pd.read_csv(labeled_data_path)
        labels = labels[["Hostname_MMDDHH", "bot_family"]]

        # Split hostname and datetime into two columns. Datetime will be used to match with the processed csv files
        labels[["Hostname", "MMDDHH"]] = labels["Hostname_MMDDHH"].str.split(
            "_", expand=True
        )

        # Select rows for the current datetime
        labels = labels[labels["MMDDHH"] == mmddhh]

        # If there are no labels (strange case for hour 042700), consider all labels unknown
        if len(labels) == 0:
            return np.concatenate(
                [queries, np.full((len(queries), 1), self._LABEL_IDS["Unknown"])],
                axis=-1,
            )

        # Replace labels with their corresponding ids
        labels["bot_family"] = labels["bot_family"].replace(
            to_replace=[k for k in self._LABEL_IDS],
            value=[self._LABEL_IDS[k] for k in self._LABEL_IDS],
        )

        # Convert labels to a numpy array and select useful columns
        labels = labels.to_numpy()[:, [2, 1]]  # columns: [hostname, label]

        # Match labels with queries having the same host
        sorter = np.argsort(labels[:, 0])
        idx = sorter[np.searchsorted(labels[:, 0], queries[:, 0], sorter=sorter)]
        matching_labels = labels[idx, 1]
        queries = np.concatenate(
            [queries, np.expand_dims(matching_labels, -1).astype(str)], axis=-1
        )

        return queries
