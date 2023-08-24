import gib_detection
from typing import Protocol
from abc import abstractmethod


class Tokenizer(Protocol):
    domain_vocabulary: list[str]
    host_vocabulary: list[str]

    @abstractmethod
    def fit(self, queries: list[(str, str)]):
        raise NotImplementedError()

    @abstractmethod
    def tokenize(self, domain: str) -> list[str]:
        raise NotImplementedError()

    def save_domain_vocabulary(self, path):
        with open(path, "w") as f:
            # add special tokens if they are missing. <UNK> is not added explicitly, it's handled by TF
            if "<START>" not in self.domain_vocabulary:
                self.domain_vocabulary.append("<START>")
            if "<MASK>" not in self.domain_vocabulary:
                self.domain_vocabulary.append("<MASK>")
            if "<PAD>" not in self.domain_vocabulary:
                self.domain_vocabulary.append("<PAD>")

            f.write("\n".join(self.domain_vocabulary))

    def save_host_vocabulary(self, path):
        with open(path, "w") as f:
            # add special tokens if they are missing. <UNK> is not added explicitly, it's handled by TF
            if "<START>" not in self.host_vocabulary:
                self.host_vocabulary.append("<START>")
            if "<MASK>" not in self.host_vocabulary:
                self.host_vocabulary.append("<MASK>")
            if "<PAD>" not in self.host_vocabulary:
                self.host_vocabulary.append("<PAD>")

            f.write("\n".join(self.host_vocabulary))

    def load_domain_vocabulary(self, path):
        with open(path, "r") as f:
            self.domain_vocabulary = f.read().splitlines()

    def load_host_vocabulary(self, path):
        with open(path, "r") as f:
            self.host_vocabulary = f.read().splitlines()

    @staticmethod
    def top_dict_keys(dictionary: dict, truncate_to: int = None) -> list[str]:
        """Extract a list of the keys with the highest value in descending order from a dictionary.

        Args:
            dictionary (dict): _description_
            truncate_to (int, optional): If specified, truncate the output list to that length. Defaults to None.

        Returns:
            list (str): List of the keys with the highest value.
        """
        return list(
            list(
                zip(
                    *sorted(
                        dictionary.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
            )[0]
        )[:truncate_to]


class TrivialTokenizer(Tokenizer):
    def __init__(self, max_tokens=30000):
        super().__init__()
        self.max_tokens = max_tokens
        self.domain_counts = {}
        self.domain_vocabulary = []
        self.host_counts = {}
        self.host_vocabulary = []

    def fit(self, queries: list[(str, str)]):
        for (host, domain) in queries:
            if domain not in self.domain_counts.keys():
                self.domain_counts[domain] = 0
            self.domain_counts[domain] += 1
            if host not in self.host_counts.keys():
                self.host_counts[host] = 0
            self.host_counts[host] += 1

        self.domain_vocabulary = self.top_dict_keys(
            self.domain_counts, truncate_to=self.max_tokens
        )
        self.host_vocabulary = self.top_dict_keys(
            self.host_counts, truncate_to=self.max_tokens
        )

    def tokenize(self, domain: str) -> list[str]:
        return [domain if domain in self.domain_vocabulary else "<UNK>"]


class SubdomainTokenizer(Tokenizer):
    def __init__(self, max_tokens=30000, reverse=True):
        super().__init__()
        self.max_tokens = max_tokens
        self.domain_counts = {}
        self.domain_vocabulary = []
        self.host_counts = {}
        self.host_vocabulary = []
        self.reverse = reverse
        self.processors: list[TokenProcessor] = [GibberishDetector()]
        if reverse:
            self.processors.append(TokenReverser())

    def fit(self, queries: list[(str, str)]):
        for (host, domain) in queries:
            tokens = domain.split(".")
            tokens = ["." + token for token in tokens]
            for processor in self.processors:
                tokens = processor.process(tokens)
            for token in tokens:
                if token not in self.domain_counts.keys():
                    self.domain_counts[token] = 0
                self.domain_counts[token] += 1
            if host not in self.host_counts.keys():
                self.host_counts[host] = 0
            self.host_counts[host] += 1

        self.domain_vocabulary = self.top_dict_keys(
            self.domain_counts, truncate_to=self.max_tokens
        )
        self.host_vocabulary = self.top_dict_keys(
            self.host_counts, truncate_to=self.max_tokens
        )

    def tokenize(self, domain: str) -> list[str]:
        # Actual tokenization / splitting
        tokens = domain.split(".")
        tokens = ["." + token for token in tokens]

        # Tokens processing
        for processor in self.processors:
            tokens = processor.process(tokens)

        tokens = [
            token if token in self.domain_vocabulary else "<UNK>" for token in tokens
        ]

        return tokens


class TokenProcessor(Protocol):
    @abstractmethod
    def process(self, tokens):
        raise NotImplementedError()


class GibberishDetector(TokenProcessor):
    def __init__(self):
        self.gd = gib_detection.GibDetector()
        self.gd.train()

    def process(self, tokens):
        return ["<GIB>" if not self.gd.detect(token) else token for token in tokens]


class TokenReverser(TokenProcessor):
    def process(self, tokens):
        return tokens[::-1]
