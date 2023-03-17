import tensorflow as tf
import yaml
from colorama import Fore, Style


class Word2Vec(tf.keras.Model):
    def __init__(self, **conf):
        super(Word2Vec, self).__init__()

        # Configuration
        default_conf_file = "conf/W2V.yaml"
        try:
            with open(default_conf_file, "r") as f:
                default_conf = yaml.safe_load(f)
        except OSError as e:
            self._logger.warning(
                f"{Fore.RED}Could not open conf file {default_conf_file}:\n{e}"
                + "\nTrying to default to argument configuration.{Style.RESET_ALL}"
            )
            default_conf = {}
        self.conf = default_conf
        for key in conf:
            if conf[key] is not None:
                self.conf[key] = conf[key]

        self.domain_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["domains_vocab_path"], num_oov_indices=0
        )
        self.ndomains = self.domain_lookup.vocabulary_size()
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["domain_dim"],
        )
        return

    def call(self, seq):
        # [B,L,2]
        B, L, _ = tf.shape(seq)
        domains = tf.slice(seq, [0, 0, 1], [B, L, 1])
        # pairs =
        return

    @staticmethod
    def create_pairs(seq, window):
        # For each domain in the sequence, creates tuples [(seq_i, seq_i-window), ..., (seq_i, seq_i+window)]
        # Accepts an arbitrarily long sequence. seq is a list of tokens.
        pairs = []
        for index, target in enumerate(seq):
            for other in seq[
                max(0, index - window) : min(index + window + 1, len(seq))
            ]:
                if target != other:
                    pairs.append((target, other))
        return pairs
