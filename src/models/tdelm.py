import yaml
import logging
import tensorflow as tf


# NOTE This is the initial boilerplate for the tokenization version of the model.
class TDELM(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.addHandler(logging.StreamHandler(sys.stdout)))

        # Configuration
        conf_path = {
            "small": "conf/DELM_small.yaml",
            "all": "conf/DELM_large.yaml",
            "clean": "conf/DELM_clean.yaml",
        }[kwargs.get("version")]
        self.conf = ModelConfiguration().load(conf_path)
        for key, value in kwargs.items():
            if value is not None:
                self.conf.set(key, value)

        # Vocabularies
        self.hosts_vocabulary = (
            open(self.conf.get("hosts_vocab_path"), "r").read().split("\n")
        )
        self.subdomains_vocabulary = (
            open(self.conf.get("subdomains_vocab_path"), "r").read().split("\n")
        )
        self.hosts_vocabulary = tf.constant(self.hosts_vocabulary)
        self.subdomains_vocabulary = tf.constant(self.subdomains_vocabulary)
        self.ndomains = self.domains_lookup.vocabulary_size()
        self.nhosts = self.hosts_lookup.vocabulary_size()
        self.subdomains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.subdomains_vocabulary
        )
        self.hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.hosts_vocabulary
        )

        # Architecture
        self.host_embeddings = tf.keras.layers.Embedding(
            input_dim=self.nhosts,
            output_dim=self.conf.get("host_dim"),
            input_length=self.conf.get("seqlen"),
        )
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.get("subdomain_dim"),
            input_length=self.conf.get("seqlen"),
        )
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, seq, training_None, **kwargs):
        pass

    def train_step(self, seq):
        pass

    def test_step(self, seq):
        pass

    def predict(self, seq):
        pass


class ModelConfiguration:
    def __init__(self):
        self.conf = {}

    def get(self, property):
        return self.conf[property]

    def set(self, property, value):
        self.conf[property] = value

    def set(self, dictionary):
        self.conf = self.conf | dictionary

    def load(self, path):
        with open(path, "r") as f:
            conf = yaml.safe_load(f)

    def reset(self):
        self.conf = {}

    def reset(self, property):
        self.conf.pop(property)
