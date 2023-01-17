import tensorflow as tf
import yaml
import sys
import logging
from colorama import Fore, Style
import os
from types import SimpleNamespace as NS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# @TODO Should I calculate the loss only on masked tokens?
# Yes. https://stackoverflow.com/questions/68043950/how-are-we-making-prediction-for-masked-tokens-alone-in-bert
# @TODO Stop gradient calculation for non-masked tokens.
# Look: https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
# and: https://www.tensorflow.org/api_docs/python/tf/stop_gradient

# @TODO Why does BERT replace 10% of masked tokens with the same word?
# https://stats.stackexchange.com/questions/575002/bert-mlm-80-mask-10-random-words-and-10-same-word-how-does-this-work
class DELM(tf.keras.Model):
    def __init__(self, **conf):

        super(DELM, self).__init__()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        default_conf_file = "conf/DELM_small.yaml"
        try:
            with open(default_conf_file, "r") as f:
                default_conf = yaml.safe_load(f)
        except OSError as e:
            self._logger.warning(
                f"{Fore.RED}Could not open conf file {default_conf_file}:\n{e}"
                + "\nTrying to default to argument configuration.{Style.RESET_ALL}"
            )
            default_conf = {}

        self.conf = default_conf | conf

        # with open(self.conf['domains_vocab_path'], 'r') as f:
        #    domains_vocab = f.read().splitlines()

        # with open(self.conf['hosts_vocab_path'], 'r') as f:
        #    hosts_vocab = f.read().splitlines()

        self.domains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["domains_vocab_path"],  # domains_vocab,
            num_oov_indices=0,
        )
        self.hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["hosts_vocab_path"], num_oov_indices=0  # hosts_vocab,
        )

        self.ndomains = self.domains_lookup.vocabulary_size()  # len(domains_vocab)
        self.nhosts = self.hosts_lookup.vocabulary_size()  # len(hosts_vocab)

        # self._logger.debug(f'domains_vocab: \n{domains_vocab}')
        # self._logger.debug(f'hosts_vocab: \n{hosts_vocab}')

        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["domain_dim"],
            input_length=self.conf["sequence_length"],
        )

        self.host_embeddings = tf.keras.layers.Embedding(
            input_dim=self.nhosts,
            output_dim=self.conf["host_dim"],
            input_length=self.conf["sequence_length"],
        )

        self.multihead_gat = MultiHeadGAT(
            n_heads=self.conf["n_heads"], emb_dim=self.conf["domain_dim"]
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.classifier = Classifier([256], self.ndomains)

    @tf.function
    def mask(self, sequence, p):
        rnd = tf.random.uniform(shape=tf.shape(sequence), dtype=tf.float32)
        mask = tf.fill(dims=tf.shape(sequence), value="<MASK>")
        masked_sequence = tf.where(tf.math.less(rnd, p), mask, sequence)
        return masked_sequence

    @tf.function
    def call(self, inputs, training=None):

        hosts = self.slice_hosts(inputs)
        domains = self.slice_domains(inputs)

        hosts = tf.squeeze(hosts, axis=-1)
        domains = tf.squeeze(domains, axis=-1)

        if training or self.conf["mask_test"]:
            domains = self.mask(
                domains, p=tf.constant(self.conf["mask_p"], dtype=tf.float32)
            )
            self._logger.debug(f"Masked Domains: {domains}")

        domain_indexes = self.domains_lookup(domains)
        host_indexes = self.hosts_lookup(hosts)
        self._logger.debug(f"Domain indexes: {domain_indexes}")
        self._logger.debug(f"Host indexes: {host_indexes}")

        e_d = self.domain_embeddings(domain_indexes)
        e_h = self.host_embeddings(host_indexes)
        self._logger.debug(f"Domain embeddings:\n{e_d}")
        self._logger.debug(f"Host embeddings:\n{e_h}")

        # @TODO @ISSUE This can only be done if domain dim and host dim are the same!
        emb = self.conf["omega"] * e_d + (1 - self.conf["omega"]) * e_h

        emb = self.multihead_gat(emb)
        emb = self.bn(emb)
        emb = self.classifier(emb)
        self._logger.debug(f"Prediction:\n{emb}")

        return emb

    def get_config(self):
        return self.conf

    @tf.function
    def onehot(self, seq, vocab_length):
        """seq must contain token indexes (integers)."""
        return tf.squeeze(tf.one_hot(seq, vocab_length), axis=-2)

    @tf.function
    def slice_hosts(self, t):
        """t must be a tf.Tensor of shape [Batch, Seqlen, 2]"""
        return tf.slice(
            t, [0, 0, 0], tf.concat((tf.gather(tf.shape(t), [0, 1]), [1]), axis=0)
        )

    @tf.function
    def slice_domains(self, t):
        """t must be a tf.Tensor of shape [Batch, Seqlen, 2]"""
        return tf.slice(
            t, [0, 0, 1], tf.concat((tf.gather(tf.shape(t), [0, 1]), [1]), axis=0)
        )

    @tf.function
    def train_step(self, data):
        x = data
        domains = self.slice_domains(x)
        domains = tf.squeeze(domains, axis=-1)
        domain_indexes = self.domains_lookup(domains)  # [B, L, 1] o [B, L]

        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            loss = self.compiled_loss(
                domain_indexes, pred, regularization_losses=self.losses
            )

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x = data
        domains = self.slice_domains(x)
        domains = tf.squeeze(domains, axis=-1)
        domain_indexes = self.domains_lookup(domains)

        pred = self(x, training=False)

        self.compiled_loss(domain_indexes, pred, regularization_losses=self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class MultiHeadGAT(tf.keras.layers.Layer):
    def __init__(self, n_heads, emb_dim):
        super(MultiHeadGAT, self).__init__()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        self.n_heads = tf.constant(n_heads)
        self.emb_dim = tf.constant(emb_dim)
        self.head_dim = tf.math.floordiv(self.emb_dim, self.n_heads)

        self.Wq = [tf.keras.layers.Dense(self.head_dim) for i in range(self.n_heads)]
        self.Wk = [tf.keras.layers.Dense(self.head_dim) for i in range(self.n_heads)]
        self.Wv = [tf.keras.layers.Dense(self.head_dim) for i in range(self.n_heads)]
        self.Wo = tf.keras.layers.Dense(self.emb_dim)

    def call(self, inputs):

        # inputs (embeddings) [B, L, emb_dim]
        Q = tf.stack(
            [Wqi(inputs) for Wqi in self.Wq], axis=1
        )  # [B, n_heads, L, head_dim]
        K = tf.stack(
            [Wki(inputs) for Wki in self.Wk], axis=1
        )  # [B, n_heads, L, head_dim]
        V = tf.stack(
            [Wvi(inputs) for Wvi in self.Wv], axis=1
        )  # [B, n_heads, L, head_dim]

        scores = tf.linalg.matmul(
            Q, tf.transpose(K, (0, 1, 3, 2))
        )  # [B, n_heads, L, L]
        scores = tf.math.divide(
            scores, tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        )  # [B, n_heads, L, L] (normalization)
        scores = tf.nn.softmax(scores)  # [B, n_heads, L, L] attention weights
        result = tf.linalg.matmul(scores, V)  # [B, n_heads, L, head_dim]
        result = tf.concat(
            tf.unstack(result, axis=1), axis=-1
        )  # [B, L, n_heads*head_dim]

        result = self.Wo(result)  # [B, L, emb_dim]

        return result


class Classifier(tf.keras.layers.Layer):
    def __init__(self, dense_dims, softmax_dim):
        super(Classifier, self).__init__()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # the last layer mapping to vocab length is not included. this could be empty
        self.dense_layers = [
            tf.keras.layers.Dense(dim, activation="relu") for dim in dense_dims
        ]

        self.out = tf.keras.layers.Dense(softmax_dim, activation="linear")

    @tf.function
    def call(self, emb):
        for l in self.dense_layers:
            emb = l(emb)
        emb = self.out(emb)
        # emb = tf.nn.softmax(emb)
        return emb
