import tensorflow as tf
import yaml
import sys
import logging
from colorama import Fore, Style
import os
from types import SimpleNamespace as NS
import datetime


class DELM(tf.keras.Model):
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

    def log_weight_histograms(self, vars, every):
        def body():
            with self.tb_writer.as_default():
                for v in vars:
                    tf.summary.histogram(v.name, v, step=self.step)

        tf.cond(tf.math.floormod(self.step, every) == 0, body, lambda: None)

    @tf.function
    def mask(self, sequence, mask_p, same_p, random_p):
        # Randomly decide the tokens to mask for the current batch
        rnd = tf.random.uniform(shape=tf.shape(sequence), dtype=tf.float32)  # [B,L]

        # Create masks of <MASK>, unchanged and random tokens
        all_mask_tokens = tf.fill(dims=tf.shape(sequence), value="<MASK>")  # [B,L]
        all_same_tokens = tf.identity(sequence)  # [B,L]
        all_random_tokens = tf.random.uniform(
            shape=tf.shape(sequence),
            minval=0,
            maxval=self.domains_lookup.vocabulary_size(),
            dtype=tf.dtypes.int64,
        )  # [B,L]
        all_random_tokens = self.inverse_domains_lookup(all_random_tokens)

        # Replace the predefined % of tokens with the correct mask
        masked_sequence = tf.where(
            tf.math.less(rnd, mask_p + same_p + random_p), all_mask_tokens, sequence
        )  # replace <MASK>s
        masked_sequence = tf.where(
            tf.math.less(rnd, same_p + random_p), all_same_tokens, masked_sequence
        )  # also replace same
        masked_sequence = tf.where(
            tf.math.less(rnd, random_p), all_random_tokens, masked_sequence
        )  # also replace randoms; [B,L]

        # A token is considered *masked* if any type of mask is applied to it
        # (<MASK>, unchanged and random all count as masks)
        mask = tf.math.less(rnd, mask_p + same_p + random_p)  # [B,L]

        return masked_sequence, mask

    def __init__(self, **conf):
        super(DELM, self).__init__()

        # TensorBoard Init
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.tb_writer = tf.summary.create_file_writer(
            f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # Configuration
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
        self.conf = dict(default_conf, **conf)

        # Tokens Lookup
        self.domains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["domains_vocab_path"],
            num_oov_indices=0,
        )
        self.inverse_domains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["domains_vocab_path"], num_oov_indices=0, invert=True
        )
        self.hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["hosts_vocab_path"], num_oov_indices=0
        )
        self.ndomains = self.domains_lookup.vocabulary_size()
        self.nhosts = self.hosts_lookup.vocabulary_size()

        # Tokens Embeddings
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

        self.dropout = tf.keras.layers.Dropout(0.1)

        # MHGAT Blocks
        self.blocks = [
            MHGAT_Block(
                n_heads=self.conf["n_heads"],
                emb_dim=self.conf["domain_dim"],
                block_id=block,
                tb_writer=self.tb_writer,
            )
            for block in range(self.conf["blocks"])
        ]

        # Classification Layers
        self.classifier = Classifier([512], self.ndomains)

    @tf.function
    def call(self, inputs, training=None):
        # Separate host from domain tokens in the given sequence
        hosts = self.slice_hosts(inputs)
        domains = self.slice_domains(inputs)  # [B,L,1]
        hosts = tf.squeeze(hosts, axis=-1)
        domains = tf.squeeze(domains, axis=-1)  # [B,L]

        # Mask the tokens if required
        mask = None
        if training or self.conf["mask_test"]:
            mask_p = tf.constant(self.conf["mask_p"]["mask"])
            same_p = tf.constant(self.conf["mask_p"]["same"])
            random_p = tf.constant(self.conf["mask_p"]["random"])
            domains, mask = self.mask(domains, mask_p, same_p, random_p)

        # Lookup vocab index for each token
        domain_indexes = self.domains_lookup(domains)
        host_indexes = self.hosts_lookup(hosts)

        # Retrieve embedding for each token index
        e_d = self.domain_embeddings(domain_indexes)
        e_h = self.host_embeddings(host_indexes)

        # Weighted sum of host and domain embeddings (according to omega)
        # @TODO @ISSUE This can only be done if domain dim and host dim are the same!
        emb = self.conf["omega"] * e_d + (1 - self.conf["omega"]) * e_h

        # Dropout the embeddings before feeding to MHGAT blocks
        emb = self.dropout(emb)

        # Forward embeddings through MHGAT blocks
        for block in self.blocks:
            emb = block(emb, step=self.step)

        # Call classification layers on final embeddings
        emb = self.classifier(emb)

        return emb, mask

    @tf.function
    def train_step(self, data):
        x = data
        domains = self.slice_domains(x)
        domains = tf.squeeze(domains, axis=-1)
        domain_indexes = self.domains_lookup(domains)  # [B, L]

        with tf.GradientTape() as tape:
            pred, mask = self(x, training=True)  # [B,L,vsize], [nmasked,2] o [B,L]
            true_onehot = tf.one_hot(
                domain_indexes,
                tf.cast(self.domains_lookup.vocabulary_size(), tf.int32),
            )  # one-hot encoding of the original sequence, [B,L,vsize]

            mask = tf.reshape(
                tf.tile(
                    mask,
                    [1, self.domains_lookup.vocabulary_size()],
                ),
                tf.shape(true_onehot),
            )
            updated = tf.where(mask, true_onehot, pred)

            loss = self.compiled_loss(
                domain_indexes, updated, regularization_losses=self.losses
            )

        with self.tb_writer.as_default():
            tf.summary.scalar("train_loss", loss, step=self.step)

        # Compute gradients and update weights
        trainable_variables = self.trainable_variables

        # TensorBoard -- Visualize weights
        if self.conf["tensorboard"]:
            self.log_weight_histograms(
                (
                    trainable_variables[0],
                    trainable_variables[1],
                    trainable_variables[2],
                    trainable_variables[18],
                    trainable_variables[34],
                    trainable_variables[50],
                    trainable_variables[54],
                    trainable_variables[56],
                ),
                every=500,
            )

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        # TensorBoard -- Increment step
        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x = data
        domains = self.slice_domains(x)
        domains = tf.squeeze(domains, axis=-1)
        domain_indexes = self.domains_lookup(domains)

        pred, _ = self(x, training=False)

        loss = self.compiled_loss(
            domain_indexes, pred, regularization_losses=self.losses
        )

        with self.tb_writer.as_default():
            tf.summary.scalar("val_loss", loss, step=self.step)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class MHGAT_Block(tf.keras.layers.Layer):
    def __init__(self, n_heads, emb_dim, **kwargs):
        super(MHGAT_Block, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # TensorBoard Init
        self.tensorboard = kwargs.get("tensorboard", False)
        self.tb_writer = kwargs.get("tb_writer", None)
        block_id = kwargs.get("block_id", 0)

        # Configuration
        self.n_heads = tf.constant(n_heads)
        self.emb_dim = tf.constant(emb_dim)
        self.head_dim = tf.math.floordiv(self.emb_dim, self.n_heads)
        self.nonlinear_stretch = tf.constant(4)

        # Query, Key and Value matrices (multi-head)
        self.Wq = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{block_id}-Wq/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wk = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{block_id}-Wk/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wv = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{block_id}-Wv/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wo = tf.keras.layers.Dense(self.emb_dim, name=f"MHGAT{block_id}-Wo")

        # Batch Normalization
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Feed Forward NN
        self.linear1 = tf.keras.layers.Dense(self.emb_dim)
        self.nonlinear = tf.keras.layers.Dense(
            self.emb_dim * self.nonlinear_stretch, activation="relu"
        )
        self.linear2 = tf.keras.layers.Dense(self.emb_dim)

        # Batch Normalization
        self.bn2 = tf.keras.layers.BatchNormalization()

    @tf.function
    def log_score_heatmaps(self, scores, step):
        b_range = tf.range(tf.size(scores))
        b_range = tf.expand_dims(b_range, axis=1)
        indices = tf.concat(
            [b_range, tf.zeros_like(b_range)], axis=1
        )  # log only the first head for each sequence in the batch @TODO log all of them?
        image = tf.expand_dims(tf.gather_nd(scores, indices), axis=-1)
        with self.tb_writer.as_default():
            tf.summary.image(
                "att_scores",
                image,
                step=step,
            )

    @tf.function
    def call(self, inputs, **kwargs):
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

        # <--- Inject adjacency mask here (Vaswani says it's done after normalization)

        scores = tf.nn.softmax(scores)  # [B, n_heads, L, L] attention weights

        # Log attention heatmaps
        if self.tensorboard:
            self.log_score_heatmaps(scores, kwargs.get("step", None))

        result = tf.linalg.matmul(scores, V)  # [B, n_heads, L, head_dim]
        result = tf.concat(
            tf.unstack(result, axis=1), axis=-1
        )  # [B, L, n_heads*head_dim]

        result = self.Wo(result)  # [B, L, emb_dim]

        # Add & Norm
        result = tf.math.add(result, inputs)
        result = self.bn1(result)

        # Feed Forward NN
        proj = self.linear1(result)
        proj = self.nonlinear(proj)
        proj = self.linear2(proj)

        # Add & Norm
        result = tf.math.add(result, proj)
        result = self.bn2(result)

        return result


class Classifier(tf.keras.layers.Layer):
    def __init__(self, dense_dims, softmax_dim):
        super(Classifier, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # Optional nonlinear layers; they don't include the one mapping to vocab length
        self.dense_layers = [
            tf.keras.layers.Dense(dim, activation="relu") for dim in dense_dims
        ]

        self.out = tf.keras.layers.Dense(softmax_dim, activation="linear")

    @tf.function
    def call(self, emb):
        for l in self.dense_layers:
            emb = l(emb)
        emb = self.out(emb)
        emb = tf.nn.softmax(emb)
        return emb


# class AdjacencyEstimator(tf.keras.layers.Layer):
#     def __init__(self):
#         super(AdjacencyEstimator, self).__init__()

#     # def hierarchical_similarity(x):
#         # d1, d2 = x # List of subdomains for each domain
#         # common_subdomains = tf.Variable(0, trainable=False)
#         # i = tf.Variable(0, trainable=False)
#         # tf.while_loop(
#         #     tf.math.logical_and(tf.math.less(i, tf.size(d1)), tf.math.less(i,tf.size(d2))),
#         #     lambda d1, d2, i, common_subdomains: tf.cond(tf.math.equal()),
#         #     [d1, d2, i, common_subdomains]
#         # )

#     # @tf.function
#     def call(self, inputs):
#         B, L = inputs.shape # inputs: [B,L]
#         subdomains = tf.strings.split(inputs, sep=".")  # [B,L,?]

#         subdomains = tf.keras.utils.pad_sequences(subdomains)
#         tf.pad(subdomains, tf.constant([]))
#         tf.where(tf.math.equals())
#         # for b, _ in enumerate(B):
#         #     for d, domain in enumerate(L):
#         #         for j, other in enumerate(L):
#         #             common = 0
#         #             for _, name in enumerate(j)
#         #             subdomains[b][d]
#         # tf.map_fn(fn=hierarchical_similarity, elems=[subdomains, subdomains], fn_output_signature=tf.TensorSpec(tf.shape(inputs), tf.float32))
#         return tf.ones_like(inputs, dtype=tf.float32)
