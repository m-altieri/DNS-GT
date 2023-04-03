import tensorflow as tf
import numpy as np
import yaml
import sys
import logging
from colorama import Fore, Style
import os
import datetime
from utils.distribute import DummyStrategy


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

    @tf.function
    def tb_log_weights(self, vars, every):  # @TODO
        pass
        # def body():
        #     with self.tb_writer.as_default():
        #         for v in vars:
        #             tf.summary.histogram(v.name, v, step=self.step)
        # tf.cond(tf.math.floormod(self.step, every) == 0, body, lambda: None)

    @tf.function
    def mask(self, hosts, domains, mask_p, same_p, random_p, prevent_masking=False):
        tf.debugging.assert_equal(tf.shape(hosts), tf.shape(domains))
        # Randomly decide the tokens to mask for the current batch
        rnd = tf.random.uniform(
            shape=tf.shape(domains),
            minval=tf.cond(
                tf.math.equal(prevent_masking, True), lambda: 1.0, lambda: 0.0
            ),
            maxval=1.0,
            dtype=tf.float32,
        )  # [B,L]

        rnd = tf.where(
            tf.math.equal(domains, "<START>"), tf.ones_like(rnd), rnd
        )  # <START> is never masked

        # Create masks of <MASK>, unchanged and random tokens
        all_mask_tokens = tf.fill(dims=tf.shape(domains), value="<MASK>")  # [B,L]
        all_same_host_tokens = tf.identity(hosts)  # [B,L]
        all_same_domain_tokens = tf.identity(domains)  # [B,L]
        all_random_host_tokens = tf.random.uniform(
            shape=tf.shape(hosts),
            minval=0,
            maxval=self.hosts_lookup.vocabulary_size(),
            dtype=tf.dtypes.int64,
        )  # [B,L]
        all_random_host_tokens = self.inverse_hosts_lookup(all_random_host_tokens)
        all_random_domain_tokens = tf.random.uniform(
            shape=tf.shape(domains),
            minval=0,
            maxval=self.domains_lookup.vocabulary_size(),
            dtype=tf.dtypes.int64,
        )  # [B,L]
        all_random_domain_tokens = self.inverse_domains_lookup(all_random_domain_tokens)

        # Replace the predefined % of tokens with the correct mask
        masked_hosts = tf.where(
            tf.math.less(rnd, mask_p + same_p + random_p),
            all_mask_tokens,
            hosts,
        )  # replace <MASK>s
        masked_hosts = tf.where(
            tf.math.less(rnd, same_p + random_p),
            all_same_host_tokens,
            masked_hosts,
        )  # also replace same
        masked_hosts = tf.where(
            tf.math.less(rnd, random_p),
            all_random_host_tokens,
            masked_hosts,
        )  # also replace randoms; [B,L]
        masked_domains = tf.where(
            tf.math.less(rnd, mask_p + same_p + random_p),
            all_mask_tokens,
            domains,
        )  # replace <MASK>s
        masked_domains = tf.where(
            tf.math.less(rnd, same_p + random_p),
            all_same_domain_tokens,
            masked_domains,
        )  # also replace same
        masked_domains = tf.where(
            tf.math.less(rnd, random_p),
            all_random_domain_tokens,
            masked_domains,
        )  # also replace randoms; [B,L]

        # A token is considered *masked* if any type of mask is applied to it
        # (<MASK>, unchanged and random all count as masks)
        mask = tf.math.less(rnd, mask_p + same_p + random_p)  # [B,L]

        return hosts, masked_domains, mask  # now i'm not masking the host

    def pretrain(self):
        self.domain_embeddings.trainable = True
        self.host_embeddings.trainable = True
        for block in self.blocks:
            block.trainable = True
        self.frozen = False

    def finetune(self):
        self.domain_embeddings.trainable = False
        self.host_embeddings.trainable = False
        for block in self.blocks:
            block.trainable = False
        self.frozen = True

    def __init__(self, **conf):
        super(DELM, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # Configuration
        default_conf_file = (
            "conf/DELM_small.yaml"
            if conf["version"] == "small"
            else "conf/DELM_large.yaml"
        )
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

        self.frozen = False

        # TensorBoard Init
        TB_FOLDER = "tensorboard"
        self.tb_path = None
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        if not os.path.exists(TB_FOLDER):
            os.makedirs(TB_FOLDER)
        if not self.conf["quick_tb"]:
            self.tb_path = os.path.join(
                TB_FOLDER,
                self.conf.get("run_name", None)
                or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        else:
            self.tb_path = tf.summary.create_file_writer(os.path.join(TB_FOLDER, "tmp"))
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)

        if "dist_strategy" in self.conf:
            self.dist_strategy = self.conf["dist_strategy"]
            self.distributed = self.dist_strategy is not DummyStrategy
        if self.distributed:
            self._logger.info(
                f"Initializing model with distribution strategy: {self.dist_strategy}"
            )
        # Token Adjacency
        # self.adj_estimator = AdjacencyEstimator(
        #     type="binary", normalize=False, tb_writer=self.tb_writer
        # )

        # Token Indexes Lookup
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
        self.inverse_hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["hosts_vocab_path"], num_oov_indices=0, invert=True
        )
        self.ndomains = self.domains_lookup.vocabulary_size()
        self.nhosts = self.hosts_lookup.vocabulary_size()

        # Tokens Embeddings
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["dim"],
            input_length=self.conf["seqlen"],
        )
        self.host_embeddings = tf.keras.layers.Embedding(
            input_dim=self.nhosts,
            output_dim=self.conf["dim"],
            input_length=self.conf["seqlen"],
        )

        self.dropout = tf.keras.layers.Dropout(0.1)

        # MHGAT Blocks
        self.blocks = [
            MHGAT_Block(
                n_heads=self.conf["n_heads"],
                emb_dim=self.conf["dim"],
                block_id=block,
                tensorboard=self.conf["tensorboard"],
                tb_writer=self.tb_writer,
            )
            for block in range(self.conf["blocks"])
        ]

        # Classification Layers
        self.masked_classifier = FF(
            [self.ndomains], ["softmax"]
        )  # Softmax masking classifier
        self.binary_classifier = FF([1], ["sigmoid"])  # Binary classifier

    # @tf.function
    def call(self, inputs, training=None):
        # Separate host from domain tokens in the given sequence
        hosts = self.slice_hosts(inputs)
        domains = self.slice_domains(inputs)  # [B,L,1]
        hosts = tf.squeeze(hosts, axis=-1)
        domains = tf.squeeze(domains, axis=-1)  # [B,L]

        # Mask the tokens if required
        mask_p = tf.constant(self.conf["mask_p"]["mask"])
        same_p = tf.constant(self.conf["mask_p"]["same"])
        random_p = tf.constant(self.conf["mask_p"]["random"])
        hosts, domains, mask = self.mask(
            hosts,
            domains,
            mask_p,
            same_p,
            random_p,
            prevent_masking=not training or not self.conf["mask_test"] or self.frozen,
        )

        # adj_h = self.adj_estimator(
        #     domains
        # )  # about 33% time increase compared to using tf.ones(), and most importantly makes the GPU util unstable
        adj_h = tf.ones(
            [tf.shape(domains)[0], tf.shape(domains)[1], tf.shape(domains)[1]]
        )

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
            emb = block(emb, adj_h, step=self.step)

        if not self.frozen:
            # Call classification layers on final embeddings
            emb = self.masked_classifier(emb)
        else:
            emb = self.binary_classifier(emb)
        return emb, mask

    # Unused
    @tf.function
    def correct_unmasked(self, pred, mask, truth):
        true_onehot = tf.one_hot(
            truth,
            tf.cast(self.domains_lookup.vocabulary_size(), tf.int32),
        )  # one-hot encoding of the true tokens [B,L,vsize]

        mask = tf.transpose(
            tf.reshape(
                tf.tile(
                    mask,
                    [1, self.domains_lookup.vocabulary_size()],
                ),
                (-1, tf.shape(true_onehot)[2], tf.shape(true_onehot)[1]),
            ),
            (0, 2, 1),
        )

        return tf.where(mask, pred, true_onehot)

    @tf.function
    def distributed_train_step(self, seq):
        loss = self.dist_strategy.run(self.train_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    @tf.function
    def distributed_test_step(self, seq):
        loss = self.dist_strategy.run(self.test_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    def train_step(self, seq, y=None):
        domains = tf.squeeze(self.slice_domains(seq), axis=-1)
        domain_indexes = self.domains_lookup(domains)  # [B,L]

        with tf.GradientTape() as tape:
            pred, mask = self(seq, training=True)  # [B,L,vsize], [B,L]

            if not self.frozen:
                loss = self.compiled_loss(
                    tf.boolean_mask(domain_indexes, mask),
                    tf.boolean_mask(pred, mask),
                    regularization_losses=self.losses,
                )
            else:
                loss = self.compiled_loss(pred, y)
            if self.distributed:
                # TODO Check that the distributed loss is calculated correctly,
                # especially considering that tf.size(loss) is different every time and a bit random
                loss = tf.math.divide(
                    tf.math.reduce_mean(loss), self.dist_strategy.num_replicas_in_sync
                )

        # Compute gradients and update weights
        trainable_variables = self.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        with self.tb_writer.as_default():
            tf.summary.scalar("train_loss", loss, step=self.step)

        # TensorBoard -- Increment step
        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return loss if self.distributed else {m.name: m.result() for m in self.metrics}

    def test_step(self, seq):
        domains = tf.squeeze(self.slice_domains(seq), axis=-1)
        domain_indexes = self.domains_lookup(domains)

        pred, mask = self(seq, training=False)

        loss = self.compiled_loss(
            tf.boolean_mask(domain_indexes, mask),
            tf.boolean_mask(pred, mask),
            regularization_losses=self.losses,
        )
        if self.distributed:
            # TODO Check that the distributed loss is calculated correctly,
            # especially considering that tf.size(loss) is different every time and a bit random
            loss = tf.math.divide(
                tf.math.reduce_mean(loss), self.dist_strategy.num_replicas_in_sync
            )

        with self.tb_writer.as_default():
            tf.summary.scalar("val_loss", loss, step=self.step)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(domain_indexes, pred)

        # Return a dict mapping metric names to current value
        return loss if self.distributed else {m.name: m.result() for m in self.metrics}

    # @tf.function
    def _predict(self, seq, mask):
        mask = tf.constant(mask, dtype=tf.bool)
        domains_mask = tf.squeeze(self.slice_domains(mask), axis=-1)

        masked_seq = tf.where(mask, tf.fill(tf.shape(seq), "<MASK>"), seq)
        pred, _ = self(masked_seq, training=False)

        domains = tf.squeeze(
            self.slice_domains(seq), axis=-1
        )  # there was a bug here. it's important to get domain indexes from the NON-masked sequence
        domain_indexes = self.domains_lookup(domains)

        loss = self.compiled_loss(
            tf.boolean_mask(domain_indexes, domains_mask),
            tf.boolean_mask(pred, domains_mask),
            regularization_losses=self.losses,
        )

        return pred, loss


class MHGAT_Block(tf.keras.layers.Layer):
    @tf.function
    def tb_log_image(self, name, tensor, step, minmax=False):
        if self.tensorboard:
            tensor = tf.cond(
                tf.math.equal(tf.rank(tensor), tf.constant(3)),
                lambda: tf.expand_dims(tensor, -1),
                lambda: tensor,
            )
            tensor = tf.cond(
                tf.constant(minmax), lambda: self.minmax_norm(tensor), lambda: tensor
            )
            with self.tb_writer.as_default():
                tf.summary.image(
                    name=name,
                    data=tensor,
                    step=step,
                )

    @tf.function
    def minmax_norm(self, tensor):
        return tf.divide(
            tf.math.subtract(tensor, tf.math.reduce_min(tensor)),
            tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)),
        )

    def __init__(self, n_heads, emb_dim, **kwargs):
        super(MHGAT_Block, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # TensorBoard Init
        self.tensorboard = kwargs.get("tensorboard", False)
        self.tb_writer = kwargs.get("tb_writer", None)
        self.block_id = kwargs.get("block_id", 0)

        # Configuration
        self.n_heads = tf.constant(n_heads)
        self.emb_dim = tf.constant(emb_dim)
        self.head_dim = tf.math.floordiv(self.emb_dim, self.n_heads)
        self.nonlinear_stretch = tf.constant(4)

        # Query, Key and Value matrices (multi-head)
        self.Wq = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{self.block_id}-Wq/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wk = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{self.block_id}-Wk/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wv = [
            tf.keras.layers.Dense(self.head_dim, name=f"MHGAT{self.block_id}-Wv/h{i}")
            for i in range(self.n_heads)
        ]
        self.Wo = tf.keras.layers.Dense(self.emb_dim, name=f"MHGAT{self.block_id}-Wo")

        # Softmax
        self.softmax = tf.keras.layers.Softmax()

        # Batch Normalization
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Feed Forward NN
        self.linear1 = tf.keras.layers.Dense(
            self.emb_dim
        )  # dall'eq. (2) di Vaswani sembra che questo linear1 non ci sia
        self.nonlinear = tf.keras.layers.Dense(
            self.emb_dim * self.nonlinear_stretch, activation="relu"
        )
        self.linear2 = tf.keras.layers.Dense(self.emb_dim)

        # Batch Normalization
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Residual Dropout
        self.dropout = tf.keras.layers.Dropout(0.1)

    @tf.function
    def call(self, inputs, adj_h, **kwargs):
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

        # tf.print(scores[:, 7])
        # tf.print(tf.math.reduce_max(scores[:, 7]))
        # tf.print(tf.math.reduce_min(scores[:, 7]))
        self.tb_log_image(f"MHGAT{self.block_id}/7", scores[:, 7], step=0, minmax=True)

        # <--- Inject adjacency mask here (Vaswani says it's done after normalization)
        adj_h = tf.expand_dims(adj_h, axis=1)
        adj_h = tf.tile(adj_h, [1, 1, tf.shape(scores)[1], 1])
        adj_h = tf.reshape(adj_h, tf.shape(scores))
        self.tb_log_image(f"MHGAT{self.block_id}/adj_h", adj_h[:, 7], step=0)
        # --->

        # Calculate softmax masking disconnected scores
        scores = self.softmax(
            scores, mask=adj_h
        )  # [B, n_heads, L, L] attention weights
        # tf.print(scores[:, 7], summarize=-1)
        # tf.print(tf.math.reduce_max(scores[:, 7]))
        # tf.print(tf.math.reduce_min(scores[:, 7]))
        self.tb_log_image(
            f"MHGAT{self.block_id}/7-after-softmax", scores[:, 7], step=0, minmax=True
        )

        # Calculate weighted values
        result = tf.linalg.matmul(scores, V)  # [B, n_heads, L, head_dim]
        result = tf.concat(
            tf.unstack(result, axis=1), axis=-1
        )  # [B, L, n_heads*head_dim]

        result = self.Wo(result)  # [B, L, emb_dim]
        result = self.dropout(result)  # Residual Dropout

        # Add & Norm
        result = tf.math.add(result, inputs)
        result = self.bn1(result)

        # Feed Forward NN
        proj = self.linear1(result)
        proj = self.nonlinear(proj)
        proj = self.linear2(proj)
        proj = self.dropout(proj)  # Residual Dropout

        # Add & Norm
        result = tf.math.add(result, proj)
        result = self.bn2(result)

        return result


class FF(tf.keras.layers.Layer):
    def __init__(self, dims, activations):
        super(FF, self).__init__()
        assert len(dims) == len(activations)
        self.dense_layers = [
            tf.keras.layers.Dense(dim, activation=activation)
            for dim, activation in zip(dims, activations)
        ]

    @tf.function
    def call(self, x):
        for l in self.dense_layers:
            emb = l(x)
        return emb


class AdjacencyEstimator(tf.keras.layers.Layer):
    @tf.function
    def duplicate_axis(self, tensor, from_axis, to_axis, order="C"):
        """@TODO only works with exactly from_axis=1, to_axis=2 and rank(tensor)==3; generalize
        tensor: input tensor
        from_axis: the axis to duplicate
        to_axis: the position of the new duplicated axis
        order: 'C' for c-style ordering, 'F' for fortran-style ordering
        """
        tf.assert_equal(tf.constant(from_axis), tf.constant(1))
        tf.assert_equal(tf.constant(to_axis), tf.constant(2))
        tf.assert_equal(tf.rank(tensor), tf.constant(3))

        dim = tf.shape(tensor)[from_axis]
        tensor = tf.expand_dims(tensor, axis=to_axis)  # [B,L,1,maxlen]
        tensor = tf.tile(tensor, [1, dim, 1, 1])  # [B,L*L,1,maxlen]
        tensor = tf.reshape(
            tensor, [tf.shape(tensor)[0], dim, dim, tf.shape(tensor)[-1]]
        )  # [B,L,L,maxlen] controllare che reshapa bene

        tensor = tf.cond(
            tf.math.equal(order, tf.constant("F")),
            lambda: tf.transpose(tensor, perm=[0, 2, 1, 3]),
            lambda: tensor,
        )

        return tensor

    @tf.function
    def hierarchical_similarity(self, domains, **kwargs):
        """
        domains: Tensor of shape (Batch size, Seqlen)
        Returns a Tensor of shape (Batch size, Seqlen, Seqlen),
        where result[_,di,dj] is the hierarchical similarity
        between domains di and dj.
        """
        splitted = tf.strings.split(domains, sep=".")  # [B,L,?] (RaggedTensor)
        padded = tf.reverse(
            splitted, axis=[-1]
        ).to_tensor()  # [B,L,maxlen] reversed, right-padded
        # (e.g. if maxlen is 4, 'graph.facebook.com' is now ['com', 'facebook', 'graph', ''])

        padding_mask = tf.where(
            tf.math.not_equal(padded, tf.constant("", dtype=tf.string)),
            tf.ones_like(padded, dtype=tf.bool),
            tf.zeros_like(padded, dtype=tf.bool),
        )  # [B,L,maxlen] (following previous example: [True, True, True, False])

        tiled = self.duplicate_axis(
            padded, from_axis=1, to_axis=2, order="C"
        )  # [B,L,L,maxlen]
        tiled_f_order = tf.transpose(tiled, perm=[0, 2, 1, 3])  # [B,L,L,maxlen]

        commons = tf.where(
            tf.math.equal(tiled, tiled_f_order)
            | tf.math.equal(tiled, "<MASK>")
            | tf.math.equal(tiled_f_order, "<MASK>")
            | tf.math.equal(tiled, "<START>")
            | tf.math.equal(tiled_f_order, "<START>"),
            tf.ones_like(tiled, dtype=tf.bool),
            tf.zeros_like(tiled, dtype=tf.bool),
        )  # [B,L,L,maxlen]

        commons = tf.math.logical_and(
            commons, self.duplicate_axis(padding_mask, from_axis=1, to_axis=2)
        )  # [B,L,L,maxlen]

        commons = tf.math.reduce_sum(tf.cast(commons, tf.int32), axis=-1)  # [B,L,L]

        # For each pair of domains (d_i, d_j), calculate the number of subdomains
        # of the one that has fewer between d_i and d_j
        pairwise_shorter = tf.math.logical_and(
            self.duplicate_axis(padding_mask, from_axis=1, to_axis=2, order="C"),
            self.duplicate_axis(padding_mask, from_axis=1, to_axis=2, order="F"),
        )  # [B,L,L,maxlen]
        pairwise_shorter = tf.math.reduce_sum(
            tf.cast(pairwise_shorter, dtype=tf.int32), axis=-1
        )  # [B,L,L]

        # Calculate the similarity between domains d_i and d_j as the ratio
        # between the number of common subdomains and the number of subdomains
        # of the one that has fewer
        similarity = tf.math.divide(commons, pairwise_shorter)

        return tf.cast(similarity, tf.float32)

    @tf.function
    def construct_adjacency(self, similarity, type, threshold):
        return tf.case(
            [
                (
                    tf.math.equal(type, "binary"),
                    lambda: tf.where(
                        tf.math.less(similarity, threshold),
                        tf.zeros_like(similarity),
                        tf.ones_like(similarity),
                    ),
                ),
                (
                    tf.math.equal(type, "cutoff"),
                    lambda: tf.where(
                        tf.math.less(similarity, threshold),
                        tf.zeros_like(similarity),
                        similarity,
                    ),
                ),
                (
                    tf.math.equal(type, "weighted"),
                    lambda: similarity,
                ),
            ],
            exclusive=True,
        )

    @tf.function
    def _normalize(self, adj):
        """
        Normalize adjacency matrix using degree matrix.
        adj <- D^(-1/2) * adj * D^(-1/2)
        Note: adj is assumed to already contain self-loops (adj_ii == 1 in any case)
        """
        adj = tf.cast(adj, tf.float32)
        D = tf.reduce_sum(adj, axis=-1)  # [B,L]
        D = tf.linalg.diag(D)
        D = tf.math.reciprocal_no_nan(tf.math.sqrt(D))
        return tf.matmul(D, tf.matmul(adj, D))

    def __init__(self, type, threshold=0.3, normalize=True, **kwargs):
        """
        Estimate the adjacency of the domains graph.
        type (string): either 'binary', 'cutoff', or 'weighted'.
        If 'binary', the adjacency between each pair of domains is either 0 or 1
        depending on the threshold.
        If 'cutoff', the adjacency between each pair of domains is 0 if their similarity
        is lower than the threshold, and is equal to their similarity otherwise
        If 'weighted', threshold has no effect and the adjacency is equal to the similarity
        threshold (float): if binary is True, threshold is the lowest similarity
        between any two domains for them to be considered adjacent in the graph.
        If binary is False, it has no effect.
        laplacian_norm (bool): whether to normalize the resulting adjacency matrix
        using the Laplacian
        """
        super(AdjacencyEstimator, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # TensorBoard
        self.tb_writer = kwargs.get("tb_writer", None)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Adjacency Conf
        self.type = type
        self.threshold = threshold
        self.normalize = normalize

    @tf.function
    def call(self, inputs):
        # inputs [B,L]
        hierarchical_similarity = self.hierarchical_similarity(inputs, step=self.step)

        adj_h = self.construct_adjacency(
            hierarchical_similarity, self.type, self.threshold
        )
        adj_h = tf.cond(
            tf.math.equal(self.normalize, True),
            lambda: self._normalize(adj_h),
            lambda: adj_h,
        )

        # if self.tb_writer:
        #     with self.tb_writer.as_default():
        #         tf.summary.image("adj", tf.expand_dims(adj_h, axis=-1), step=0)

        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        return adj_h
