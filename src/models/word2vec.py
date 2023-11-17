import os
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
from datetime import datetime

from utils.nn import FF
from utils.distribute import DummyStrategy


class ParallelW2V(tf.keras.Model):
    def pretrain(self):
        self.domain_embeddings.trainable = True
        self.out.trainable = True
        self.classifier.trainable = False
        self.finetuning = False

    def finetune(self):
        freeze = self.conf.get("freeze", False)
        self.domain_embeddings.trainable = not freeze
        self.out.trainable = not freeze
        self.classifier.trainable = True
        self.finetuning = True

    @tf.function
    def compute_M(self, indexes):
        # indexes: [B,L]
        B, L = indexes.shape

        # Create M from the band matrix to select the context domains
        radius = L  # hyperparameter
        band_matrix = tf.linalg.band_part(
            tf.ones([L, L]), num_lower=radius, num_upper=radius
        )
        M = band_matrix - tf.eye(L)  # [L, L]
        M = tf.cast(M, dtype=tf.int32)

        # Set <PAD> indexes to 0 in M to exclude them from the context:
        # > get the index for <PAD>
        pad_index = self.domain_lookup(b"<PAD>")

        # > repeat indexes for each row
        # repeated_domain_indexes:
        # [[d1 d2 ... dL],
        #  ...
        #  [d1 d2 ... dL]]
        repeated_domain_indexes = tf.einsum(
            "iu,buj->bij",
            tf.ones([L, 1], dtype=tf.int64),
            tf.expand_dims(indexes, axis=1),
        )  # [L,1] x [B,1,L] -> [B,L,L]

        # > create boolean mask
        M = tf.where(
            tf.math.equal(repeated_domain_indexes, pad_index),
            0,
            M,
        )
        M = tf.where(
            tf.math.equal(tf.transpose(repeated_domain_indexes, [0, 2, 1]), pad_index),
            0,
            M,
        )

        return M

    def __init__(self, conf, dist_strategy):
        super().__init__()

        # Configuration
        self.conf = conf
        assert self.conf["type"] == "CBOW" or self.conf["type"] == "SkipGram"

        self.finetuning = False
        self.initialized = False

        # TODO the whole test folds thing should be refactored out
        if self.conf.get("test_fold") is not None:
            fold = np.load(
                os.path.join(
                    self.conf.get("test_folds_path"),
                    f"partition-{self.conf.get('test_partition')}",
                    f"fold-{self.conf.get('test_fold')}.npy",
                )
            )
            self.test_fold = tf.constant(fold)

        # Distribution
        self.dist_strategy = dist_strategy
        self.distributed = self.dist_strategy is not DummyStrategy
        if self.distributed:
            print(
                f"Initializing model with distribution strategy: {self.dist_strategy}"
            )

        # TensorBoard
        TB_FOLDER = f"../tensorboard/{conf.get('model')}"
        self.tb_path = None
        self.train_step_count = tf.Variable(
            0, trainable=False, dtype=tf.int64, name="train_step_count"
        )
        self.val_step_count = tf.Variable(
            0, trainable=False, dtype=tf.int64, name="val_step_count"
        )
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

        # <--- this is the same as DELM; make it external?
        self.domains_vocabulary = (
            open(self.conf.get("domains_vocab_path"), "r").read().split("\n")
        )
        if self.conf.get("max_tokens"):
            self.domains_vocabulary = self.domains_vocabulary[
                : self.conf.get("max_tokens")
            ]
            # If I truncate, I have to add back the special ones that are now excluded
            self.domains_vocabulary.append("<PAD>")
            print(
                f"Truncating the vocabulary to the first {self.conf.get('max_tokens')} tokens."
            )
        self.domains_vocabulary = tf.constant(self.domains_vocabulary)

        # Token Indexes Lookup
        self.domain_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.domains_vocabulary, num_oov_indices=1
        )
        # --->

        # Layers
        self.ndomains = self.domain_lookup.vocabulary_size()
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["dim"],
        )

        self.out = FF(
            [self.conf["dim"], self.ndomains], [None, None]
        )  # tf.keras.layers.Dense(self.ndomains)
        self.classifier = FF(
            [self.conf["dim"], 1], [None, "sigmoid"]
        )  # tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, domain_indexes):
        # domains: [B, L]
        L = tf.shape(domain_indexes)[1]
        domain_embs = self.domain_embeddings(domain_indexes)  # [B, L, dim]

        if self.conf["type"] == "CBOW":
            M = self.compute_M(domain_indexes)  # [B, L, L]
            C = tf.linalg.matmul(tf.cast(M, tf.float32), domain_embs)  # [B, L, dim]

        else:
            C = domain_embs  # [B, L, dim]

        # Initialize all weights
        if not self.initialized:
            out = self.out(C)
            out = tf.nn.softmax(out)
            classification = self.classifier(C)
            self.initialized = True

        if not self.finetuning:
            out = self.out(C)  # Dense(dim) -> Dense(vsize) -> [B,L,vsize]
            out = tf.nn.softmax(out)
        else:
            classification = self.classifier(C)  # Dense(1) -> [B,L]

        return out if not self.finetuning else classification

    @tf.function
    def distributed_train_step(self, seq):
        loss = self.dist_strategy.run(self.train_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    @tf.function
    def distributed_test_step(self, seq):
        loss = self.dist_strategy.run(self.test_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    def train_step(self, seq):
        # seq: [B,L,2], or [B,L,3] if --ft
        B, L, _ = tf.shape(seq)
        domains = seq[..., 1]  # [B, L]

        # Create boolean mask with True if that domain is in test fold
        if self.finetuning:
            in_fold_mask = tf.math.reduce_any(
                tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                axis=-1,
            )  # [B, L]

        with tf.GradientTape() as tape:
            domain_indexes = self.domain_lookup(domains)
            # domain_embs = self.domain_embeddings(domain_indexes)

            # Get predictions: [B, vsize] or [B, L]
            pred = self(domain_indexes, training=True)

            if self.conf["type"] == "CBOW":
                if not self.finetuning:
                    # CBOW Pretraining
                    M = self.compute_M(domain_indexes)

                    nonpad_tokens = tf.math.reduce_any(tf.cast(M, tf.bool), axis=-1)
                    domain_indexes = tf.boolean_mask(domain_indexes, nonpad_tokens)
                    pred = tf.boolean_mask(pred, nonpad_tokens)

                    loss = self.compiled_loss(
                        domain_indexes, pred, regularization_losses=self.losses
                    )

                else:
                    # CBOW Finetuning
                    loss = self.compiled_loss(
                        tf.boolean_mask(
                            tf.strings.to_number(seq[..., -1], tf.float32),
                            ~in_fold_mask,
                        ),
                        tf.boolean_mask(pred, ~in_fold_mask),
                        regularization_losses=self.losses,
                    )

            elif self.conf["type"] == "SkipGram":
                if not self.finetuning:
                    # SkipGram Pretraining

                    repeated_domain_indexes = tf.einsum(
                        "iu,buj->bij",
                        tf.ones([L, 1], dtype=tf.int64),
                        tf.expand_dims(domain_indexes, axis=1),
                    )  # [L,1] x [B,1,L] -> [B,L,L]
                    # repeated_domain_indexes:
                    # [[d1 d2 ... dL],
                    #  [d1 d2 ... dL],
                    #  ...
                    #  [d1 d2 ... dL]]

                    # Create matrix M, where for each row (target), 1 denotes domains that are in context
                    # M ultimately determines what (context) domains the loss will be computed for
                    radius = L  # hyperparameter
                    band_matrix = tf.linalg.band_part(
                        tf.ones([B, L, L]), num_lower=radius, num_upper=radius
                    )
                    M = band_matrix - tf.eye(L)  # [B,L,L]
                    # e.g. for radius=2 and L=5, M is:
                    # [[0 1 1 0 0],
                    #  [1 0 1 1 0],
                    #  [1 1 0 1 1],
                    #  [0 1 1 0 1],
                    #  [0 0 1 1 0]]

                    # We want to exclude <PAD> from the loss computation, so:
                    # Get the index for <PAD>
                    pad_index = self.domain_lookup(b"<PAD>")

                    # Set <PAD> rows (predictions) in M to 0
                    M = tf.where(
                        tf.math.equal(
                            tf.transpose(repeated_domain_indexes, perm=[0, 2, 1]),
                            pad_index,
                        ),
                        0,
                        M,
                    )
                    # Also set <PAD> columns (context domains) in M to 0
                    M = tf.where(
                        tf.math.equal(repeated_domain_indexes, pad_index), 0, M
                    )

                    # Take (repeated) context indexes that are in M
                    context_indexes = tf.boolean_mask(
                        repeated_domain_indexes,
                        tf.cast(M, dtype=tf.bool),
                        # axis=1,
                    )

                    # <-----
                    # TAKE PREDICTIONS THAT ARE IN M
                    # for a given context domain d_j in M_ij, the loss will be computed by
                    # comparing it with p(d_i)! predictions for each row are all the same
                    # -----
                    # NOTE This causes OOM (B x L x L x vsize) ~ 117GB with B=512, L=32, vsize=30006
                    # -----
                    # pred = tf.repeat(
                    #     tf.expand_dims(pred, axis=2), L, axis=2
                    # )  # [B,L,L,vsize]
                    # # pred: [p(d1) p(d1) ... p(d1)
                    # #        p(d2) p(d2) ... p(d2)
                    # #        ...
                    # #        p(dL) p(dL) ... p(dL)]
                    # pred = tf.boolean_mask(
                    #     pred, tf.cast(M, dtype=tf.bool)  # , axis=1
                    # )
                    # -----
                    # NOTE Attempt to rewrite it in a memory-efficient way (O(L) times more efficient)

                    # convert M to integer
                    M = tf.cast(M, dtype=tf.int16)

                    # compute how many context tokens are to be considered for each target token
                    per_token_context_cardinality = tf.reduce_sum(M, axis=-1)  # [B,L]

                    # flatten the per-token context cardinality: [B*L]
                    per_token_context_cardinality = tf.reshape(
                        per_token_context_cardinality, [-1]
                    )

                    # flatten pred: [B*L, vsize]
                    pred = tf.reshape(pred, [B * L, -1])

                    # Repeat each token in pred as many times as the corresponding element in per-token context cardinality
                    # -----
                    # ISSUE Known issue that tf.repeat is memory-inefficient: https://github.com/tensorflow/tensorflow/issues/50712
                    # pred = tf.repeat(
                    #     pred, repeats=per_token_context_cardinality, axis=0
                    # )
                    # -----
                    indices = tf.repeat(
                        tf.range(B * L), repeats=per_token_context_cardinality
                    )
                    pred = tf.gather(pred, indices, axis=0)
                    # ----->

                    # Compute loss
                    # * = number of tokens in all windows (band matrix minus identity),
                    # excluding <PAD> tokens
                    loss = self.compiled_loss(
                        context_indexes,  # [B,*]
                        pred,  # [B,*,vsize]
                        regularization_losses=self.losses,
                    )
                else:
                    # SkipGram Finetuning
                    loss = self.compiled_loss(
                        tf.boolean_mask(
                            tf.strings.to_number(seq[..., -1], tf.float32),
                            ~in_fold_mask,
                        ),
                        tf.boolean_mask(pred, ~in_fold_mask),
                        regularization_losses=self.losses,
                    )

            # Divide loss by number of GPUs if distributed
            if self.distributed:
                loss = tf.math.divide(
                    tf.math.reduce_mean(loss),
                    self.dist_strategy.num_replicas_in_sync,
                )

        # Write training loss to TensorBoard
        if self.conf.get("tensorboard"):
            with self.tb_writer.as_default():
                tf.summary.scalar("train_loss", loss, step=self.train_step_count)

        # Compute gradients and update weights, ignoring warnings about None gradients
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(
            (grad, _)
            for (grad, _) in zip(gradients, self.trainable_variables)
            if grad is not None
        )

        # TensorBoard -- Increment train step
        self.train_step_count.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return loss

    def test_step(self, seq):
        # seq: [B,L,2], or [B,L,3] if --ft
        B, L, _ = tf.shape(seq)
        domains = seq[..., 1]  # [B, L]
        domain_indexes = self.domain_lookup(domains)
        # domain_embs = self.domain_embeddings(domain_indexes)

        # Get predictions: [B, vsize] or [B, L]
        pred = self(domain_indexes, training=False)

        # Create boolean mask with True if that domain is in test fold
        if self.finetuning:
            in_fold_mask = tf.math.reduce_any(
                tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                axis=-1,
            )  # [B, L]

        if self.conf["type"] == "CBOW":
            if not self.finetuning:
                # CBOW Pretraining
                M = self.compute_M(domain_indexes)

                nonpad_tokens = tf.math.reduce_any(tf.cast(M, tf.bool), axis=-1)
                domain_indexes = tf.boolean_mask(domain_indexes, nonpad_tokens)
                pred = tf.boolean_mask(pred, nonpad_tokens)

                loss = self.compiled_loss(
                    domain_indexes, pred, regularization_losses=self.losses
                )

            else:
                # CBOW Finetuning
                loss = self.compiled_loss(
                    tf.boolean_mask(
                        tf.strings.to_number(seq[..., -1], tf.float32),
                        in_fold_mask,
                    ),
                    tf.boolean_mask(pred, in_fold_mask),
                    regularization_losses=self.losses,
                )

        elif self.conf["type"] == "SkipGram":
            if not self.finetuning:
                # SkipGram Pretraining

                repeated_domain_indexes = tf.einsum(
                    "iu,buj->bij",
                    tf.ones([L, 1], dtype=tf.int64),
                    tf.expand_dims(domain_indexes, axis=1),
                )  # [L,1] x [B,1,L] -> [B,L,L]
                # repeated_domain_indexes:
                # [[d1 d2 ... dL],
                #  [d1 d2 ... dL],
                #  ...
                #  [d1 d2 ... dL]]

                # Create matrix M, where for each row (target), 1 denotes domains that are in context
                # M ultimately determines what (context) domains the loss will be computed for
                radius = L  # hyperparameter
                band_matrix = tf.linalg.band_part(
                    tf.ones([B, L, L]), num_lower=radius, num_upper=radius
                )
                M = band_matrix - tf.eye(L)  # [B,L,L]
                # e.g. for radius=2 and L=5, M is:
                # [[0 1 1 0 0],
                #  [1 0 1 1 0],
                #  [1 1 0 1 1],
                #  [0 1 1 0 1],
                #  [0 0 1 1 0]]

                # We want to exclude <PAD> from the loss computation, so:
                # Get the index for <PAD>
                pad_index = self.domain_lookup(b"<PAD>")
                # print(pad_index)

                # Set <PAD> rows (predictions) in M to 0
                M = tf.where(
                    tf.math.equal(
                        tf.transpose(repeated_domain_indexes, perm=[0, 2, 1]),
                        pad_index,
                    ),
                    0,
                    M,
                )
                # Also set <PAD> columns (context domains) in M to 0
                M = tf.where(tf.math.equal(repeated_domain_indexes, pad_index), 0, M)

                # Take (repeated) context indexes that are in M
                context_indexes = tf.boolean_mask(
                    repeated_domain_indexes,
                    tf.cast(M, dtype=tf.bool),
                    # axis=1,
                )

                # <-----
                # TAKE PREDICTIONS THAT ARE IN M
                # for a given context domain d_j in M_ij, the loss will be computed by
                # comparing it with p(d_i)! predictions for each row are all the same
                # -----
                # NOTE This causes OOM (B x L x L x vsize) ~ 117GB with B=512, L=32, vsize=30006
                # -----
                # pred = tf.repeat(
                #     tf.expand_dims(pred, axis=2), L, axis=2
                # )  # [B,L,L,vsize]
                # # pred: [p(d1) p(d1) ... p(d1)
                # #        p(d2) p(d2) ... p(d2)
                # #        ...
                # #        p(dL) p(dL) ... p(dL)]
                # pred = tf.boolean_mask(
                #     pred, tf.cast(M, dtype=tf.bool)  # , axis=1
                # )
                # -----
                # NOTE Attempt to rewrite it in a memory-efficient way (O(L) times more efficient)

                # convert M to integer
                M = tf.cast(M, dtype=tf.int16)

                # compute how many context tokens are to be considered for each target token
                per_token_context_cardinality = tf.reduce_sum(M, axis=-1)  # [B,L]

                # flatten the per-token context cardinality: [B*L]
                per_token_context_cardinality = tf.reshape(
                    per_token_context_cardinality, [-1]
                )

                # flatten pred: [B*L, vsize]
                pred = tf.reshape(pred, [B * L, -1])

                # Repeat each token in pred as many times as the corresponding element in per-token context cardinality
                # -----
                # ISSUE Known issue that tf.repeat is memory-inefficient: https://github.com/tensorflow/tensorflow/issues/50712
                # pred = tf.repeat(
                #     pred, repeats=per_token_context_cardinality, axis=0
                # )
                # -----
                indices = tf.repeat(
                    tf.range(B * L), repeats=per_token_context_cardinality
                )
                pred = tf.gather(pred, indices, axis=0)
                # ----->

                # Compute loss
                # * = number of tokens in all windows (band matrix minus identity),
                # excluding <PAD> tokens
                loss = self.compiled_loss(
                    context_indexes,  # [B,*]
                    pred,  # [B,*,vsize]
                    regularization_losses=self.losses,
                )

            else:
                # SkipGram Finetuning
                loss = self.compiled_loss(
                    tf.boolean_mask(
                        tf.strings.to_number(seq[..., -1], tf.float32),
                        in_fold_mask,
                    ),
                    tf.boolean_mask(pred, in_fold_mask),
                    regularization_losses=self.losses,
                )

        # Divide loss by number of GPUs if distributed
        if self.distributed:
            loss = tf.math.divide(
                tf.math.reduce_mean(loss),
                self.dist_strategy.num_replicas_in_sync,
            )

        # Write training loss to TensorBoard
        if self.conf.get("tensorboard"):
            with self.tb_writer.as_default():
                tf.summary.scalar("val_loss", loss, step=self.val_step_count)

        # TensorBoard -- Increment val step
        self.val_step_count.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return loss

    def _predict(self, seq, mask=None):  # mask only used for compatibility
        # seq: [B,L,2] or [B,L,3] se --ft

        domains = seq[..., 1]  # [B, L]
        domain_indexes = self.domain_lookup(domains)

        # Get predictions: [B, vsize] or [B, L]
        pred = self(domain_indexes, training=False)

        # Create boolean mask with True if that domain is in test fold
        if self.finetuning:
            in_fold_mask = tf.math.reduce_any(
                tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                axis=-1,
            )  # [B, L]

            loss = self.compiled_loss(
                tf.boolean_mask(
                    tf.strings.to_number(seq[..., -1], tf.float32),
                    in_fold_mask,
                ),
                tf.boolean_mask(pred, in_fold_mask),
                regularization_losses=self.losses,
            )
        else:
            raise NotImplementedError()

        return pred, loss, in_fold_mask


# class Word2Vec(tf.keras.Model):
#     def pretrain(self):
#         self.domain_embeddings.trainable = True
#         # self.hidden.trainable = True
#         self.out.trainable = True
#         self.classifier.trainable = False
#         self.finetuning = False

#     def finetune(self):
#         freeze = self.conf.get("freeze", False)
#         self.domain_embeddings.trainable = not freeze
#         # self.hidden.trainable = not freeze
#         self.out.trainable = not freeze
#         self.classifier.trainable = True
#         self.finetuning = True

#     def __init__(self, conf, dist_strategy):
#         super().__init__()

#         # Logger
#         self._logger = logging.getLogger(__name__)
#         self._logger.setLevel(logging.INFO)
#         self._logger.addHandler(logging.StreamHandler(sys.stdout))

#         # Configuration
#         self.conf = conf
#         assert self.conf["type"] == "CBOW" or self.conf["type"] == "SkipGram"

#         self.finetuning = False

#         # TODO the whole test folds thing should be refactored out
#         if self.conf.get("test_fold") is not None:
#             fold = np.load(
#                 os.path.join(
#                     self.conf.get("test_folds_path"),
#                     f"partition-{self.conf.get('test_partition')}",
#                     f"fold-{self.conf.get('test_fold')}.npy",
#                 )
#             )
#             self.test_fold = tf.constant(fold)

#         # Distribution
#         self.dist_strategy = dist_strategy
#         self.distributed = self.dist_strategy is not DummyStrategy
#         if self.distributed:
#             self._logger.info(
#                 f"Initializing model with distribution strategy: {self.dist_strategy}"
#             )

#         # TensorBoard
#         TB_FOLDER = f"../tensorboard/{conf.get('model')}"
#         self.tb_path = None
#         self.train_step_count = tf.Variable(
#             0, trainable=False, dtype=tf.int64, name="train_step_count"
#         )
#         self.val_step_count = tf.Variable(
#             0, trainable=False, dtype=tf.int64, name="val_step_count"
#         )
#         if not os.path.exists(TB_FOLDER):
#             os.makedirs(TB_FOLDER)
#         if not self.conf["quick_tb"]:
#             self.tb_path = os.path.join(
#                 TB_FOLDER,
#                 self.conf.get("run_name", None)
#                 or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
#             )
#         else:
#             self.tb_path = tf.summary.create_file_writer(os.path.join(TB_FOLDER, "tmp"))
#         self.tb_writer = tf.summary.create_file_writer(self.tb_path)

#         # <--- this is the same as DELM; make it external?
#         self.domains_vocabulary = (
#             open(self.conf.get("domains_vocab_path"), "r").read().split("\n")
#         )
#         if self.conf.get("max_tokens"):
#             self.domains_vocabulary = self.domains_vocabulary[
#                 : self.conf.get("max_tokens")
#             ]
#             # If I truncate, I have to add back the special ones that are now excluded
#             self.domains_vocabulary.append("<PAD>")
#             self._logger.critical(
#                 f"Truncating the vocabulary to the first {self.conf.get('max_tokens')} tokens."
#             )
#         self.domains_vocabulary = tf.constant(self.domains_vocabulary)

#         # Token Indexes Lookup
#         self.domain_lookup = tf.keras.layers.StringLookup(
#             vocabulary=self.domains_vocabulary, num_oov_indices=1
#         )
#         # --->

#         self.initialized = False

#         # Layers
#         self.ndomains = self.domain_lookup.vocabulary_size()
#         self.domain_embeddings = tf.keras.layers.Embedding(
#             input_dim=self.ndomains,
#             output_dim=self.conf["dim"],
#         )

#         # self.hidden = tf.keras.layers.Dense(self.conf["dim"], activation=None)
#         self.out = FF(
#             [self.conf["dim"], self.ndomains], [None, None]
#         )  # tf.keras.layers.Dense(self.ndomains)
#         self.classifier = FF(
#             [1], ["sigmoid"]
#         )  # tf.keras.layers.Dense(1, activation="sigmoid")

#     def call(self, seq):
#         # seq: [B,L,2], or [B,L,3] if --ft
#         L = tf.shape(seq)[1]

#         # Get target domains (central ones) and context domains (others)
#         domains = seq[..., 1]  # [B, L]
#         domain_indexes = self.domain_lookup(domains)
#         domain_embs = self.domain_embeddings(domain_indexes)

#         if not self.finetuning:
#             if self.conf["type"] == "CBOW":
#                 context_embs = tf.concat(
#                     [domain_embs[:, : L // 2], domain_embs[:, L // 2 + 1 :]],
#                     axis=1,
#                 )
#                 context_emb = tf.math.reduce_sum(context_embs, axis=1)
#                 # hidden = self.hidden(context_emb)  # [B, dim]
#                 hidden = context_emb

#             elif self.conf["type"] == "SkipGram":
#                 target_embs = domain_embs[:, L // 2]
#                 # <--- POSSIBILE PUNTO DI ROTTURA
#                 hidden = target_embs  # [B, dim]
#                 # hidden = self.hidden(target_embs)  # [B, dim]
#                 # --->

#         else:
#             # <--- POSSIBILE PUNTO DI ROTTURA
#             hidden = domain_embs  # [B, L, dim]
#             # hidden = self.hidden(domain_embs)  # [B, L, dim]
#             # --->

#         # Initialize all weights
#         if not self.initialized:
#             out = self.out(hidden)
#             out = tf.nn.softmax(out)
#             c = self.classifier(hidden)
#             self.initialized = True

#         if not self.finetuning:
#             out = self.out(hidden)  # Dense(dim) -> Dense(vsize) -> [B, vsize]
#             out = tf.nn.softmax(out)
#         else:
#             c = self.classifier(hidden)  # Dense(1) -> [B, L]

#         return out if not self.finetuning else c

#     @tf.function
#     def distributed_train_step(self, seq):
#         loss = self.dist_strategy.run(self.train_step, args=(seq,))
#         return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

#     @tf.function
#     def distributed_test_step(self, seq):
#         loss = self.dist_strategy.run(self.test_step, args=(seq,))
#         return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

#     def train_step(self, seq):
#         L = tf.shape(seq)[1]
#         domains = seq[..., 1]
#         target_domains = domains[:, L // 2]
#         context_domains = tf.concat(
#             [domains[:, : L // 2], domains[:, L // 2 + 1 :]], axis=1
#         )
#         target_indexes = self.domain_lookup(target_domains)
#         context_indexes = self.domain_lookup(context_domains)

#         # Create boolean mask with True if that domain is in test fold
#         if self.finetuning:
#             in_fold_mask = tf.math.reduce_any(
#                 tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
#                 axis=-1,
#             )  # [B, L]

#         with tf.GradientTape() as tape:
#             # Get predictions: [B, vsize] or [B, L]
#             pred = self(seq, training=True)

#             if self.conf["type"] == "CBOW":
#                 if not self.finetuning:
#                     # CBOW Pretraining
#                     loss = self.compiled_loss(
#                         target_indexes, pred, regularization_losses=self.losses
#                     )

#                 else:
#                     # CBOW Finetuning
#                     loss = self.compiled_loss(
#                         tf.boolean_mask(
#                             tf.strings.to_number(seq[..., -1], tf.float32),
#                             ~in_fold_mask,
#                         ),
#                         tf.boolean_mask(pred, ~in_fold_mask),
#                         regularization_losses=self.losses,
#                     )

#             elif self.conf["type"] == "SkipGram":
#                 if not self.finetuning:
#                     # SkipGram Pretraining
#                     loss = self.compiled_loss(
#                         context_indexes,  # [B, L-1]
#                         tf.repeat(
#                             tf.expand_dims(pred, axis=1),
#                             tf.shape(context_indexes)[1],
#                             axis=1,
#                         ),  # [B, L-1, vsize]
#                         regularization_losses=self.losses,
#                     )
#                 else:
#                     # SkipGram Finetuning
#                     loss = self.compiled_loss(
#                         tf.boolean_mask(
#                             tf.strings.to_number(seq[..., -1], tf.float32),
#                             ~in_fold_mask,
#                         ),
#                         tf.boolean_mask(pred, ~in_fold_mask),
#                         regularization_losses=self.losses,
#                     )

#             # Divide loss by number of GPUs if distributed
#             if self.distributed:
#                 loss = tf.math.divide(
#                     tf.math.reduce_mean(loss),
#                     self.dist_strategy.num_replicas_in_sync,
#                 )

#         # Write training loss to TensorBoard
#         if self.conf.get("tensorboard"):
#             with self.tb_writer.as_default():
#                 tf.summary.scalar("train_loss", loss, step=self.train_step_count)

#         # Compute gradients and update weights, ignoring warnings about None gradients
#         trainable_variables = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_variables)
#         self.optimizer.apply_gradients(
#             (grad, _)
#             for (grad, _) in zip(gradients, self.trainable_variables)
#             if grad is not None
#         )

#         # TensorBoard -- Increment train step
#         self.train_step_count.assign_add(tf.constant(1, dtype=tf.int64))

#         # Return a dict mapping metric names to current value
#         return loss

#     def test_step(self, seq):
#         L = tf.shape(seq)[1]
#         domains = seq[..., 1]
#         target_domains = domains[:, L // 2]
#         context_domains = tf.concat(
#             [domains[:, : L // 2], domains[:, L // 2 + 1 :]], axis=1
#         )
#         target_indexes = self.domain_lookup(target_domains)
#         context_indexes = self.domain_lookup(context_domains)

#         # Get predictions: [B, vsize] or [B, L]
#         pred = self(seq, training=False)

#         # Create boolean mask with True if that domain is in test fold
#         if self.finetuning:
#             in_fold_mask = tf.math.reduce_any(
#                 tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
#                 axis=-1,
#             )  # [B, L]

#         if self.conf["type"] == "CBOW":
#             if not self.finetuning:  # CBOW Pretraining
#                 loss = self.compiled_loss(
#                     target_indexes, pred, regularization_losses=self.losses
#                 )

#             else:  # CBOW Finetuning
#                 loss = self.compiled_loss(
#                     tf.boolean_mask(
#                         tf.strings.to_number(seq[..., -1], tf.float32),
#                         in_fold_mask,
#                     ),
#                     tf.boolean_mask(pred, in_fold_mask),
#                     regularization_losses=self.losses,
#                 )

#         elif self.conf["type"] == "SkipGram":
#             if not self.finetuning:  # SkipGram Pretraining
#                 loss = self.compiled_loss(
#                     context_indexes,  # [B, L-1]
#                     tf.repeat(
#                         tf.expand_dims(pred, axis=1),
#                         tf.shape(context_indexes)[1],
#                         axis=1,
#                     ),  # [B, L, vsize]
#                     regularization_losses=self.losses,
#                 )
#             else:  # SkipGram Finetuning
#                 loss = self.compiled_loss(
#                     tf.boolean_mask(
#                         tf.strings.to_number(seq[..., -1], tf.float32),
#                         in_fold_mask,
#                     ),
#                     tf.boolean_mask(pred, in_fold_mask),
#                     regularization_losses=self.losses,
#                 )

#         # Divide loss by number of GPUs if distributed
#         if self.distributed:
#             loss = tf.math.divide(
#                 tf.math.reduce_mean(loss),
#                 self.dist_strategy.num_replicas_in_sync,
#             )

#         # Write training loss to TensorBoard
#         if self.conf.get("tensorboard"):
#             with self.tb_writer.as_default():
#                 tf.summary.scalar("train_loss", loss, step=self.val_step_count)

#         # TensorBoard -- Increment val step
#         self.val_step_count.assign_add(tf.constant(1, dtype=tf.int64))

#         # Return a dict mapping metric names to current value
#         return loss

#     def _predict(self, seq):
#         # seq: [B,L,2] or [B,L,3] se --ft

#         domains = seq[..., 1]  # [B, L]

#         # Create boolean mask with True if that domain is in test fold
#         if self.finetuning:
#             in_fold_mask = tf.math.reduce_any(
#                 tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
#                 axis=-1,
#             )  # [B, L]

#         # Get predictions: [B, vsize] or [B, L]
#         pred = self(seq, training=False)

#         return pred, None, in_fold_mask
