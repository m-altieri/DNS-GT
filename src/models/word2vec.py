import os
import sys
import yaml
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from colorama import Fore, Style
from utils.distribute import DummyStrategy


class Word2Vec(tf.keras.Model):
    def pretrain(self):
        self.domain_embeddings.trainable = True
        self.hidden.trainable = True
        self.out.trainable = True
        self.classifier.trainable = False
        self.finetuning = False

    def finetune(self):
        freeze = self.conf.get("freeze", False)
        self.domain_embeddings.trainable = not freeze
        self.hidden.trainable = not freeze
        self.out.trainable = not freeze
        self.classifier.trainable = True
        self.finetuning = True

    def __init__(self, conf, dist_strategy):
        super(Word2Vec, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

        # Configuration
        self.conf = conf

        # <--- REFACTORING OUT
        # default_conf_file = "conf/W2V.yaml"
        # try:
        #     with open(default_conf_file, "r") as f:
        #         default_conf = yaml.safe_load(f)
        # except OSError as e:
        #     self._logger.warning(
        #         f"{Fore.RED}Could not open conf file {default_conf_file}:\n{e}"
        #         + "\nTrying to default to argument configuration.{Style.RESET_ALL}"
        #     )
        #     default_conf = {}
        # self.conf = default_conf
        # for key in conf:
        #     if conf[key] is not None:
        #         self.conf[key] = conf[key]
        # --->

        assert self.conf["type"] == "CBOW" or self.conf["type"] == "SkipGram"
        self.finetuning = False

        if (
            self.conf.get("test_fold") is not None
        ):  # TODO the whole test folds thing should be refactored out
            fold = np.load(
                os.path.join(
                    self.conf.get("test_folds_path"),
                    f"partition-{self.conf.get('test_partition')}",
                    f"fold-{self.conf.get('test_fold')}.npy",
                )
            )
            self.test_fold = tf.constant(fold)

        # Distribution
        self.dist_strategy = (
            dist_strategy  # self.conf.get("dist_strategy", DummyStrategy)
        )
        self.distributed = self.dist_strategy is not DummyStrategy
        if self.distributed:
            self._logger.info(
                f"Initializing model with distribution strategy: {self.dist_strategy}"
            )

        # TensorBoard Init
        TB_FOLDER = "../tensorboard"
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
            self.tb_path = tf.summary.create_file_writer(
                os.path.join(TB_FOLDER, "tmp")
            )
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)

        # TODO <--- this is copied; make it external
        self.domains_vocabulary = (
            open(self.conf.get("domains_vocab_path"), "r").read().split("\n")
        )
        if self.conf.get("max_tokens"):
            self.domains_vocabulary = self.domains_vocabulary[
                : self.conf.get("max_tokens")
            ]
            self._logger.critical(
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
        self.hidden = tf.keras.layers.Dense(self.conf["dim"], activation=None)
        self.out = tf.keras.layers.Dense(self.ndomains)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        target_idx, context_idx = inputs

        target_embs = self.domain_embeddings(target_idx)  # [B, dim]
        context_embs = self.domain_embeddings(context_idx)  # [B, L-1, dim]

        if self.finetuning:
            embs = tf.concat(
                [context_embs, tf.expand_dims(target_embs, axis=1)], axis=1
            )
            hidden = self.hidden(embs)  # [B, L, dim]

        else:  # if not self.finetuning
            if self.conf["type"] == "CBOW":

                context_emb = tf.math.reduce_sum(context_embs, axis=1)
                hidden = self.hidden(context_emb)  # [B, dim]

            elif self.conf["type"] == "SkipGram":
                hidden = self.hidden(target_embs)  # [B, dim]

        out = self.out(hidden)
        out = tf.nn.softmax(out)  # [B, vsize]
        c = self.classifier(hidden)  # [B, L]

        return out if not self.finetuning else c

    @tf.function
    def distributed_train_step(self, seq):
        loss = self.dist_strategy.run(self.train_step, args=(seq,))
        return self.dist_strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None
        )

    @tf.function
    def distributed_test_step(self, seq):
        loss = self.dist_strategy.run(self.test_step, args=(seq,))
        return self.dist_strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None
        )

    def train_step(self, seq):

        # seq: [B,L,1] or [B,L,2] se --ft
        L = tf.shape(seq)[1]

        # Get target domains (central ones) and context domains (others)
        domains = seq[..., 0]  # [B, L]
        target_domains = domains[:, L // 2]
        context_domains = tf.concat(
            [domains[:, : L // 2], domains[:, L // 2 + 1 :]], axis=-1
        )
        target_indexes = self.domain_lookup(target_domains)
        context_indexes = self.domain_lookup(context_domains)

        with tf.GradientTape() as tape:

            # Get predictions: [B, vsize] or [B, L]
            pred = self((target_indexes, context_indexes), training=True)

            # Create boolean mask with True if that domain is in test fold
            if self.finetuning:
                in_fold_mask = tf.math.reduce_any(
                    tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                    axis=-1,
                )  # [B, L]

            if self.conf["type"] == "CBOW":
                if not self.finetuning:  # CBOW Pretraining
                    loss = self.compiled_loss(
                        target_indexes, pred, regularization_losses=self.losses
                    )

                else:  # CBOW Finetuning
                    loss = self.compiled_loss(
                        tf.boolean_mask(
                            tf.strings.to_number(seq[..., -1], tf.float32),
                            ~in_fold_mask,
                        ),
                        tf.boolean_mask(pred, ~in_fold_mask),
                        regularization_losses=self.losses,
                    )

            elif self.conf["type"] == "SkipGram":
                if not self.finetuning:  # SkipGram Pretraining
                    loss = self.compiled_loss(
                        context_indexes,  # [B, L-1]
                        tf.repeat(
                            tf.expand_dims(pred, axis=1),
                            tf.shape(context_indexes)[1],
                            axis=1,
                        ),  # [B, L, vsize]
                        regularization_losses=self.losses,
                    )
                else:  # SkipGram Finetuning
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
                tf.summary.scalar(
                    "train_loss", loss, step=self.train_step_count
                )

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

        # seq: [B,L,1] or [B,L,2] se --ft
        L = tf.shape(seq)[1]

        # Get target domains (central ones) and context domains (others)
        domains = seq[..., 0]  # [B, L]
        target_domains = domains[:, L // 2]
        context_domains = tf.concat(
            [domains[:, : L // 2], domains[:, L // 2 + 1 :]], axis=-1
        )
        target_indexes = self.domain_lookup(target_domains)
        context_indexes = self.domain_lookup(context_domains)

        # Get predictions: [B, vsize] or [B, L]
        pred = self((target_indexes, context_indexes), training=False)

        # Create boolean mask with True if that domain is in test fold
        if self.finetuning:
            in_fold_mask = tf.math.reduce_any(
                tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                axis=-1,
            )  # [B, L]

        if self.conf["type"] == "CBOW":
            if not self.finetuning:  # CBOW Pretraining
                loss = self.compiled_loss(
                    target_indexes, pred, regularization_losses=self.losses
                )

            else:  # CBOW Finetuning
                loss = self.compiled_loss(
                    tf.boolean_mask(
                        tf.strings.to_number(seq[..., -1], tf.float32),
                        ~in_fold_mask,
                    ),
                    tf.boolean_mask(pred, ~in_fold_mask),
                    regularization_losses=self.losses,
                )

        elif self.conf["type"] == "SkipGram":
            if not self.finetuning:  # SkipGram Pretraining
                loss = self.compiled_loss(
                    context_indexes,  # [B, L-1]
                    tf.repeat(
                        tf.expand_dims(pred, axis=1),
                        tf.shape(context_indexes)[1],
                        axis=1,
                    ),  # [B, L, vsize]
                    regularization_losses=self.losses,
                )
            else:  # SkipGram Finetuning
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
                tf.summary.scalar("train_loss", loss, step=self.val_step_count)

        # TensorBoard -- Increment val step
        self.val_step_count.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return loss

    def _predict(self, seq):

        # seq: [B,L,1] or [B,L,2] se --ft
        L = tf.shape(seq)[1]

        # Get target domains (central ones) and context domains (others)
        domains = seq[..., 0]  # [B, L]
        target_domains = domains[:, L // 2]
        context_domains = tf.concat(
            [domains[:, : L // 2], domains[:, L // 2 + 1 :]], axis=-1
        )
        target_indexes = self.domain_lookup(target_domains)
        context_indexes = self.domain_lookup(context_domains)

        # Create boolean mask with True if that domain is in test fold
        if self.finetuning:
            in_fold_mask = tf.math.reduce_any(
                tf.equal(tf.expand_dims(domains, axis=-1), self.test_fold),
                axis=-1,
            )  # [B, L]

        # Get predictions: [B, vsize] or [B, L]
        pred = self((target_indexes, context_indexes), training=True)

        return pred, None, in_fold_mask
