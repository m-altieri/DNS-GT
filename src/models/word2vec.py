import tensorflow as tf
import yaml
from colorama import Fore, Style
import sys
import numpy as np
import logging
import os
from datetime import datetime


class Word2Vec(tf.keras.Model):
    def __init__(self, **conf):
        super(Word2Vec, self).__init__()

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.StreamHandler(sys.stdout))

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
        assert self.conf["type"] == "CBOW" or self.conf["type"] == "SkipGram"

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

        self.domain_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.conf["domains_vocab_path"], num_oov_indices=0
        )
        self.ndomains = self.domain_lookup.vocabulary_size()
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["dim"],
        )

        self.hidden = tf.keras.layers.Dense(self.conf["dim"], activation=None)
        self.out = tf.keras.layers.Dense(self.ndomains)

    @tf.function
    def call(self, inputs):
        target_idx, context_idx = inputs

        target_embs = self.domain_embeddings(target_idx)
        context_embs = self.domain_embeddings(context_idx)

        if self.conf["type"] == "CBOW":  # for CBOW, x is the context, y is the target
            context_emb = tf.math.reduce_sum(context_embs, axis=1)
            hidden = self.hidden(context_emb)
            out = self.out(hidden)
            out = tf.nn.softmax(out)
        elif (
            self.conf["type"] == "SkipGram"
        ):  # for SkipGram, x is the target, y is the context
            hidden = self.hidden(target_embs)
            out = self.out(hidden)
            out = tf.nn.softmax(out)

        return out

    def train_step(self, seq):
        # seq: [B,L]
        L = tf.shape(seq)[-1]

        target_domains = seq[:, L // 2]
        context_domains = tf.concat([seq[:, : L // 2], seq[:, L // 2 + 1 :]], axis=-1)

        target_indexes = self.domain_lookup(target_domains)
        context_indexes = self.domain_lookup(context_domains)

        with tf.GradientTape() as tape:
            pred = self((target_indexes, context_indexes), training=True)  # [B,vsize]

            if self.conf["type"] == "CBOW":
                loss = self.compiled_loss(
                    target_indexes,
                    pred,
                    regularization_losses=self.losses,
                )
            elif self.conf["type"] == "SkipGram":
                loss = tf.reduce_mean(
                    tf.map_fn(
                        lambda e: self.compiled_loss(
                            e, pred, regularization_losses=self.losses
                        ),
                        tf.transpose(context_indexes),  # [L,B]
                        fn_output_signature=tf.float32,
                    ),
                    axis=-1,
                )  # [B]

        with self.tb_writer.as_default():
            tf.summary.scalar("train_loss", loss, step=self.step)

        # Compute gradients and update weights
        trainable_variables = self.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # TensorBoard -- Increment step
        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, seq):
        L = tf.shape(seq)[-1]

        target_domains = seq[:, L // 2]
        context_domains = tf.concat([seq[:, : L // 2], seq[:, L // 2 + 1 :]], axis=-1)

        target_indexes = self.domain_lookup(target_domains)
        context_indexes = self.domain_lookup(context_domains)

        pred = self((target_indexes, context_indexes), training=False)  # [B,vsize]

        if self.conf["type"] == "CBOW":
            loss = self.compiled_loss(
                target_indexes,
                pred,
                regularization_losses=self.losses,
            )
        elif self.conf["type"] == "SkipGram":
            loss = tf.reduce_mean(
                tf.map_fn(
                    lambda e: self.compiled_loss(
                        e, pred, regularization_losses=self.losses
                    ),
                    tf.transpose(context_indexes),  # [L,B]
                    fn_output_signature=tf.float32,
                ),
                axis=-1,
            )  # [B]

        with self.tb_writer.as_default():
            tf.summary.scalar("val_loss", loss, step=self.step)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def create_pairs(seq, seqlen):
        # Input: all queries
        # Output: sequences having target domain at the middle. [..., context_-2, context_-1, target, context_+1, context_+2, ...]
        assert seqlen % 2 == 1
        window = seqlen // 2
        pairs = []
        for index, target in enumerate(seq):
            left_overflow = max(window - index, 0)
            right_overflow = max(window - (len(seq) - index - 1), 0)
            pairs.append(
                seq[
                    max(0, index - window - right_overflow) : min(
                        index + window + 1 + left_overflow, len(seq)
                    )
                ]
            )
        return pairs
