import os
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
# from lib.tf_matplotlib import tfmpl

import pytftk.nn
from pytftk.logbooks import TBManager
from pytftk.distribute import DummyStrategy

from utils.constants import Constants
from utils.graphs import (
    AdjacencyEstimator,
    TrivialAdjacencyEstimator,
    HierarchicalSimilarityEstimator,
)


class DNS_GT(tf.keras.Model):
    def __init__(self, conf, dist_strategy):
        super().__init__()

        # Configuration
        self.conf = conf

        # Distribution
        self.dist_strategy = dist_strategy
        self.distributed = self.dist_strategy is not DummyStrategy
        if self.distributed:
            print(
                f"Initializing model with distribution strategy: {self.dist_strategy}"
            )

        self.finetuning = False
        self.initialized = False  # TODO if this is a tf.Variable(False), and i modify it with .assign(), the weights won't save

        # TensorBoard Init
        self.tb_manager = TBManager(
            f"../tensorboard/{conf.get('model')}",
            self.conf.get("run_name"),
            tmp=self.conf.get("quick_tb"),
            interval=1 if self.conf.get("demo") else None,
            enabled=self.conf.get("tensorboard"),
            verbose=self.conf.get("verbose"),
            port=self.conf.get("tb_port"),
        )
        if self.conf.get("tensorboard"):
            self.tb_manager.run()

        # Token Adjacency
        self.adj_estimators: list[AdjacencyEstimator] = []
        if self.conf.get("adj_estimator"):
            self.adj_estimators.append(
                HierarchicalSimilarityEstimator(
                    kind="binary", normalize=False, tb_manager=self.tb_manager
                )
            )
        if len(self.adj_estimators) == 0:
            # the "base" adjacency. i need something to logical_end pad_adj with
            self.adj_estimators.append(TrivialAdjacencyEstimator())

        # Load test fold if needed (finetuning)
        if self.conf.get("test_fold") is not None:
            fold = np.load(
                os.path.join(
                    self.conf.get("data_path"),
                    "test_folds",
                    f"partition-{self.conf.get('test_partition')}",
                    f"fold-{self.conf.get('test_fold')}.npy",
                )
            )
            self.test_fold = tf.constant(fold)

        # Load vocabularies
        self.hosts_vocabulary = (
            open(
                os.path.join(self.conf.get("data_path"), "vocab", "hosts_vocab.txt"),
                "r",
            )
            .read()
            .split("\n")
        )
        self.domains_vocabulary = (
            open(
                os.path.join(self.conf.get("data_path"), "vocab", "domains_vocab.txt"),
                "r",
            )
            .read()
            .split("\n")
        )

        # NOTE if max_tokens, I am trimming both hosts and domains to max_tokens;
        # in theory hosts are a lot less problematic and could be left untrimmed
        if self.conf.get("max_tokens"):
            self.hosts_vocabulary = self.hosts_vocabulary[: self.conf.get("max_tokens")]
            self.domains_vocabulary = self.domains_vocabulary[
                : self.conf.get("max_tokens")
            ]
            # If I truncate, I have to add back the special ones that are now excluded
            self.hosts_vocabulary.append("<PAD>")
            self.domains_vocabulary.append("<PAD>")
            print(
                f"Truncating the vocabulary to the first {self.conf.get('max_tokens')} tokens."
            )

        self.hosts_vocabulary = tf.constant(self.hosts_vocabulary)
        self.domains_vocabulary = tf.constant(self.domains_vocabulary)

        # Token Indexes Lookup
        self.hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.hosts_vocabulary,
            num_oov_indices=1,
            name="hosts_lookup",
        )
        self.inverse_hosts_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.hosts_vocabulary,
            num_oov_indices=1,
            invert=True,
            name="inverse_hosts_lookup",
        )
        self.domains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.domains_vocabulary,
            num_oov_indices=1,
            name="domains_lookup",
        )
        self.inverse_domains_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.domains_vocabulary,
            num_oov_indices=1,
            invert=True,
            name="inverse_domains_lookup",
        )
        self.nhosts = self.hosts_lookup.vocabulary_size()
        self.ndomains = self.domains_lookup.vocabulary_size()

        # Tokens Embeddings
        self.host_embeddings = tf.keras.layers.Embedding(
            input_dim=self.nhosts,
            output_dim=self.conf["dim"],
            input_length=self.conf["seqlen"],
            name="host_embeddings",
        )
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["dim"],
            input_length=self.conf["seqlen"],
            name="domain_embeddings",
        )

        # Batch normalization
        self.bn = tf.keras.layers.BatchNormalization()

        # MHGAT blocks
        self.blocks = [
            MHGAT_Block(
                heads=self.conf["heads"],
                emb_dim=self.conf["dim"]
                * (
                    1 + self.conf.get("concat_hosts")
                ),  # if --concat-hosts, the size of internal layers is doubled
                block_id=b,
                tensorboard=self.conf["tensorboard"],
                tb_manager=self.tb_manager,
                name=f"MHGAT_Block_{b}",
            )
            for b in range(self.conf["blocks"])
        ]

        # MLM softmax classifier
        self.masked_classifier = pytftk.nn.FF(
            [self.conf["dim"], self.ndomains],
            [None, "softmax"],
            name="softmax_layer",
        )

        # downstream task softmax classifier
        self.downstream_classifier = None

    @tf.function
    def slice_hosts(self, t):
        """t must be a tf.Tensor of shape [Batch, Seqlen, 2]"""
        return tf.slice(
            t,
            [0, 0, 0],
            tf.concat((tf.gather(tf.shape(t), [0, 1]), [1]), axis=0),
        )

    @tf.function
    def slice_domains(self, t):
        """t must be a tf.Tensor of shape [Batch, Seqlen, 2]"""
        return tf.slice(
            t,
            [0, 0, 1],
            tf.concat((tf.gather(tf.shape(t), [0, 1]), [1]), axis=0),
        )

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
            tf.math.equal(domains, b"<START>"), tf.ones_like(rnd), rnd
        )  # <START> is never masked
        rnd = tf.where(
            tf.math.equal(domains, b"<PAD>"), tf.ones_like(rnd), rnd
        )  # <PAD> is never masked
        domain_indexes = self.domains_lookup(domains)
        rnd = tf.where(
            tf.math.equal(domain_indexes, 0), tf.ones_like(rnd), rnd
        )  # [UNK] (tf assigns index 0 to it) is never masked
        # TODO [UNK] embedding gets updated according to tensorboard, which shouldn't happen. are you sure it gets mapped to 0? check bug
        # i think the bug is that now i'm replacing the masked tokens with "<MASK>", but it can't find "<MASK>" in the vocabulary,
        # because in reality the vocabulary contains b"<MASK>", and so it is replaced by b"[UNK]".
        # this would explain why [UNK] embedding is updated (even though it shouldn't be)
        # it is also reflected in --demo mode because the masked tokens actually show b"[UNK]"
        # Fix: i'm prepending b to all occurrences of direct strings
        # Update: after prepending b to all direct strings, and confirming that <MASK> is recognized,
        # [UNK] is still updated, as well as <PAD>, and they shouldn't be according to the current code

        # Create masks of <MASK>, unchanged and random tokens
        all_mask_tokens = tf.fill(dims=tf.shape(domains), value=b"<MASK>")  # [B,L]
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

        # now i'm not masking the host. return masked_hosts instead of hosts if you want to mask them
        return hosts, masked_domains, mask

    def pretrain(self):
        self.domain_embeddings.trainable = True
        self.host_embeddings.trainable = True

        self.masked_classifier.trainable = True
        if self.downstream_classifier is not None:
            self.downstream_classifier.trainable = False
        self.finetuning = False

    def finetune(self):
        if self.conf.get("freeze") is None:
            raise ValueError(
                "When finetuning the model, the freeze attribute must be set. Now it is None."
            )

        if self.conf.get("freeze"):
            self.domain_embeddings.trainable = False
            self.host_embeddings.trainable = False
            print("Freezing layers.")

        self.masked_classifier.trainable = False
        self.downstream_classifier.trainable = True
        self.finetuning = True

    def init_downstream_classifier(self):
        if self.downstream_classifier is None:
            nclasses = {
                "m": Constants._NCLASSES_MALICIOUS_CLASSIFICATION,
                "b": Constants._NCLASSES_BOTNET_DETECTION,
            }[self.conf.get("labeling")]

            self.downstream_classifier = pytftk.nn.FF(
                [self.conf["dim"], nclasses],
                [None, "softmax"],
                name="classification_layer",
            )

    # Deprecating
    # @staticmethod
    # @tfmpl.figure_tensor
    # def draw_scatter(a, verbose=False, **kwargs):
    #     """Draw scatter plots for tf.summary.
    #     Kwargs are passed to the matplotlib Figure initialization.

    #     Args:
    #         a (tf.Tensor): 1D tensor array containing y values of the scatter plot.
    #         verbose (bool, optional): If True, print debugging information. Defaults to False.

    #     Returns:
    #         tf.Tensor: image tensor of shape (1, h, w, 3) of type uint8
    #         (values ranging between 0 and 255), plottable with tf.summary.image().
    #     """
    #     fig = tfmpl.create_figure(figsize=(5, 5), **kwargs)
    #     ax = fig.add_subplot()

    #     # Axes are constrained between 0 and 1 because values are normalized
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)

    #     scatter_array = np.array([[idx, val] for idx, val in enumerate(a)])

    #     # normalize indexes between 0 and 1
    #     scatter_array[:, 0] /= len(a)

    #     # normalize embedding  values between 0 and 1
    #     scatter_array[:, 1] = (scatter_array[:, 1] - np.min(scatter_array[:, 1])) / (
    #         np.max(scatter_array[:, 1]) - np.min(scatter_array[:, 1])
    #     )

    #     if verbose:
    #         print(scatter_array)

    #     # draw plot
    #     ax.scatter(
    #         scatter_array[:, 0],
    #         scatter_array[:, 1],
    #         s=100,
    #         marker="s",
    #     )
    #     fig.tight_layout()
    #     return fig

    def call(self, inputs, training=None, **kwargs):
        # Take host and domain tokens from the given sequence
        hosts = self.slice_hosts(inputs)  # [B,L,1]
        domains = self.slice_domains(inputs)  # [B,L,1]
        hosts = tf.squeeze(hosts, axis=-1)  # [B,L]
        domains = tf.squeeze(domains, axis=-1)  # [B,L]

        # TODO All this should be redone using purely summary.image(), without matplotlib
        # <----------------------- DEBUG: monitor some embeddings on tensorboard
        # if self.tb_manager.is_hot():
        #     # Retrieve embeddings
        #     pad_emb = self.domain_embeddings(self.domains_lookup(tf.constant(b"<PAD>")))
        #     unk_emb = self.domain_embeddings(
        #         self.domains_lookup(
        #             tf.constant(
        #                 b"somedomainnamethatdefinitelydoesnotappearinthevocabulary...wellihopesootherwiseeverythingexplodes"
        #             )
        #         )
        #     )  # make sure it doesn't exist
        #     mask_emb = self.domain_embeddings(
        #         self.domains_lookup(tf.constant(b"<MASK>"))
        #     )
        #     most_common_emb = self.domain_embeddings(
        #         self.domains_lookup(tf.constant("edge-mqtt.facebook.com"))
        #     )
        #     similar_but_not_common_emb = self.domain_embeddings(
        #         self.domains_lookup(tf.constant("edge-chat.p.facebook.com"))
        #     )
        #     Compute the scatter plot tensor
        #     pad_emb = self.draw_scatter(tf.identity(pad_emb))
        #     unk_emb = self.draw_scatter(tf.identity(unk_emb))
        #     mask_emb = self.draw_scatter(tf.identity(mask_emb))
        #     most_common_emb = self.draw_scatter(tf.identity(most_common_emb))
        #     similar_but_not_common_emb = self.draw_scatter(
        #         tf.identity(similar_but_not_common_emb)
        #     )
        #     Write the images
        #     self.tb_manager.image("<PAD>", pad_emb)
        #     self.tb_manager.image("[UNK]", unk_emb)
        #     self.tb_manager.image("<MASK>", mask_emb)
        #     self.tb_manager.image(
        #         "edge-mqtt.facebook.com (most common)", most_common_emb
        #     )
        #     self.tb_manager.image(
        #         "edge-chat.p.facebook.com (not common)", similar_but_not_common_emb
        #     )
        # --------------------------------------------------------------------->

        # Mask the tokens if required
        # if a domain becomes b<MASK>, random, or same, it will be a 1 in mask, otherwise a 0. b<PAD> is always 0 (not masked).
        mask_p = tf.constant(self.conf["p_mask"])
        same_p = tf.constant(self.conf["p_same"])
        random_p = tf.constant(self.conf["p_random"])
        hosts, domains, mask = self.mask(
            hosts,
            domains,
            mask_p,
            same_p,
            random_p,
            prevent_masking=(not training or self.finetuning)
            and not kwargs.get(
                "force_masking"
            ),  # test_step() is training=False, but it still needs masking
        )

        # NOTE moved from above
        # Lookup vocab index for each token
        domain_indexes = self.domains_lookup(domains)
        host_indexes = self.hosts_lookup(hosts)

        # set adjacency to 1 everywhere except for <PAD>, for which it's set to 0
        pad_adj = tf.einsum(
            "bi,bj->bij",
            tf.cast(tf.not_equal(domains, b"<PAD>"), tf.float32),
            tf.cast(tf.not_equal(domains, b"<PAD>"), tf.float32),
        )
        # add self-loops to the <PAD> tokens
        # NOTE without self-loops (all 0s), the output of the softmax will
        # be 1/seqlen for each <PAD> (because the sum must equal to 1),
        # which means that each <PAD> will take values from the other <PAD>s,
        # and these values can be different because of the dropout.
        # with self-loops on the <PAD>s, they are truly disconnected
        I = tf.cast(tf.eye(tf.shape(domains)[-1]), tf.bool)
        pad_adj = tf.cast(
            tf.math.logical_or(tf.cast(pad_adj, tf.bool), I), tf.int32
        )  # [B,L,L]

        # compute graph topologies
        adjs = []
        for adj_estimator in self.adj_estimators:
            # intersect pad_adj with graph topologies
            adjs.append(tf.math.multiply(pad_adj, adj_estimator(domains)))
        adjs = tf.stack(adjs, axis=1)  # [B,n_adjs,L,L]

        # Retrieve embedding for each token index
        e_d = self.domain_embeddings(domain_indexes)
        e_h = self.host_embeddings(host_indexes)

        # DEBUG: completely zeroing out <PAD> embeddings
        # NOTE this is a workaround to prevent <PAD> embedding from updating (and it works),
        # but the cause of the update is still unknown
        e_d = tf.where(
            tf.tile(
                tf.expand_dims(tf.not_equal(domains, b"<PAD>"), -1),
                [1, 1, tf.shape(e_d)[-1]],
            ),
            e_d,
            tf.zeros_like(e_d),
        )

        # combine domain and host embeddings, either by (weighted) sum or by concatenation
        if self.conf.get("concat_hosts"):
            emb = tf.concat([e_d, e_h], axis=-1)
        else:
            emb = self.conf["omega"] * e_d + (1 - self.conf["omega"]) * e_h

        # dropout the embeddings before feeding to MHGAT blocks
        emb = tf.nn.dropout(emb, 0.15)

        # NOTE NEW; check if it's better
        # batch normalize the embeddings before feeding to MHGAT blocks
        emb = self.bn(emb)

        # forward embeddings through MHGAT blocks
        for block in self.blocks:
            if self.finetuning and self.conf.get("freeze"):
                emb = tf.stop_gradient(
                    block(emb, adjs)
                )  # NOTE workaround for the (--load, --gpu all) finetuned bug
            else:
                emb = block(emb, adjs)

        # Force initializiation of weights for both layers by calling them both even if not needed;
        # this prevents problems when loading weights
        # NOTE this is memory inefficient, check if the problem can be fixed in another way
        if not self.initialized:
            res = self.masked_classifier(emb)
            if self.downstream_classifier is not None:
                res = self.downstream_classifier(emb)
            self.initialized = True

        if not self.finetuning:
            res = self.masked_classifier(emb)
        else:
            res = tf.nn.dropout(emb, 0.2)
            res = self.downstream_classifier(res)

        return res, mask

    @tf.function
    def distributed_train_step(self, seq):
        loss = self.dist_strategy.run(self.train_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    @tf.function
    def distributed_test_step(self, seq):
        loss = self.dist_strategy.run(self.test_step, args=(seq,))
        return self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    def train_step(self, seq):
        if self.finetuning:
            seq, y = seq[..., :-1], tf.strings.to_number(seq[..., -1])

        domains = tf.squeeze(self.slice_domains(seq), axis=-1)  # [B,L]
        domain_indexes = self.domains_lookup(domains)  # [B,L]

        with tf.GradientTape() as tape:
            pred, mask = self(
                seq, training=True
            )  # ([B,L,vsize], [B,L]) or ([B,L,1], _)

            if self.tb_manager.enabled:  ###############
                self.tb_manager.image("final_output", pred[:1], minmax=True)
                self.tb_manager.text("input_domains", domains[0])

            if not self.finetuning:
                loss = self.compiled_loss(
                    tf.boolean_mask(domain_indexes, mask),
                    tf.boolean_mask(pred, mask),
                    regularization_losses=self.losses,
                )

            else:
                if self.conf.get("test_fold") is not None:
                    in_fold = tf.math.reduce_any(
                        tf.equal(
                            tf.expand_dims(seq[:, :, 1], axis=-1),
                            self.test_fold,
                        ),
                        axis=-1,
                    )

                class_weights = self.compute_class_weights(tf.boolean_mask(y, ~in_fold))
                loss = self.compiled_loss(
                    tf.boolean_mask(y, ~in_fold),
                    tf.boolean_mask(pred, ~in_fold),
                    # regularization_losses=self.losses,
                    sample_weight=class_weights,
                )

                # NOTE BUGFIXING: loss becomes nan
                # The problem might be that if a batch contains
                # ONLY domains that are in test_fold, then i'm computing
                # self.compiled_loss([], [])
                if np.isnan(loss):
                    loss = tf.constant(0.0)

            if self.distributed:
                # TODO check that the distributed loss is calculated correctly,
                # especially considering that tf.size(loss) is different every time and a bit random
                loss = tf.math.divide(
                    tf.math.reduce_mean(loss),
                    self.dist_strategy.num_replicas_in_sync,
                )

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.tb_manager.enabled:
            self.tb_manager.scalar("train_loss", loss)

        if self.tb_manager.enabled:
            for l in self.layers:
                for i, w in enumerate(l.get_weights()):
                    self.tb_manager.histogram(f"{l.name}/{i}", w)

        # TensorBoard -- Increment step
        self.tb_manager.step()

        return loss

    @tf.function
    def compute_class_weights(self, y):
        """Compute class weights to be used as input for the `sample_weight`
        parameter to tf.keras.losses.Loss, to normalize the loss of each
        instance by the inverse frequency of its labels.

        Args:
            y (tf.Tensor): a Tensor of labels.

        Returns:
            tf.Tensor: a tf.Tensor with the same shape as `y`, but with each element
        replaced by its inverse frequency.
        """
        equals = tf.equal(tf.expand_dims(y, axis=-1), y)
        counts = tf.math.reduce_sum(tf.cast(equals, tf.int32), axis=-1)
        return 1 / counts

    def test_step(self, seq):
        if self.finetuning:
            seq, y = seq[..., :-1], tf.strings.to_number(seq[..., -1])

        domains = tf.squeeze(self.slice_domains(seq), axis=-1)
        domain_indexes = self.domains_lookup(domains)

        pred, mask = self(
            seq, training=False, force_masking=True
        )  # force masking because we want to see the real loss here

        if not self.finetuning:
            loss = self.compiled_loss(
                tf.boolean_mask(domain_indexes, mask),
                tf.boolean_mask(pred, mask),
                regularization_losses=self.losses,
            )
        else:
            if self.conf.get("test_fold") is not None:
                in_fold = tf.math.reduce_any(
                    tf.equal(
                        tf.expand_dims(seq[:, :, 1], axis=-1),
                        self.test_fold,
                    ),
                    axis=-1,
                )

            class_weights = self.compute_class_weights(tf.boolean_mask(y, in_fold))
            loss = self.compiled_loss(
                tf.boolean_mask(y, in_fold),
                tf.boolean_mask(pred, in_fold),
                sample_weight=class_weights,
            )

            # NOTE BUGFIXING: loss becomes nan
            # The problem might be that if a batch contains
            # ONLY domains that are in test_fold, then i'm computing
            # self.compiled_loss([], [])
            if np.isnan(loss):
                loss = tf.constant(0.0)

        if self.distributed:
            # TODO check that the distributed loss is calculated correctly,
            # especially considering that tf.size(loss) is different every time and a bit random
            loss = tf.math.divide(
                tf.math.reduce_mean(loss),
                self.dist_strategy.num_replicas_in_sync,
            )

        if self.tb_manager.enabled:
            self.tb_manager.scalar("val_loss", loss)

        return loss

    def _predict(self, seq, mask=None):
        # self.tb_manager.force(True)
        in_fold_mask = None
        if self.finetuning:
            # seq, y = seq[..., :-1], tf.strings.to_number(seq[..., -1])
            seq = seq[..., :-1]
            pred, _ = self(seq, training=False)
            # pred = tf.squeeze(pred, axis=-1)

            in_fold_mask = tf.math.reduce_any(
                tf.equal(
                    tf.expand_dims(seq[:, :, 1], axis=-1),
                    self.test_fold,
                ),
                axis=-1,
            )

        else:  # not finetuning
            mask = tf.constant(mask, dtype=tf.bool)
            domains_mask = tf.squeeze(self.slice_domains(mask), axis=-1)

            masked_seq = tf.where(mask, tf.fill(tf.shape(seq), b"<MASK>"), seq)
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
        # self.tb_manager.force(False)
        return pred, 0.0, in_fold_mask


class MHGAT_Block(tf.keras.layers.Layer):
    def __init__(self, heads, emb_dim, **kwargs):
        super().__init__()

        # TensorBoard Init
        self.tensorboard = kwargs.get("tensorboard", False)
        self.tb_manager = kwargs.get("tb_manager", None)
        self.block_id = kwargs.get("block_id", 0)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Configuration
        self.heads = tf.constant(heads)
        self.emb_dim = tf.constant(emb_dim)
        self.head_dim = tf.math.floordiv(self.emb_dim, self.heads)
        self.nonlinear_stretch = tf.constant(4)
        self.verbose = kwargs.get("verbose")

        # Query, Key and Value matrices (multi-head)
        self.Wq = [
            tf.keras.layers.Dense(
                self.head_dim,
                name=f"MHGAT{self.block_id}-Wq/h{i}",
                # activity_regularizer=tf.keras.regularizers.L2(),
            )
            for i in range(self.heads)
        ]
        self.Wk = [
            tf.keras.layers.Dense(
                self.head_dim,
                name=f"MHGAT{self.block_id}-Wk/h{i}",
                # activity_regularizer=tf.keras.regularizers.L2(),
            )
            for i in range(self.heads)
        ]
        self.Wv = [
            tf.keras.layers.Dense(
                self.head_dim,
                name=f"MHGAT{self.block_id}-Wv/h{i}",
                # activity_regularizer=tf.keras.regularizers.L2(),
            )
            for i in range(self.heads)
        ]
        self.Wo = tf.keras.layers.Dense(self.emb_dim, name=f"MHGAT{self.block_id}-Wo")

        # Softmax
        self.softmax = tf.keras.layers.Softmax()

        # Batch Normalization
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Linear-Nonlinear-Linear sandwich
        self.linear1 = tf.keras.layers.Dense(
            self.emb_dim
        )  # dall'eq. (2) di Vaswani sembra che questo linear1 non ci sia
        self.nonlinear = tf.keras.layers.Dense(
            self.emb_dim * self.nonlinear_stretch,
            activation=tf.keras.activations.gelu,
        )
        self.linear2 = tf.keras.layers.Dense(self.emb_dim)

        # Batch Normalization
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Residual Dropout
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, adjs, **kwargs):
        if self.verbose and self.tb_manager.is_hot():
            print(f"===== Block {self.block_id} =====")

        # inputs (embeddings) [B, L, emb_dim]
        Q = tf.stack(
            [Wqi(inputs) for Wqi in self.Wq], axis=1
        )  # [B, heads, L, head_dim]
        K = tf.stack(
            [Wki(inputs) for Wki in self.Wk], axis=1
        )  # [B, heads, L, head_dim]
        V = tf.stack(
            [Wvi(inputs) for Wvi in self.Wv], axis=1
        )  # [B, heads, L, head_dim]

        if self.tb_manager.enabled:
            self.tb_manager.histogram(
                f"Q-head0-block{self.block_id}", Q[:, 0]
            )  # [B, L, head_dim]
            self.tb_manager.histogram(
                f"K-head0-block{self.block_id}", K[:, 0]
            )  # [B, L, head_dim]
            self.tb_manager.histogram(
                f"V-head0-block{self.block_id}", V[:, 0]
            )  # [B, L, head_dim]

        if self.tb_manager.is_hot():
            if self.verbose:
                print("Embeddings")
                print(inputs[0])
                # print("Avg embedding abs value")
                # print(tf.math.reduce_mean(tf.math.abs(inputs[0]), axis=-1))
            self.tb_manager.image(
                f"block{self.block_id}-input_embs", inputs[:1], minmax=True
            )

            # print("Q (head 0)")
            # print("Avg Q abs value")
            # print(Q[0, 0])
            # print(tf.math.reduce_mean(tf.math.abs(Q[0, 0]), axis=-1))

            # print("Wq0")
            # Wq_kernel, Wq_bias = self.Wq[0].get_weights()
            # print(Wq_kernel)
            # print(Wq_bias)
            # print(tf.math.reduce_mean(tf.math.abs(Wq_kernel)))

            # print("K (head 0)")
            # print(K[0, 0])
            # print("Avg K abs value")
            # print(tf.math.reduce_mean(tf.math.abs(K[0, 0]), axis=-1))

            # print("Wk0")
            # Wk_kernel, Wk_bias = self.Wk[0].get_weights()
            # print(Wk_kernel)
            # print(Wk_bias)
            # print(tf.math.reduce_mean(tf.math.abs(Wk_kernel)))

            # print("V (head 0)")
            # print(V[0, 0])
            # print("Avg V abs value")
            # print(tf.math.reduce_mean(tf.math.abs(V[0, 0]), axis=-1))

            # print("Wv0")
            # Wv_kernel, Wv_bias = self.Wv[0].get_weights()
            # print(Wv_kernel)
            # print(Wv_bias)
            # print(tf.math.reduce_mean(tf.math.abs(Wv_kernel)))

        scores = tf.linalg.matmul(Q, tf.transpose(K, (0, 1, 3, 2)))  # [B, heads, L, L]

        # if self.tb_manager.is_hot():
        #     print("Scores before normalization")
        #     print(scores[0, 0])
        if self.tb_manager.is_hot():
            self.tb_manager.image(
                f"MHGAT{self.block_id}/head0-scores-before-normalization",
                scores[:, 0],
                minmax=True,
            )

        # normalize scores
        # >> Old normalization (Vaswani-style)
        scores = tf.math.divide(
            scores, tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        )  # [B, heads, L, L]
        # >> New normalizazion
        # scores = utils.nn.minmax(scores)

        # if self.tb_manager.is_hot():
        #     print("Scores before softmax (after normalization)")
        #     print(scores[0, 0])
        if self.tb_manager.is_hot():
            self.tb_manager.image(
                f"MHGAT{self.block_id}/head0-scores-before-softmax",
                scores[:, 0],
                minmax=True,
            )

        # <--- Inject adjacency masks here (Vaswani says it's done after normalization)
        # TODO this row forces adj to be [B,L,L]. Expand this code to make it work with
        # multiple adjs, ie. tensor [B,n_adjs,L,L]
        adj = tf.gather(adjs, 0, axis=1)
        # [B, L, L] -> [B, heads, L, L]
        adj = tf.expand_dims(adj, axis=1)
        adj = tf.tile(adj, [1, tf.shape(scores)[1], 1, 1])

        if self.verbose and self.tb_manager.is_hot() and self.block_id == 0:
            print("Adj")
            print(adj[0, 0])
        if self.tb_manager.is_hot():
            self.tb_manager.image(
                f"MHGAT{self.block_id}/adj",
                tf.cast(adj[:, 0], tf.float64),
            )  # adj is the same for all heads
        # --->

        # Calculate softmax masking disconnected scores
        scores = self.softmax(scores, mask=adj)  # [B, heads, L, L] attention weights

        if self.verbose and self.tb_manager.is_hot():
            print("Scores after softmax")
            print(scores[0, 0])
        if self.tb_manager.is_hot():
            self.tb_manager.image(
                f"MHGAT{self.block_id}/head0-softmax-scores",
                scores[:, 0],
                minmax=True,
            )

        # Calculate weighted values
        result = tf.linalg.matmul(scores, V)  # [B, heads, L, head_dim]

        if self.verbose and self.tb_manager.is_hot():
            print("After reimpasto")
            print(result)

        result = tf.concat(
            tf.unstack(result, axis=1), axis=-1
        )  # [B, L, heads*head_dim]

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

        if self.verbose and self.tb_manager.is_hot():
            print(f"Block{self.block_id} output embs:")
            print(result)

        # Tensorboard -- Write activation
        if self.tb_manager.is_hot():

            # Visualize the same thing but with heatmap
            self.tb_manager.image(
                f"{self.block_id}-activation-heatmap",
                result,
                minmax=True,
            )

        return result
