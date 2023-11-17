import os
import datetime
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
from lib.tf_matplotlib import tfmpl

from utils.nn import FF
from utils.distribute import DummyStrategy


class DELM(tf.keras.Model):
    def get_config(self):
        return self.conf

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
        self.binary_classifier.trainable = False
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
        self.binary_classifier.trainable = True
        self.finetuning = True

    @staticmethod
    @tfmpl.figure_tensor
    def draw_scatter(a, verbose=False, **kwargs):
        """Draw scatter plots for tf.summary.
        Kwargs are passed to the matplotlib Figure initialization.

        Args:
            a (tf.Tensor): 1D tensor array containing y values of the scatter plot.
            verbose (bool, optional): If True, print debugging information. Defaults to False.

        Returns:
            tf.Tensor: image tensor of shape (1, h, w, 3) of type uint8
            (values ranging between 0 and 255), plottable with tf.summary.image().
        """
        fig = tfmpl.create_figure(figsize=(5, 5), **kwargs)
        ax = fig.add_subplot()

        # Axes are constrained between 0 and 1 because values are normalized
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        scatter_array = np.array([[idx, val] for idx, val in enumerate(a)])

        # normalize indexes between 0 and 1
        scatter_array[:, 0] /= len(a)

        # normalize embedding  values between 0 and 1
        scatter_array[:, 1] = (scatter_array[:, 1] - np.min(scatter_array[:, 1])) / (
            np.max(scatter_array[:, 1]) - np.min(scatter_array[:, 1])
        )

        if verbose:
            print(scatter_array)

        # draw plot
        ax.scatter(
            scatter_array[:, 0],
            scatter_array[:, 1],
            s=100,
            marker="s",
        )
        fig.tight_layout()
        return fig

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
        TB_FOLDER = f"../tensorboard/{conf.get('model')}"
        self.tb_path = None
        if not os.path.exists(TB_FOLDER):
            os.makedirs(TB_FOLDER)
        if not self.conf["quick_tb"]:
            self.tb_path = os.path.join(
                TB_FOLDER,
                self.conf.get("run_name")
                or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        else:
            # set tensorboard path folder to tmp
            self.tb_path = os.path.join(TB_FOLDER, "tmp")

            # remove existing files in tmp folder if it exists
            if os.path.exists(self.tb_path):
                for filename in os.listdir(self.tb_path):
                    os.remove(os.path.join(self.tb_path, filename))

        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.tb_writer = (
            tf.summary.create_file_writer(self.tb_path)
            if self.conf["quick_tb"] or self.conf["tensorboard"] or self.conf["verbose"]
            else None
        )

        # Token Adjacency
        # self.adj_estimator = AdjacencyEstimator(
        #     type="binary", normalize=False, tb_writer=self.tb_writer
        # )

        # Load test fold if needed (finetuning)
        if self.conf.get("test_fold") is not None:
            fold = np.load(
                os.path.join(
                    self.conf.get("test_folds_path"),
                    f"partition-{self.conf.get('test_partition')}",
                    f"fold-{self.conf.get('test_fold')}.npy",
                )
            )
            self.test_fold = tf.constant(fold)

        # Load vocabularies
        self.hosts_vocabulary = (
            open(self.conf.get("hosts_vocab_path"), "r").read().split("\n")
        )
        self.domains_vocabulary = (
            open(self.conf.get("domains_vocab_path"), "r").read().split("\n")
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
            name="hosts_embedding",
        )
        self.domain_embeddings = tf.keras.layers.Embedding(
            input_dim=self.ndomains,
            output_dim=self.conf["dim"],
            input_length=self.conf["seqlen"],
            name="domains_embedding",
        )

        # Batch normalization
        self.bn = tf.keras.layers.BatchNormalization()

        # MHGAT blocks
        self.blocks = [
            MHGAT_Block(
                n_heads=self.conf["n_heads"],
                emb_dim=self.conf["dim"]
                * (
                    1 + self.conf.get("concat_hosts")
                ),  # if --concat-hosts, the size of internal layers is doubled
                block_id=b,
                tensorboard=self.conf["tensorboard"],
                tb_writer=self.tb_writer,
                name=f"MHGAT_Block_{b}",
            )
            for b in range(self.conf["blocks"])
        ]

        # MLM softmax classifier
        self.masked_classifier = FF(
            [self.conf["dim"], self.ndomains],
            [None, "softmax"],
            name="softmax_layer",
        )

        # supervised binary classifier
        self.binary_classifier = FF(
            [self.conf["dim"], 1],
            [None, "sigmoid"],
            name="classification_layer",
        )

    def call(self, inputs, training=None, **kwargs):
        # Take host and domain tokens from the given sequence
        hosts = self.slice_hosts(inputs)  # [B,L,1]
        domains = self.slice_domains(inputs)  # [B,L,1]
        hosts = tf.squeeze(hosts, axis=-1)  # [B,L]
        domains = tf.squeeze(domains, axis=-1)  # [B,L]

        # <----------------------- DEBUG: monitor some embeddings on tensorboard
        if self.tb_writer and tf.math.equal(
            tf.math.floormod(self.step, tf.constant(100, dtype=tf.int64)),
            tf.constant(0, dtype=tf.int64),
        ):
            # Retrieve embeddings
            pad_emb = self.domain_embeddings(self.domains_lookup(tf.constant(b"<PAD>")))
            unk_emb = self.domain_embeddings(
                self.domains_lookup(
                    tf.constant(
                        b"somedomainnamethatdefinitelydoesnotappearinthevocabulary...wellihopesootherwiseeverythingexplodes"
                    )
                )
            )  # make sure it doesn't exist
            mask_emb = self.domain_embeddings(
                self.domains_lookup(tf.constant(b"<MASK>"))
            )
            most_common_emb = self.domain_embeddings(
                self.domains_lookup(tf.constant("edge-mqtt.facebook.com"))
            )
            similar_but_not_common_emb = self.domain_embeddings(
                self.domains_lookup(tf.constant("edge-chat.p.facebook.com"))
            )

            # Compute the scatter plot tensor
            pad_emb = self.draw_scatter(tf.identity(pad_emb))
            unk_emb = self.draw_scatter(tf.identity(unk_emb))
            mask_emb = self.draw_scatter(tf.identity(mask_emb))
            most_common_emb = self.draw_scatter(tf.identity(most_common_emb))
            similar_but_not_common_emb = self.draw_scatter(
                tf.identity(similar_but_not_common_emb)
            )

            # Write the images
            with self.tb_writer.as_default():
                tf.summary.image("<PAD>", pad_emb, step=self.step)
                tf.summary.image("[UNK]", unk_emb, step=self.step)
                tf.summary.image("<MASK>", mask_emb, step=self.step)
                tf.summary.image(
                    "edge-mqtt.facebook.com (most common)",
                    most_common_emb,
                    step=self.step,
                )
                tf.summary.image(
                    "edge-chat.p.facebook.com (not common)",
                    similar_but_not_common_emb,
                    step=self.step,
                )
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
        adj = tf.einsum(
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
        adj = tf.cast(tf.math.logical_or(tf.cast(adj, tf.bool), I), tf.int32)

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
                    block(emb, adj, step=self.step)
                )  # NOTE workaround for the (--load, --gpu all) finetuned bug
            else:
                emb = block(emb, adj, step=self.step)

        # Force initializiation of weights for both layers by calling them both even if not needed;
        # this prevents problems when loading weights
        # NOTE this is memory inefficient, check if the problem can be fixed in another way
        if not self.initialized:
            res = self.masked_classifier(emb)
            res = self.binary_classifier(emb)
            self.initialized = True

        if not self.finetuning:
            res = self.masked_classifier(emb)
        else:
            res = tf.nn.dropout(emb, 0.2)
            res = self.binary_classifier(res)

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

            if not self.finetuning:
                loss = self.compiled_loss(
                    tf.boolean_mask(domain_indexes, mask),
                    tf.boolean_mask(pred, mask),
                    regularization_losses=self.losses,
                )

            else:
                pred = tf.squeeze(pred, axis=-1)

                if self.conf.get("test_fold") is not None:
                    in_fold = tf.math.reduce_any(
                        tf.equal(
                            tf.expand_dims(seq[:, :, 1], axis=-1),
                            self.test_fold,
                        ),
                        axis=-1,
                    )

                loss = self.compiled_loss(
                    tf.boolean_mask(y, ~in_fold),
                    tf.boolean_mask(pred, ~in_fold),
                    regularization_losses=self.losses,
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

        if self.conf.get("tensorboard"):
            with self.tb_writer.as_default():
                tf.summary.scalar("train_loss", loss, step=self.step)

        if self.conf.get("tensorboard") and self.step % 10 == 0:
            for l in self.layers:
                # print(l)
                for i, w in enumerate(l.get_weights()):
                    # print(w.shape)
                    with self.tb_writer.as_default():
                        tf.summary.histogram(f"{l.name}/{i}", w, step=self.step)
                # print()

        # TensorBoard -- Increment step
        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        return loss

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
            pred = tf.squeeze(pred, axis=-1)

            if self.conf.get("test_fold") is not None:
                in_fold = tf.math.reduce_any(
                    tf.equal(
                        tf.expand_dims(seq[:, :, 1], axis=-1),
                        self.test_fold,
                    ),
                    axis=-1,
                )

            loss = self.compiled_loss(
                tf.boolean_mask(y, in_fold), tf.boolean_mask(pred, in_fold)
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

        if self.conf.get("tensorboard"):
            with self.tb_writer.as_default():
                tf.summary.scalar("val_loss", loss, step=self.step)

        return loss

    def _predict(self, seq, mask=None):
        in_fold_mask = None
        if self.finetuning:
            # seq, y = seq[..., :-1], tf.strings.to_number(seq[..., -1])
            seq = seq[..., :-1]
            pred, _ = self(seq, training=False)
            pred = tf.squeeze(pred, axis=-1)

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

        return pred, 0.0, in_fold_mask


class MHGAT_Block(tf.keras.layers.Layer):
    @tf.function
    def tb_log_image(self, name, tensor, step, minmax=False):
        # if the channel (color) axis is missing, add it with dim 1 (greyscale)
        tensor = tf.cond(
            tf.math.equal(tf.rank(tensor), tf.constant(3)),
            lambda: tf.expand_dims(tensor, -1),
            lambda: tensor,
        )

        # if minmax, apply minmax normalization to the image
        tensor = tf.cond(
            tf.constant(minmax),
            lambda: self.minmax_norm(tensor),
            lambda: tensor,
        )

        # write image
        with self.tb_writer.as_default():
            tf.summary.image(name=name, data=tensor, step=step)

    @tf.function
    def minmax_norm(self, tensor):
        return tf.divide(
            tf.math.subtract(tensor, tf.math.reduce_min(tensor)),
            tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)),
        )

    def __init__(self, n_heads, emb_dim, **kwargs):
        super().__init__()

        # TensorBoard Init
        self.tensorboard = kwargs.get("tensorboard", False)
        self.tb_writer = kwargs.get("tb_writer", None)
        self.block_id = kwargs.get("block_id", 0)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Configuration
        self._name = kwargs.get("name") or self.name
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

    def call(self, inputs, adj, **kwargs):
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

        # normalize scores
        scores = tf.math.divide(
            scores, tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        )  # [B, n_heads, L, L]

        if self.tb_writer and self.step % 100 == 0:
            self.tb_log_image(
                f"MHGAT{self.block_id}/head0-scores-before-softmax",
                scores[:, 0],
                step=self.step,
                minmax=True,
            )

        # <--- Inject adjacency mask here (Vaswani says it's done after normalization)
        adj  # [B, L, L] -> [B, n_heads, L, L]
        adj = tf.expand_dims(adj, axis=1)
        adj = tf.tile(adj, [1, tf.shape(scores)[1], 1, 1])

        if self.tb_writer and self.step % 100 == 0:
            self.tb_log_image(
                f"MHGAT{self.block_id}/adj",
                tf.cast(adj[:, 0], tf.float64),
                step=self.step,
            )  # adj is the same for all heads
        # --->

        # Calculate softmax masking disconnected scores
        scores = self.softmax(scores, mask=adj)  # [B, n_heads, L, L] attention weights

        if self.tb_writer and self.step % 100 == 0:
            self.tb_log_image(
                f"MHGAT{self.block_id}/head0-softmax-scores",
                scores[:, 0],
                step=self.step,
                minmax=True,
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

        # Tensorboard -- Write activation
        if self.tb_writer and self.step % 100 == 0:
            # Visualize the first word of the first sequence as it evolves through blocks
            with self.tb_writer.as_default():
                result_image = DELM.draw_scatter(
                    tf.identity(result[0, 0])
                )  # [emb_dim] -> [1, h, w, 3]
                tf.summary.image(
                    name=f"{self.block_id}-firstquery",
                    data=result_image,
                    step=self.step,
                )

            # Visualize the same thing but with heatmap
            self.tb_log_image(
                f"{self.block_id}-activation-heatmap",
                result,
                self.step,
                minmax=True,
            )

        # Tensorboard -- increment step
        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        return result


class AdjacencyEstimator(tf.keras.layers.Layer):
    @tf.function
    def duplicate_axis(self, tensor, from_axis, to_axis, order="C"):
        # TODO only works with exactly from_axis=1, to_axis=2 and rank(tensor)==3; generalize
        """tensor: input tensor
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
        super().__init__()

        # TensorBoard
        self.tb_writer = kwargs.get("tb_writer", None)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Adjacency Conf
        self.type = type
        self.threshold = threshold
        self.normalize = normalize

    def call(self, inputs):
        # inputs [B,L]
        hierarchical_similarity = self.hierarchical_similarity(inputs, step=self.step)

        adj = self.construct_adjacency(
            hierarchical_similarity, self.type, self.threshold
        )
        adj = tf.cond(
            tf.math.equal(self.normalize, True),
            lambda: self._normalize(adj),
            lambda: adj,
        )

        # if self.tb_writer:
        #     with self.tb_writer.as_default():
        #         tf.summary.image("adj", tf.expand_dims(adj, axis=-1), step=0)

        self.step.assign_add(tf.constant(1, dtype=tf.int64))

        return adj
