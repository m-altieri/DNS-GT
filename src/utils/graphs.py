import tensorflow as tf


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
    def construct_adjacency(self, similarity, kind, threshold):
        return tf.case(
            [
                (
                    tf.math.equal(kind, "binary"),
                    lambda: tf.where(
                        tf.math.less(similarity, threshold),
                        tf.zeros_like(similarity),
                        tf.ones_like(similarity),
                    ),
                ),
                (
                    tf.math.equal(kind, "cutoff"),
                    lambda: tf.where(
                        tf.math.less(similarity, threshold),
                        tf.zeros_like(similarity),
                        similarity,
                    ),
                ),
                (
                    tf.math.equal(kind, "weighted"),
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

    def __init__(self, kind, threshold=0.3, normalize=True, **kwargs):
        """
        Estimate the adjacency of the domains graph.
        kind (string): either 'binary', 'cutoff', or 'weighted'.
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
        self.kind = kind
        self.threshold = threshold
        self.normalize = normalize

    def call(self, inputs):
        # inputs [B,L]
        hierarchical_similarity = self.hierarchical_similarity(inputs, step=self.step)

        adj = self.construct_adjacency(
            hierarchical_similarity, self.kind, self.threshold
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
