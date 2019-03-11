import tensorflow as tf


class MultiViewEmbedding():
    def __init__(self,
                 embedding_size,
                 ent_dict,
                 rel_total,
                 rel_pairs,
                 distributes,
                 negative_sample=5):
        self.embedding_size = embedding_size
        self.rel_total = rel_total
        self.distributes = distributes
        self.n_pairs = len(rel_pairs)
        self.rel_pairs = rel_pairs
        self.negative_sample = negative_sample

        self.ent_embedding_dict = dict()
        self.rel_biases = dict()
        for ent_name, ent_size in ent_dict:
            self.ent_embedding_dict[ent_name] = tf.get_variable(name="ent_embeddings",
                                                                shape=[ent_size, self.embedding_size],
                                                                initializer=tf.contrib.layers.xavier_initializer(
                                                                    uniform=False))

        for r, distribute in enumerate(self.distributes):
            self.rel_biases[r] = tf.get_variable(name="rel_biases", shape=[len(distribute)],
                                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.rel_total, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def loss(self, hs, ts):
        for i in range(self.n_pairs):
            h, t, r = hs[i], ts[i], i
            h_name, t_name = self.rel_pairs[i]
            h_embeddings, t_embeddings = self.ent_embedding_dict[h_name], self.ent_embedding_dict[t_name]
            h_e = tf.nn.embedding_lookup(h_embeddings, h)
            r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)

            # Negative sampling.
            distribute = self.distributes[r]
            t_size = tf.shape(t_embeddings)[0]
            negative_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=t,
                num_true=1,
                num_sampled=self.negative_sample,
                unique=False,
                range_max=t_size,
                distortion=0.75,
                unigrams=distribute))

            positive_w = tf.nn.embedding_lookup(t_embeddings, t)
            positive_b = tf.nn.embedding_lookup(self.rel_biases[r], t)

            negative_w = tf.nn.embedding_lookup(t_embeddings, negative_ids)
            negative_b = tf.nn.embedding_lookup(self.rel_biases[r], negative_ids)

            positive_logits = tf.reduce_sum(tf.multiply(h_e + r_e, positive_w), 1) + positive_b
            negative_logits = tf.reduce_sum(tf.multiply(h_e + r_e, negative_w), 1) + negative_b
            return self._nce_loss(positive_logits, negative_logits)

    def predict(self, h):
        r = 0
        h_name, t_name = self.rel_pairs[r]
        h_embeddings, t_embeddings = self.ent_embedding_dict[h_name], self.ent_embedding_dict[t_name]
        h_e = tf.nn.embedding_lookup(h_embeddings, h)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        bias = self.rel_biases[r]

        return tf.matmul(h_e + r_e, t_embeddings, transpose_b=True) + bias

    def _nce_loss(self, positive_logits, negative_logits):
        positive_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=positive_logits, labels=tf.ones_like(positive_logits))
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=negative_logits, labels=tf.zeros_like(negative_logits))
        nce_loss = (positive_xent + tf.reduce_sum(negative_xent, 1))
        return nce_loss
