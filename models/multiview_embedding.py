import tensorflow as tf


class MultiViewEmbedding():
    def __init__(self,
                 embedding_size,
                 ent_total,
                 rel_total,
                 rel_used):
        self.embedding_size = embedding_size
        self.rel_total = rel_total
        self.ent_total = ent_total
        self.rel_used = rel_used

        self.ent_embeddings = tf.get_variable(name="ent_embeddings",
                                              shape=[ent_total, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(
                                                  uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.rel_total, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_biases = tf.get_variable(name="rel_biases", shape=[rel_total, ent_total],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def loss(self, hs, ts, ns):
        loss = 0.
        for i in range(self.rel_used):
            h, t, n, r = hs[i], ts[i], ns[i], i
            h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
            r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)

            # Negative sampling.
            positive_w = tf.nn.embedding_lookup(self.ent_embeddings, t)
            positive_b = tf.nn.embedding_lookup(self.rel_biases[r], t)

            negative_w = tf.nn.embedding_lookup(self.ent_embeddings, n)
            negative_b = tf.nn.embedding_lookup(self.rel_biases[r], n)

            positive_logits = tf.reduce_sum(tf.multiply(h_e + r_e, positive_w), 1) + positive_b
            negative_logits = tf.reduce_sum(tf.multiply(h_e + r_e, negative_w), 1) + negative_b
            loss += self._nce_loss(positive_logits, negative_logits)
        return loss

    def _nce_loss(self, positive_logits, negative_logits):
        positive_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=positive_logits, labels=tf.ones_like(positive_logits))
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=negative_logits, labels=tf.zeros_like(negative_logits))
        nce_loss = tf.reduce_sum(positive_xent) + tf.reduce_sum(negative_xent)
        return nce_loss
