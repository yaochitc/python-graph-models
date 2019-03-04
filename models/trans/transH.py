import tensorflow as tf


class TransE():
    def __init__(self,
                 embedding_size,
                 ent_total,
                 rel_total):
        self.embedding_size = embedding_size
        self.ent_total = ent_total
        self.rel_total = rel_total

        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.ent_total, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.rel_total, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.norm_embeddings = tf.get_variable(name="norm_embeddings", shape=[self.rel_total, self.embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def loss(self, h, t, r):
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        norm_e = tf.nn.embedding_lookup(self.norm_embeddings, r)

