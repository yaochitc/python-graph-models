import tensorflow as tf


class TransE():
    def __init__(self,
                 ent_size,
                 rel_size,
                 ent_total,
                 rel_total):
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.ent_total = ent_total
        self.rel_total = rel_total

        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.ent_total, self.ent_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.rel_total, self.rel_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.proj_embeddings = tf.get_variable(name="proj_embeddings",
                                               shape=[self.rel_total, self.ent_size * self.rel_size],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def loss(self, h, t, r):
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        proj_e = tf.nn.embedding_lookup(self.proj_embeddings, r)
