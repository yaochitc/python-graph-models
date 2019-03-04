import tensorflow as tf


class TransD():
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
        self.ent_proj_embeddings = tf.get_variable(name="ent_proj_embeddings",
                                                   shape=[self.ent_total, self.embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_proj_embeddings = tf.get_variable(name="rel_proj_embeddings",
                                                   shape=[self.rel_total, self.embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def loss(self, h, t, r):
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        h_proj = tf.nn.embedding_lookup(self.ent_proj_embeddings, h)
        t_proj = tf.nn.embedding_lookup(self.ent_proj_embeddings, t)
        r_proj = tf.nn.embedding_lookup(self.rel_proj_embeddings, r)