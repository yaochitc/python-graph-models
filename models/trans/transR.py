import tensorflow as tf


class TransE(object):
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

    def forward(self, h, t, r):
        h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, h), [-1, self.ent_size, 1])
        t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, t), [-1, self.ent_size, 1])
        r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, r), [-1, self.rel_size])
        proj_e = tf.reshape(tf.nn.embedding_lookup(self.proj_embeddings, r), [-1, self.rel_size, self.ent_size])

        proj_h_e = tf.reshape(tf.matmul(proj_e, h_e), [-1, self.rel_size])
        proj_t_e = tf.reshape(tf.matmul(proj_e, t_e), [-1, self.rel_size])
        score = tf.reduce_sum(tf.abs(proj_h_e + r_e - proj_t_e))
        return score
