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

        proj_h_e = _projection_transH(h_e, h_proj, r_proj)
        proj_t_e = _projection_transH(t_e, t_proj, r_proj)
        loss = tf.reduce_sum(tf.abs(proj_h_e + r_e - proj_t_e))
        return loss


def _projection_transH(e, proj, r_proj):
    return tf.nn.l2_normalize(e + tf.reduce_sum(e * proj, 1, keep_dims=True) * r_proj, -1)
