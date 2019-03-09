import tensorflow as tf


class BPRMF(object):
    def __init__(self,
                 embedding_size,
                 user_total,
                 item_total):
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total

        self.user_embeddings = tf.get_variable(name="user_embeddings", shape=[self.user_total, self.embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.item_embeddings = tf.get_variable(name="item_embeddings", shape=[self.item_total, self.embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def forward(self, u_ids, i_ids):
        u_e = tf.nn.embedding_lookup(self.user_embeddings, u_ids)
        i_e = tf.nn.embedding_lookup(self.item_embeddings, i_ids)

        score = tf.reduce_sum(tf.multiply(u_e, i_e), 1)
        return score
