import tensorflow as tf
from base import Model


def _entity(name, vocab, embed_size, init_width):
    return {
        'name': name,
        'vocab': vocab,
        'size': len(vocab),
        'embedding': tf.Variable(tf.random_uniform(
            [len(vocab), embed_size], -init_width, init_width),
            name="%s_emb" % name)
    }


def _relation(name, distribute, embed_size, init_width):
    return {
        'distribute': distribute,
        'idxs': tf.placeholder(tf.int64, shape=[None], name="%s_idxs" % name),
        'weight': tf.placeholder(tf.float32, shape=[None], name="%s_weight" % name),
        'embedding': tf.Variable(tf.random_uniform(
            [embed_size], -init_width, init_width),
            name="%s_emb" % name),
        'bias': tf.Variable(tf.zeros([len(distribute)]), name="%s_b" % name)
    }


class MultiViewEmbedding(Model):
    def __init__(self, entities, relations, embed_size):
        self.embed_size = embed_size
        init_width = 0.5 / self.embed_size

        self.entity_dict = dict((key, _entity(key, ids, self.embed_size, init_width)) for (key, ids) in entities.items())
        self.relation_dict = dict((key, _relation(key, ids, self.embed_size, init_width)) for (key, ids) in relations.items())

