import tensorflow as tf
from tensorflow.python.ops import array_ops
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


def _relation_nce_loss(model, example_idxs, head_entity_name, relation_name, tail_entity_name):
    relation_vec = model.relation_dict[relation_name]['embedding']
    example_emb = model.entity_dict[head_entity_name]['embedding']
    label_idxs = model.relation_dict[relation_name]['idxs']
    label_emb = model.entity_dict[tail_entity_name]['embedding']
    label_bias = model.relation_dict[relation_name]['bias']
    label_size = model.entity_dict[tail_entity_name]['size']
    label_distribution = model.relation_dict[relation_name]['distribute']
    loss, embs = _pair_search_loss(model, relation_vec, example_idxs, example_emb, label_idxs, label_emb,
                                   label_bias, label_size, label_distribution)
    return tf.reduce_sum(model.relation_dict[relation_name]['weight'] * loss), embs


def _pair_search_loss(model, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias,
                      label_size, label_distribution):
    batch_size = array_ops.shape(example_idxs)[0]  # get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs, dtype=tf.int64), [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=model.negative_sample,
        unique=False,
        range_max=label_size,
        distortion=0.75,
        unigrams=label_distribution))

    # get example embeddings [batch_size, embed_size]
    # example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-add_weight) + relation_vec * add_weight
    example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) + relation_vec

    # get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    # get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
    sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])
    sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

    return _nce_loss(true_logits, sampled_logits), [example_vec, true_w, sampled_w]


def _nce_loss(true_logits, sampled_logits):
    "Build the graph for the NCE loss."

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sampled_logits, labels=tf.zeros_like(sampled_logits))
    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (true_xent + tf.reduce_sum(sampled_xent, 1))
    return nce_loss_tensor


class MultiViewEmbedding(Model):
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.init_width = 0.5 / self.embed_size

    def encoder(self, inputs):
        entities, relations = inputs

        entity_dict = dict(
            (key, _entity(key, ids, self.embed_size, self.init_width)) for (key, ids) in entities.items())
        relation_dict = dict(
            (key, _relation(key, ids, self.embed_size, self.init_width)) for (key, ids) in relations.items())

        return entity_dict, relation_dict

