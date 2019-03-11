import os
import collections
import numpy as np


def load_rating(rating_file, min_rating=0):
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    idx = np.where(rating_np[:, 2] > min_rating)[0]

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    return n_user, n_item, rating_np[idx][:, [0, 1]]


def load_kg(kg_file):
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    kg = dict()
    for head, relation, tail in kg_np:
        if head not in kg:
            kg[head] = collections.defaultdict(list)
        kg[head][relation].append(tail)
    return kg


def get_tails(kg, n_relation, items):
    def _get_tails(item):
        tails = []
        for relation in range(n_relation):
            if relation in kg[item]:
                tails.append(np.random.choice(kg[item][relation]))
            else:
                tails.append(-1)
        return tails

    tails = []
    for item in items:
        tails.append(_get_tails(item))
    return np.array(tails)


def sample_kg(kg, n_relation, items):
    relation_tails = collections.defaultdict(set)
    for item in kg.keys():
        for relation in range(n_relation):
            relation_tails[relation].update(kg[item][relation])

    subset = dict()
    def _get_sample_list(item, relation, subset):
        if item not in subset:
            subset[item] = dict()

        if relation not in subset[item]:
            subset[item][relation] = list(relation_tails[relation] - set(kg[item][relation]))
        return subset[item]

    def _sample_kg(item):
        tails = []
        for relation in range(n_relation):
            sample_list = _get_sample_list(item, relation, subset)

            if len(sample_list) == 0:
                tails.append(-1)
            else:
                tails.append(np.random.choice(list(sample_list)))

        return tails

    tails = []
    for item in items:
        tails.append(_sample_kg(item))
    return np.array(tails)

def sample_ratings(ratings, users):
    rating_items = collections.defaultdict(set)
    item_set = set()
    for u, i in ratings:
        rating_items[u].add(i)
        item_set.add(i)

    subset = dict()

    def _get_sample_list(user, subset):
        if user not in subset:
            subset[user] = list(item_set - set(rating_items[user]))
        return subset[user]

    def _sample_ratings(user):
        sample_list = _get_sample_list(user, subset)

        if len(sample_list) == 0:
            return tails.append(-1)
        else:
            return np.random.choice(sample_list)

    tails = []
    for user in users:
        tails.append(_sample_ratings(user))
    return np.array(tails)

