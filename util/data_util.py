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
    return rating_np[idx][:, [0, 1]]


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
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_tails(kg, n_relation, item):
    relations_dict = collections.defaultdict(list)
    for tail, relation in kg[item]:
        relations_dict[relation].append(tail)

    tails = []
    for relation in range(n_relation):
        if relation in relations_dict:
            tails.append(np.random.choice(relations_dict[relation]))
        else:
            tails.append(-1)
    return tails
