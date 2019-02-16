import collections
import numpy as np


def load_rating(rating_file):
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
    return rating_np


def load_kg(kg_file):
    kg_np = np.loadtxt(kg_file, dtype=np.int32)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg
