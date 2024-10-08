from scipy.spatial.distance import squareform, pdist
from scipy import stats
import networkx as nx

from ..utils.structure_properties import *
from ..utils.alignn import *
from ..utils.graph import atoms2graph
from ..utils.graph import AngularGraphPairData
import torch

import numpy as np
import pandas as pd

__all__ = [
    'nearest_path_node',
    'generate_latent_space_path',
]

def generate_latent_space_path(X,k=7):
    D = squareform(pdist(X, 'minkowski', p=2.))

    # select the kNN for each datapoint
    neighbors = np.sort(np.argsort(D, axis=1)[:, 0:k])

    G = nx.Graph()
    for i, x in enumerate(X):
        G.add_node(i)
    for i, row in enumerate(neighbors):
        for column in row:
            if i != column:
                G.add_edge(i, column)
                G[i][column]['weight'] = D[i][column]

    sink = len(X) - 1
    current_node = 0
    weighted_path = [current_node]
    checked_nodes = [current_node]
    while current_node != sink:
        nn = G.neighbors(current_node)

        chosen_nn = -1
        e_weight = 1E30
        for neigh in nn:
            if neigh not in checked_nodes:
                if nx.has_path(G,neigh,sink):
                    if G[current_node][neigh]['weight'] < e_weight:
                        e_weight = G[current_node][neigh]['weight']
                        chosen_nn = neigh
        if chosen_nn < 0:
            weighted_path.pop()
            current_node = weighted_path[-1]
        else:
            current_node = chosen_nn
            checked_nodes.append(current_node)
            weighted_path.append(current_node)

    path_edges = list(zip(weighted_path, weighted_path[1:]))

    path_distances = []
    for edge in path_edges:
        path_distances.append(G[edge[0]][edge[1]]['weight'])
    total_distance = sum(path_distances)
    travelled_distances = [0.0]
    d = 0.0
    for dist in path_distances:
        d += dist
        travelled_distances.append((d / total_distance))
    x_to_path, x_to_path_dist = nearest_path_node(X, weighted_path, travelled_distances)

    data = {
        "path": x_to_path,
        "path_dist": x_to_path_dist,
        "graph": G,
        "weighted_path": weighted_path,
        "d": travelled_distances,
        "path_edges": path_edges
    }

    return data

def nearest_path_node(x,nodes,distances):
    nn = []
    nd = []
    for val in x:
        nn.append(-1)
        nd.append(-1)
        d = 1E30
        for node in nodes:
            A = tuple([val,x[node]])
            D = squareform(pdist(A, 'minkowski', p=2.))
            r = D[0][1]
            if r < d:
                d = r
                nn[-1] = node
                nd[-1] = distances[nodes.index(node)]
    return nn, nd












