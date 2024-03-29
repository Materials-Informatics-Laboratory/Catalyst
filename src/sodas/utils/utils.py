from scipy.spatial.distance import squareform, pdist
from scipy import stats
import networkx as nx

from sodas.utils.structure_properties import *
from sodas.utils.alignn import *
from sodas.utils.graph import atoms2graph
from sodas.utils.graph import AngularGraphPairData
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

def generate_graph(atoms,cutoff=3.0,dihedral=False):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
        The angular graph holds bond angle information, but can also calculate dihedral information upon request.
        """

    elements = np.unique(atoms.get_chemical_symbols())
    ohe = []
    for atom in atoms:
        tx = [0.0] * len(elements)
        for i in range(len(elements)):
            if atom.number == elements[i]:
                tx[i] = 1.0
                break
        ohe.append(tx)
    x_atm = np.array(ohe)

    edge_index_G, x_bnd = atoms2graph(atoms, cutoff=cutoff)
    edge_index_bnd_ang = line_graph(edge_index_G)
    x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang)

    if dihedral:
        edge_index_dih_ang = dihedral_graph(edge_index_G)
        edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
        x_dih_ang = get_dih_angs(atoms, edge_index_G, edge_index_dih_ang)
        x_ang = np.concatenate([x_bnd_ang, x_dih_ang])
        mask_dih_ang = [False] * len(x_bnd_ang) + [True] * len(x_dih_ang)

        data = MolData(
            edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
            edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
            x_atm=torch.tensor(x_atm, dtype=torch.float),
            x_bnd=torch.tensor(x_bnd, dtype=torch.float),
            x_ang=torch.tensor(x_ang, dtype=torch.float),
            mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool)
        )
    else:
        edge_index_A = np.hstack([edge_index_bnd_ang])
        x_ang = np.concatenate([x_bnd_ang])
        mask_dih_ang = [False] * len(x_bnd_ang)

        data = MolData(
            edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
            edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
            x_atm=torch.tensor(x_atm, dtype=torch.float),
            x_bnd=torch.tensor(x_bnd, dtype=torch.float),
            x_ang=torch.tensor(x_ang, dtype=torch.float),
            mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool)
        )

    return data













