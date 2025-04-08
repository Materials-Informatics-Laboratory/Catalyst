from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from scipy import stats
import networkx as nx
import numpy as np
import random
import torch
import math
import sys

__all__ = [
    'nearest_path_node',
    'generate_latent_space_path',
]

def generate_latent_space_path(X,k=7,boundaries=[0,-1],version='1',reduction=1.0,random_neighbors=False):
    XX = X.tolist()
    if boundaries[1] < -1:
        avg = [0.0]*len(XX[0])
        points = XX[boundaries[1]:]
        for point in points:
            for i in range(len(point)):
                avg[i] += point[i]
        for i in range(len(avg)):
            avg[i] /= len(points)
        XX.append(avg)
    if boundaries[0] > 0:
        avg = [0.0]*len(X[0])
        points = XX[:boundaries[0]]
        for point in points:
            for i in range(len(point)):
                avg[i] += point[i]
        for i in range(len(avg)):
            avg[i] /= len(points)
        XX.insert(0,avg)
    X = np.array(XX)
    sink = len(X) - 1
    D = squareform(pdist(X, 'minkowski', p=2.))
    # select the kNN for each datapoint
    check = True
    while check:
        neighbors = np.sort(np.argsort(D, axis=1)[:, 0:k])
        G = nx.Graph()
        for i, x in enumerate(X):
            G.add_node(i)
        for i, row in enumerate(neighbors):
            for column in row:
                if i != column:
                    G.add_edge(i, column)
                    G[i][column]['weight'] = D[i][column]

        current_node = 0
        weighted_path = [current_node]
        checked_nodes = [current_node]
        while current_node != sink:
            tmp_k = k
            while True:
                nn = G.neighbors(current_node)
                chosen_nn = -1
                e_weight = 1E30
                if random_neighbors:
                    tmp_neighs = []
                    xx = []
                    for neigh in nn:
                        if neigh not in checked_nodes:
                            tmp_neighs.append(neigh)
                        xx.append(neigh)
                    if len(tmp_neighs) > 0:
                        end_range = len(tmp_neighs)-1
                        if len(tmp_neighs) == 1:
                            end_range = 0
                        chosen_nn = tmp_neighs[random.randint(0, end_range)]
                        e_weight = G[current_node][chosen_nn]['weight']
                    else:
                        #print(current_node,' ',list(nn),' ',xx,' ',checked_nodes)
                        print('Node has no neighbors...')
                        pass
                else:
                    for neigh in nn:
                        if neigh not in checked_nodes:
                            if nx.has_path(G,neigh,sink):
                                if G[current_node][neigh]['weight'] < e_weight:
                                    e_weight = G[current_node][neigh]['weight']
                                    chosen_nn = neigh
                if chosen_nn > -1:
                    break
                else:
                    print('Failed pathfinding...increasing local k to ' + str(tmp_k) + '...')
                    tmp_k += 1
                    #print(neighbors[current_node], ' ',D[current_node])
                    neighbors = np.sort(np.argsort(D[current_node], axis=0)[0:tmp_k]) # this can be improved later
                    #for i, row in enumerate(neighbors):
                    #print(neighbors,' ',D[current_node],' ',tmp_k)
                    G.add_edge(current_node, neighbors[-1])
                    G[current_node][neighbors[-1]]['weight'] = D[current_node][neighbors[-1]]

            if chosen_nn < 0:
                weighted_path.pop()
                current_node = weighted_path[-1]
            else:
                current_node = chosen_nn
                checked_nodes.append(current_node)
                weighted_path.append(current_node)
        check = False
        #except:
        #    print('Failed pathfinding...increasing k to ' + str(k) + '...')
        #    k += 1

    if version == '1':
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
        x_to_path, x_to_path_dist = nearest_path_node(X, weighted_path)

        data = {
            "path": x_to_path,
            "path_dist": x_to_path_dist,
            "graph": G,
            "weighted_path": X[weighted_path],
            "d": travelled_distances,
            "path_edges": path_edges
        }

        return data
    elif version == '2':
        new_weighted_path = []
        new_weighted_path.append(weighted_path[0])
        take = math.ceil(reduction * len(weighted_path))
        for i in range(len(weighted_path)):
            if i % take == 0:
                new_weighted_path.append(weighted_path[i])
        new_weighted_path.append(weighted_path[-1])
        binned_data = bin_data_by_nearest_node(X,new_weighted_path)
        weighted_path = get_cluster_centroids(X,binned_data)
        path_distances = []
        for i in range(len(weighted_path[1:])):
            A = tuple([weighted_path[i-1], weighted_path[i]])
            D = squareform(pdist(A, 'minkowski', p=2.))
            path_distances.append(D[0][1])
        total_distance = sum(path_distances)
        travelled_distances = [0.0]
        d = 0.0
        for dist in path_distances:
            d += dist
            travelled_distances.append((d / total_distance))

        data = {
            "weighted_path": np.array(weighted_path),
            "d": np.array(travelled_distances),
        }

        return data
    elif version == '3':
        new_weighted_path = []
        take = math.ceil(reduction * len(weighted_path))
        accumulated_path = []
        for i in range(len(weighted_path)):
            accumulated_path.append(weighted_path[i])
            if i % take == 0:
                avg_path = [0.0]*len(X[0])
                for a_path in accumulated_path:
                    for j in range(len(X[a_path])):
                        avg_path[j] += X[a_path][j]
                for j in range(len(avg_path)):
                    avg_path[j] /= len(accumulated_path)
                new_weighted_path.append(avg_path)
                accumulated_path = []
        binned_data = bin_data_by_nearest_node(X,new_weighted_path,version='3')
        weighted_path = get_cluster_centroids(X,binned_data)
        path_distances = []
        for i in range(len(weighted_path[1:])):
            A = tuple([weighted_path[i-1], weighted_path[i]])
            D = squareform(pdist(A, 'minkowski', p=2.))
            path_distances.append(D[0][1])
        total_distance = sum(path_distances)
        travelled_distances = [0.0]
        d = 0.0
        for dist in path_distances:
            d += dist
            travelled_distances.append((d / total_distance))

        data = {
            "weighted_path": np.array(weighted_path),
            "d": np.array(travelled_distances),
        }

        return data

def get_cluster_centroids(X,bins):
    centroids = []
    for bin in bins:
        subset = X[bin]
        if len(subset) > 0:
            centroid = [0.0]*len(subset[0])
            for point in subset:
                for i,p in enumerate(point):
                    centroid[i] += p
            for i in range(len(centroid)):
                centroid[i] /= len(subset)
            centroids.append(centroid)
    return centroids

def bin_data_by_nearest_node(X,nodes,version='1'):
    bins = []
    for n in nodes:
        bins.append([])

    for p,point in enumerate(X):
        d = 1E30
        bin = -1
        for i,node in enumerate(nodes):
            if version != '3':
                A = tuple([point, X[node]])
            else:
                A = tuple([point, node])
            D = squareform(pdist(A, 'minkowski', p=2.))
            r = D[0][1]
            if r < d:
                d = r
                bin = i
        bins[bin].append(p)
    return bins

def nearest_path_node(x,nodes):
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn.fit(x[nodes])
    distances, indices = knn.kneighbors(x)

    return indices.flatten(), distances.flatten()

def find_nearest_nodes(x,nodes,k=1):
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
    knn.fit(nodes)
    distances, indices = knn.kneighbors(x)
    return indices,distances

def assign_gammas(ref_data,new_data,path_data,smearing='sum',k=1,iterations=1,cutoff=1000.0,scale=10.0):
    gammas = path_data['d']

    nodes = []

    if iterations > 1:
        new_ids = []
        new_dists = []
        new_ids, new_dists = find_nearest_nodes(new_data, new_data, k)
    reference = [path_data['weighted_path'][i] for i in range(len(path_data['weighted_path']))]
    ref_ids, ref_dists = find_nearest_nodes(new_data, reference, k)

    iter = 0
    while iter < iterations:
        assigned_gammas = [0.0] * len(new_data)
        for i, point in enumerate(new_data):
            avg_gamma = 0.0
            weight = 0.0
            if iter == 0:
                ids = ref_ids[i]
                dists = ref_dists[i]
            else:
                ids = new_ids[i]
                dists = new_dists[i]
            if smearing != 'sum':
                for j in range(len(ids)):
                    if dists[j] > cutoff:
                        weight = 0.0
                    else:
                        if smearing == 'radial':
                            weight = 0.5 * math.fabs(math.cos((dists[j] * math.pi) / cutoff) + 1)
                        elif smearing == 'tanh':
                            weight = math.tanh(scale/dists[j])
                        elif smearing == 'sigmoid':
                            weight = 1/0 / (1.0 + math.exp(dists[j]))
                    avg_gamma += weight*gammas[ids[j]]
                    if avg_gamma < 0.0:
                        avg_gamma = 0.0
                avg_gamma /= float(len(ids))
            else:
                for j in range(len(ids)):
                    avg_gamma += gammas[ids[0]]
                avg_gamma /= float(len(ids))

            assigned_gammas[i] = avg_gamma
        gammas = assigned_gammas
        iter += 1
    return assigned_gammas

def manual_convolution(data,gammas,k=2,iterations=1,cutoff=10):
    iter = 0
    tree = KDTree(data, leaf_size=2)
    dists, ids = tree.query(data, k=k + 1)
    while iter < iterations:
        new_gammas = [0.0] * len(data)
        for i in range(len(gammas)):
            if isinstance(gammas[i], float):
                gammas[i] = [torch.tensor(gammas[i])]
        for i, d in enumerate(data):
            for j in range(len(ids[i])):
                if dists[i][j] > cutoff:
                    weight = 0.0
                elif dists[i][j] < 0.001:
                    weight = 1.0
                else:
                    weight = 0.5 * (math.cos((dists[i][j] * math.pi) / cutoff) + 1)
                new_gammas[i] += weight * gammas[ids[i][j]][0].item()
                if new_gammas[i] < 0.0:
                    new_gammas[i] = 0.0
            new_gammas[i] /= float(len(ids[i]))
        gammas = new_gammas
        iter += 1
    return new_gammas












