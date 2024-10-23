from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from scipy import stats
import networkx as nx
import numpy as np
import math

def check_clusters(data,labels):
    import matplotlib.pyplot as plt
    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=labels)
    plt.show()

def check_samples(data,samples,non_samples):
    import matplotlib.pyplot as plt
    plt.scatter(np.array(data)[samples, 0], np.array(data)[samples, 1], color='r')
    plt.scatter(np.array(data)[non_samples, 0], np.array(data)[non_samples, 1], color='b')
    plt.show()

def resample(data,delta,rng):
    print('Requested points not satisfied...using random sampling to choose remaining points')
    X = np.arange(len(data))
    sampled_data = rng.choice(X, delta, replace=False)
    non_samples = np.delete(X, sampled_data)

    return [data[index] for index in sampled_data],[data[index] for index in non_samples]
def random(data,split,rng,cp=0):
    if cp > 0:
        npoints = cp
        if npoints > len(data):
            npoints = len(data)-1
    else:
        npoints = math.ceil(split * float(len(data)))
    X = np.arange(len(data))
    sampled_data = rng.choice(X, npoints, replace=False)
    non_samples = np.delete(X, sampled_data)
    return sampled_data, non_samples

def kmeans(data,split,clusters,rng):
    points_per_cluster = math.ceil(split*float(len(data))/float(clusters))
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit(data)
    labels = kmeans.labels_
    sampled_data = None
    remaining_data = None
    for cluster in range(clusters):
        idx = np.where(labels == cluster)[0]
        if points_per_cluster < len(idx):
            samples = rng.choice(idx, points_per_cluster, replace=False)
            sort_idx = idx.argsort()
            non_samples = np.delete(idx,sort_idx[np.searchsorted(idx,samples,sorter = sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data,samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data,non_samples)
        else:
            if sampled_data is None:
                sampled_data = idx
            else:
                sampled_data = np.append(sampled_data,idx)
    delta = points_per_cluster * clusters - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def property_binning(data,y,split,clusters,rng):
    binned_idx = []
    for i in range(clusters + 1):
        binned_idx.append([])
    x = np.linspace(min(y),max(y),clusters)
    for id,val in enumerate(y):
        for i,xx in enumerate(x):
            if i == 0:
                if val <= x[0]:
                    binned_idx[0].append(id)
                    break
                elif x[i] < val <= x[i + 1]:
                    binned_idx[i + 1].append(id)
                    break
            elif i > 0 and i < len(x) - 1:
                if x[i] < val <= x[i + 1]:
                    binned_idx[i + 1].append(id)
                    break
            else:
                binned_idx[i + 1].append(id)
    binned_idx = [bx for bx in binned_idx if bx != []]
    points_per_cluster = math.ceil(split * float(len(data)) / float(len(binned_idx)))

    sampled_data = None
    remaining_data = None
    for bx in binned_idx:
        if len(bx) <= points_per_cluster:
            if sampled_data is None:
                sampled_data = bx
        else:
            samples = rng.choice(bx, points_per_cluster, replace=False)
            sort_idx = np.array(bx).argsort()
            non_samples = np.delete(bx, sort_idx[np.searchsorted(bx, samples, sorter=sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
    delta = points_per_cluster*(len(binned_idx)) - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ',len(sampled_data), ' points out of ',points_per_cluster*(len(binned_idx)),' requested points')
    return sampled_data, remaining_data

def gaussian_mixture(data,split,clusters,rng):
    points_per_cluster = math.ceil(split * float(len(data)) / float(clusters))

    labels = GaussianMixture(n_components=clusters).fit_predict(data)

    sampled_data = None
    remaining_data = None
    for cluster in range(clusters):
        idx = np.where(labels == cluster)[0]
        if points_per_cluster < len(idx):
            samples = rng.choice(idx, points_per_cluster, replace=False)
            sort_idx = idx.argsort()
            non_samples = np.delete(idx, sort_idx[np.searchsorted(idx, samples, sorter=sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
        else:
            if sampled_data is None:
                sampled_data = idx
            else:
                sampled_data = np.append(sampled_data, idx)
    delta = points_per_cluster * clusters - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def spectral(data,split,clusters,rng):
    points_per_cluster = math.ceil(split * float(len(data)) / float(clusters))
    labels = SpectralClustering(n_clusters=clusters).fit_predict(data)

    sampled_data = None
    remaining_data = None
    for cluster in range(clusters):
        idx = np.where(labels == cluster)[0]
        if points_per_cluster < len(idx):
            samples = rng.choice(idx, points_per_cluster, replace=False)
            sort_idx = idx.argsort()
            non_samples = np.delete(idx, sort_idx[np.searchsorted(idx, samples, sorter=sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
        else:
            if sampled_data is None:
                sampled_data = idx
            else:
                sampled_data = np.append(sampled_data, idx)
    delta = int(len(data)*split) - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def birch(data,split,clusters,rng):
    points_per_cluster = math.ceil(split * float(len(data)) / float(clusters))
    labels = Birch(threshold=0.01, n_clusters=clusters).fit_predict(data)

    sampled_data = None
    remaining_data = None
    for cluster in range(clusters):
        idx = np.where(labels == cluster)[0]
        if points_per_cluster < len(idx):
            samples = rng.choice(idx, points_per_cluster, replace=False)
            sort_idx = idx.argsort()
            non_samples = np.delete(idx, sort_idx[np.searchsorted(idx, samples, sorter=sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
        else:
            if sampled_data is None:
                sampled_data = idx
            else:
                sampled_data = np.append(sampled_data, idx)

    delta = points_per_cluster * clusters - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def subgraph_clustering(data,split,leaf_size,neighbors,metric,rng):
    from sklearn.neighbors import KDTree
    G = nx.Graph()
    tree = KDTree(data,metric=metric,leaf_size=leaf_size)
    dist, ind = tree.query(data,k=neighbors+1)
    for i, x in enumerate(data):
        G.add_node(int(i))
        for j in range(len(dist[i])):
            if i != ind[i][j]:
                G.add_edge(i,ind[i][j])
    SG = list(nx.connected_components(G))
    points_per_cluster = math.ceil(split * float(len(data)) / float(len(SG)))
    sampled_data = None
    remaining_data = None
    for i,sg in enumerate(SG):
        idg = list(G.subgraph(sg).copy().nodes())
        if len(idg) > points_per_cluster:
            sort_idg = np.array(idg).argsort()
            samples = rng.choice(idg, points_per_cluster, replace=False)
            non_samples = np.delete(idg, sort_idg[np.searchsorted(idg, samples, sorter=sort_idg)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
        else:
            if sampled_data is None:
                sampled_data = idg
            else:
                sampled_data = np.append(sampled_data, idg)
    delta = points_per_cluster * len(SG) - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data,delta,rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * len(SG), ' requested points')
    return sampled_data, remaining_data

def grid(data,split,grids,rng):
    # scales data if negative
    data = np.array(data)
    dimensions = len(data[0])
    for dim in range(dimensions):
        dim_range = max(data[:,dim]) - min(data[:,dim])
        if min(data[:,dim]) - dim_range/grids < 0:
            for i in range(len(data)):
                data[i][dim] += 2*(dim_range/grids)
    # get grid centroids
    mesh = []
    for dim in range(dimensions):
        dim_range = max(data[:,dim]) - min(data[:,dim])
        mesh.append(np.linspace(min(data[:,dim]) - dim_range/grids, max(data[:,dim]) + dim_range/grids, grids+1))
    grid_centroids = []
    for i in range(dimensions):
        grid_centroids.append([])
        for j in range(grids+1):
            if j > 0:
                grid_centroids[-1].append(mesh[i][j-1] + (mesh[i][j] - mesh[i][j-1])/2.0)
    # create coordinate system and bin data
    mesh_coords = np.vstack(np.meshgrid(*grid_centroids)).reshape(dimensions,-1).T
    binned_data = []
    for gp in mesh_coords:
        binned_data.append([])
    for i, d in enumerate(data):
        min_d = 1E30
        idx = -1
        for j,gp in enumerate(mesh_coords):
            dist = math.dist(d,gp)
            if dist < min_d:
                min_d = dist
                idx = j
        binned_data[idx].append(i)
    binned_data = [ele for ele in binned_data if ele != []] # remove empty bins
    #sample data
    points_per_cluster = math.ceil(split * float(len(data)) / float(len(binned_data)))
    sampled_data = None
    remaining_data = None
    for bx in binned_data:
        if len(bx) <= points_per_cluster:
            if sampled_data is None:
                sampled_data = bx
        else:
            samples = rng.choice(bx, points_per_cluster, replace=False)
            sort_idx = np.array(bx).argsort()
            non_samples = np.delete(bx, sort_idx[np.searchsorted(bx, samples, sorter=sort_idx)])
            if sampled_data is None:
                sampled_data = samples
            else:
                sampled_data = np.append(sampled_data, samples)
            if remaining_data is None:
                remaining_data = non_samples
            else:
                remaining_data = np.append(remaining_data, non_samples)
    delta = points_per_cluster*(len(binned_data)) - len(sampled_data)
    if delta > 0:
        new_samples, remaining_data = resample(remaining_data, delta, rng)
        sampled_data = np.append(sampled_data, new_samples)
    print('Found ',len(sampled_data), ' points out of ',points_per_cluster*(len(binned_data)),' requested points')
    return sampled_data, remaining_data

def run_sampling(data,sampling_type,split,rng,params_group,y=None):
    print('Performing',sampling_type,'sampling using a training ratio of',str(split*100.0),'%')
    if sampling_type == 'random':
        sampled_data, remaining_data = random(data=data,split=split,rng=rng)
    elif sampling_type == 'kmeans':
        sampled_data, remaining_data = kmeans(data=data,split=split,clusters=params_group['clusters'],rng=rng)
    elif sampling_type == 'y_bin':
        sampled_data, remaining_data = property_binning(data=data,y=y,split=split,clusters=params_group['clusters'],rng=rng)
    elif sampling_type == 'gaussian_mixture':
        sampled_data, remaining_data = gaussian_mixture(data=data,split=split,clusters=params_group['clusters'],rng=rng)
    elif sampling_type == 'spectral':
        sampled_data, remaining_data = spectral(data=data,split=split,clusters=params_group['clusters'],rng=rng)
    elif sampling_type == 'birch':
        sampled_data, remaining_data = birch(data=data,split=split,clusters=params_group['clusters'],rng=rng)
    elif sampling_type == 'subgraph_clustering':
        sampled_data, remaining_data = subgraph_clustering(data=data,split=split,leaf_size=params_group['leaf_size'],
                                            neighbors=params_group['neighbors'],metric=params_group['metric'],rng=rng)
    elif sampling_type == 'grid':
        sampled_data, remaining_data = grid(data=data,split=split,grids=params_group['grids'],rng=rng)
    return sampled_data, remaining_data







