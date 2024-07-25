from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import numpy as np

def check_clusters(data,labels):
    import matplotlib.pyplot as plt
    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=labels)
    plt.show()

def check_samples(data,samples,non_samples):
    import matplotlib.pyplot as plt
    plt.scatter(np.array(data)[samples, 0], np.array(data)[samples, 1], color='r')
    plt.scatter(np.array(data)[non_samples, 0], np.array(data)[non_samples, 1], color='b')
    plt.show()

def random(data,split,rng):
    npoints = int(split * float(len(data)))
    X = np.arange(len(data))
    sampled_data = rng.choice(X, npoints, replace=False)
    non_samples = np.delete(X, sampled_data)
    return sampled_data, non_samples

def kmeans(data,split,clusters,rng):
    points_per_cluster = int(split*float(len(data))/float(clusters))
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
    print('Found ',len(sampled_data), ' points out of ',points_per_cluster*clusters,' requested points')
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
    points_per_cluster = int(split * float(len(data)) / float(len(binned_idx)))

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

    print('Found ',len(sampled_data), ' points out of ',points_per_cluster*(len(binned_idx)),' requested points')
    return sampled_data, remaining_data

def gaussian_mixture(data,split,clusters,rng):
    points_per_cluster = int(split * float(len(data)) / float(clusters))
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
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def spectral(data,split,clusters,rng):
    points_per_cluster = int(split * float(len(data)) / float(clusters))
    labels = SpectralClustering(n_components=clusters).fit_predict(data)

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
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def birch(data,split,clusters,rng):
    points_per_cluster = int(split * float(len(data)) / float(clusters))
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
    print('Found ', len(sampled_data), ' points out of ', points_per_cluster * clusters, ' requested points')
    return sampled_data, remaining_data

def run_sampling(data,sampling_type,split,rng,nclusters=1,y=None):
    print('Performing',sampling_type,'sampling using a training ratio of',str(split*100.0),'%')
    if sampling_type == 'random':
        sampled_data, remaining_data = random(data=data,split=split,rng=rng)
    elif sampling_type == 'kmeans':
        sampled_data, remaining_data = kmeans(data=data,split=split,clusters=nclusters,rng=rng)
    elif sampling_type == 'y_bin':
        sampled_data, remaining_data = property_binning(data=data,y=y,split=split,clusters=nclusters,rng=rng)
    elif sampling_type == 'gaussian_mixture':
        sampled_data, remaining_data = gaussian_mixture(data=data,split=split,clusters=nclusters,rng=rng)
    elif sampling_type == 'spectral':
        sampled_data, remaining_data = spectral(data=data,split=split,clusters=nclusters,rng=rng)
    elif sampling_type == 'birch':
        sampled_data, remaining_data = birch(data=data,split=split,clusters=nclusters,rng=rng)

    return sampled_data, remaining_data







