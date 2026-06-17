"""
Optimized latent-space path utilities.

This version keeps the original public API and pathfinding logic, but removes the
full NxN pairwise distance matrix from generate_latent_space_path().  Instead it
uses a cached k-nearest-neighbor table with per-node fallback expansion when the
pathfinder runs out of locally cached neighbors.

""" 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import networkx as nx
import numpy as np
import random
import math

__all__ = [
    'nearest_path_node',
    'generate_latent_space_path',
]


def _as_2d_float_array(X):
    """Convert input coordinates to a 2D float ndarray."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array-like object with shape (n_points, n_features).")
    if len(X) == 0:
        raise ValueError("X must contain at least one point.")
    return X


def _apply_boundaries(X, boundaries):
    """
    Add optional averaged start/end boundary points.

    This preserves the original ordering logic:
    1. If boundaries[1] < -1, append the average of X[boundaries[1]:].
    2. If boundaries[0] > 0, prepend the average of the first boundaries[0] points.
    """
    X = _as_2d_float_array(X)

    if boundaries[1] < -1:
        end_points = X[boundaries[1]:]
        if len(end_points) == 0:
            raise ValueError("boundaries[1] selects no end points to average.")
        X = np.vstack([X, end_points.mean(axis=0)])

    if boundaries[0] > 0:
        start_points = X[:boundaries[0]]
        if len(start_points) == 0:
            raise ValueError("boundaries[0] selects no start points to average.")
        X = np.vstack([start_points.mean(axis=0), X])

    return X


def _build_knn_cache(X, initial_k):
    """
    Build an initial kNN cache.

    The returned neighbor lists include the query point itself when sklearn returns
    it, which matches the original full-distance implementation where argsort(D)
    includes the self-index first.
    """
    n_points = len(X)
    initial_k = max(1, min(int(initial_k), n_points))

    knn_model = NearestNeighbors(n_neighbors=initial_k, algorithm='auto')
    knn_model.fit(X)
    dists, ids = knn_model.kneighbors(X, n_neighbors=initial_k)

    cached_ids = [ids[i].copy() for i in range(n_points)]
    cached_dists = [dists[i].copy() for i in range(n_points)]

    return knn_model, cached_ids, cached_dists


def _ensure_node_has_neighbors(
    X,
    knn_model,
    cached_ids,
    cached_dists,
    node_id,
    required_k,
    expand_by=1,
):
    """
    Ensure node_id has at least required_k cached nearest neighbors.

    If the requested neighbor count exceeds the current cache length for this
    specific node, only this node's neighbor list is re-queried with a larger k.
    """
    n_points = len(X)
    required_k = int(required_k)

    if required_k <= len(cached_ids[node_id]):
        return

    if required_k > n_points:
        raise RuntimeError(
            f"Pathfinding failed at node {node_id}: requested {required_k} neighbors, "
            f"but only {n_points} points exist."
        )

    expand_by = max(1, int(expand_by))
    new_k = min(n_points, max(required_k, len(cached_ids[node_id]) + expand_by))

    dists, ids = knn_model.kneighbors(X[node_id:node_id + 1], n_neighbors=new_k)
    cached_ids[node_id] = ids[0].copy()
    cached_dists[node_id] = dists[0].copy()


def _add_edge_from_cache(G, current_node, neighbor_id, neighbor_dist):
    """Add a weighted edge from the kNN cache, skipping self-edges."""
    if current_node == neighbor_id:
        return False
    G.add_edge(current_node, int(neighbor_id))
    G[current_node][int(neighbor_id)]['weight'] = float(neighbor_dist)
    return True


def _normalized_cumulative_distances_from_edge_lengths(edge_lengths):
    """Return normalized cumulative distances beginning with 0.0."""
    edge_lengths = np.asarray(edge_lengths, dtype=float)
    if len(edge_lengths) == 0:
        return [0.0]

    cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    total = cumulative[-1]
    if total == 0.0:
        return cumulative.tolist()
    return (cumulative / total).tolist()


def _normalized_cumulative_distances_from_coords(path_coords):
    """Return normalized cumulative distances along a coordinate path."""
    path_coords = np.asarray(path_coords, dtype=float)
    if len(path_coords) < 2:
        return np.array([0.0])

    edge_lengths = np.linalg.norm(np.diff(path_coords, axis=0), axis=1)
    return np.asarray(_normalized_cumulative_distances_from_edge_lengths(edge_lengths))


def generate_latent_space_path(
    X,
    k=7,
    boundaries=[0, -1],
    version='1',
    reduction=1.0,
    random_neighbors=False,
    knn_cache_k=None,
    knn_expand_by=1,
):
    """
    Generate a path through latent/descriptor space using a kNN graph.

    Parameters
    ----------
    X : array-like, shape (n_points, n_features)
        Latent-space or descriptor-space coordinates.
    k : int, default=7
        Number of nearest-neighbor entries used to build the initial graph. As in
        the original script, the self-neighbor is included in the kNN table and
        skipped when graph edges are added.
    boundaries : list[int, int], default=[0, -1]
        Optional start/end averaging controls matching the original script.
    version : {'1', '2', '3'}, default='1'
        Output/path coarsening mode from the original script.
    reduction : float, default=1.0
        Coarsening factor used by versions 2 and 3.
    random_neighbors : bool, default=False
        Preserve original random-neighbor traversal option.
    knn_cache_k : int or None, default=None
        Size of the initial cached neighbor table. If None, uses k + 10, capped at
        n_points. The initial graph still uses only the first k cached neighbors;
        the extra cached neighbors are only used when the pathfinder gets stuck.
    knn_expand_by : int, default=1
        Per-node fallback expansion increment once a node runs out of cached
        neighbors.
    """
    X = _apply_boundaries(X, boundaries)
    n_points = len(X)
    sink = n_points - 1

    if k < 1:
        raise ValueError("k must be >= 1.")

    # Cache a larger neighbor table than the initial graph uses.  This keeps the
    # initial graph behavior tied to k while avoiding full NxN distance storage.
    if knn_cache_k is None:
        initial_cache_k = min(n_points, max(k, k + 10))
    else:
        initial_cache_k = min(n_points, max(k, int(knn_cache_k)))

    knn_model, cached_ids, cached_dists = _build_knn_cache(X, initial_cache_k)

    G = nx.Graph()
    G.add_nodes_from(range(n_points))

    # Initial graph construction from the first k cached neighbors.
    for i in range(n_points):
        for neigh_id, neigh_dist in zip(cached_ids[i][:k], cached_dists[i][:k]):
            _add_edge_from_cache(G, i, neigh_id, neigh_dist)

    # Because the graph is undirected, nx.has_path(G, neigh, sink) can be replaced
    # by membership in the sink connected component.  Recompute only after adding
    # a fallback edge.
    sink_component = set(nx.node_connected_component(G, sink))

    current_node = 0
    weighted_path = [current_node]
    checked_nodes = {current_node}

    while current_node != sink:
        tmp_k = k

        while True:
            chosen_nn = -1
            e_weight = 1E30

            if random_neighbors:
                tmp_neighs = []
                for neigh in G.neighbors(current_node):
                    if neigh not in checked_nodes:
                        tmp_neighs.append(neigh)

                if len(tmp_neighs) > 0:
                    chosen_nn = tmp_neighs[random.randint(0, len(tmp_neighs) - 1)]
                    e_weight = G[current_node][chosen_nn]['weight']
                else:
                    print('Node has no neighbors...')
            else:
                for neigh in G.neighbors(current_node):
                    if neigh not in checked_nodes and neigh in sink_component:
                        edge_weight = G[current_node][neigh]['weight']
                        if edge_weight < e_weight:
                            e_weight = edge_weight
                            chosen_nn = neigh

            if chosen_nn > -1:
                break

            print('Failed pathfinding...increasing local k to ' + str(tmp_k) + '...')
            tmp_k += 1

            _ensure_node_has_neighbors(
                X=X,
                knn_model=knn_model,
                cached_ids=cached_ids,
                cached_dists=cached_dists,
                node_id=current_node,
                required_k=tmp_k,
                expand_by=knn_expand_by,
            )

            new_neighbor = int(cached_ids[current_node][tmp_k - 1])
            new_dist = float(cached_dists[current_node][tmp_k - 1])
            edge_added = _add_edge_from_cache(G, current_node, new_neighbor, new_dist)

            if edge_added:
                sink_component = set(nx.node_connected_component(G, sink))

            if tmp_k >= n_points and chosen_nn < 0:
                # At this point all possible neighbors for current_node have been
                # considered.  If no valid unvisited neighbor exists, fail clearly
                # rather than looping forever.
                has_unvisited_neighbor = any(neigh not in checked_nodes for neigh in G.neighbors(current_node))
                if not has_unvisited_neighbor:
                    raise RuntimeError(
                        f"Pathfinding failed at node {current_node}: all possible "
                        "neighbors have already been considered."
                    )

        if chosen_nn < 0:
            weighted_path.pop()
            current_node = weighted_path[-1]
        else:
            current_node = chosen_nn
            checked_nodes.add(current_node)
            weighted_path.append(current_node)

    if version == '1':
        path_edges = list(zip(weighted_path, weighted_path[1:]))
        path_distances = [G[u][v]['weight'] for u, v in path_edges]
        travelled_distances = _normalized_cumulative_distances_from_edge_lengths(path_distances)
        x_to_path, x_to_path_dist = nearest_path_node(X, weighted_path)

        data = {
            "path": x_to_path,
            "path_dist": x_to_path_dist,
            "graph": G,
            "weighted_path": X[weighted_path],
            "d": travelled_distances,
            "path_edges": path_edges,
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

        binned_data = bin_data_by_nearest_node(X, new_weighted_path)
        weighted_path = np.asarray(get_cluster_centroids(X, binned_data), dtype=float)
        travelled_distances = _normalized_cumulative_distances_from_coords(weighted_path)

        data = {
            "weighted_path": weighted_path,
            "d": travelled_distances,
        }
        return data

    elif version == '3':
        new_weighted_path = []
        take = math.ceil(reduction * len(weighted_path))
        accumulated_path = []

        for i in range(len(weighted_path)):
            accumulated_path.append(weighted_path[i])
            if i % take == 0:
                new_weighted_path.append(X[accumulated_path].mean(axis=0))
                accumulated_path = []

        if len(accumulated_path) > 0:
            new_weighted_path.append(X[accumulated_path].mean(axis=0))

        binned_data = bin_data_by_nearest_node(X, new_weighted_path, version='3')
        weighted_path = np.asarray(get_cluster_centroids(X, binned_data), dtype=float)
        travelled_distances = _normalized_cumulative_distances_from_coords(weighted_path)

        data = {
            "weighted_path": weighted_path,
            "d": travelled_distances,
        }
        return data

    else:
        raise ValueError("version must be one of '1', '2', or '3'.")


def get_cluster_centroids(X, bins):
    """Compute centroids for non-empty bins of point indices."""
    X = _as_2d_float_array(X)
    centroids = []
    for bin_ids in bins:
        if len(bin_ids) > 0:
            centroids.append(X[np.asarray(bin_ids, dtype=int)].mean(axis=0).tolist())
    return centroids


def bin_data_by_nearest_node(X, nodes, version='1'):
    """
    Bin each point in X by its nearest node.

    For version != '3', nodes are interpreted as indices into X.
    For version == '3', nodes are interpreted as coordinate vectors.
    """
    X = _as_2d_float_array(X)

    if len(nodes) == 0:
        return []

    if version != '3':
        node_coords = X[np.asarray(nodes, dtype=int)]
    else:
        node_coords = np.asarray(nodes, dtype=float)
        if node_coords.ndim != 2:
            raise ValueError("For version='3', nodes must be coordinate vectors.")

    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn.fit(node_coords)
    _, nearest = knn.kneighbors(X)
    nearest = nearest.flatten()

    bins = [[] for _ in range(len(node_coords))]
    for point_idx, bin_idx in enumerate(nearest):
        bins[int(bin_idx)].append(point_idx)

    return bins


def nearest_path_node(x, nodes):
    """Return nearest path-node index and distance for every point in x."""
    x = _as_2d_float_array(x)
    nodes = np.asarray(nodes, dtype=int)

    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn.fit(x[nodes])
    distances, indices = knn.kneighbors(x)

    return indices.flatten(), distances.flatten()


def find_nearest_nodes(x, nodes, k=1):
    """Return nearest node indices and distances using the original k+1 convention."""
    x = _as_2d_float_array(x)
    nodes = _as_2d_float_array(nodes)

    n_neighbors = min(k + 1, len(nodes))
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(nodes)
    distances, indices = knn.kneighbors(x)
    return indices, distances


def assign_gammas(ref_data, new_data, path_data, smearing='sum', k=1, iterations=1, cutoff=1000.0, scale=10.0):
    """
    Assign path-coordinate gamma values to new_data.

    This preserves the original default smearing='sum' behavior, where the first
    nearest reference gamma is assigned.  The sigmoid branch has been corrected to
    avoid the original division-by-zero typo.
    """
    new_data = _as_2d_float_array(new_data)
    gammas = list(path_data['d'])

    if iterations > 1:
        new_ids, new_dists = find_nearest_nodes(new_data, new_data, k)

    reference = np.asarray(path_data['weighted_path'], dtype=float)
    ref_ids, ref_dists = find_nearest_nodes(new_data, reference, k)

    for iteration in range(iterations):
        assigned_gammas = [0.0] * len(new_data)

        for i, _point in enumerate(new_data):
            avg_gamma = 0.0

            if iteration == 0:
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
                            weight = math.tanh(scale / dists[j])
                        elif smearing == 'sigmoid':
                            weight = 1 / (1.0 + math.exp(dists[j]))
                        else:
                            raise ValueError("smearing must be one of 'sum', 'radial', 'tanh', or 'sigmoid'.")

                    avg_gamma += weight * gammas[ids[j]]
                    if avg_gamma < 0.0:
                        avg_gamma = 0.0

                avg_gamma /= float(len(ids))
            else:
                # Preserve original behavior: use the first nearest gamma even if
                # multiple neighbors are returned by find_nearest_nodes().
                for _j in range(len(ids)):
                    avg_gamma += gammas[ids[0]]
                avg_gamma /= float(len(ids))

            assigned_gammas[i] = avg_gamma

        gammas = assigned_gammas

    return assigned_gammas


def _gamma_value_to_float(value):
    """Convert original gamma representations to float."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        value = value[0]
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def manual_convolution(data, gammas, k=2, iterations=1, cutoff=10):
    """
    Smooth gamma values over a kNN neighborhood using cosine cutoff weights.

    This vectorized implementation preserves the original normalization by the
    number of queried neighbors, not by the sum of weights.
    """
    data = _as_2d_float_array(data)
    gammas = np.asarray([_gamma_value_to_float(g) for g in gammas], dtype=float)

    n_neighbors = min(k + 1, len(data))
    tree = KDTree(data, leaf_size=2)
    dists, ids = tree.query(data, k=n_neighbors)

    weights = 0.5 * (np.cos((dists * np.pi) / cutoff) + 1.0)
    weights[dists > cutoff] = 0.0
    weights[dists < 0.001] = 1.0

    for _iteration in range(iterations):
        neighbor_gammas = gammas[ids]
        gammas = np.sum(weights * neighbor_gammas, axis=1) / float(ids.shape[1])
        gammas = np.maximum(gammas, 0.0)

    return gammas.tolist()
