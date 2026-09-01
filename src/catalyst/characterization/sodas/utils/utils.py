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


# =============================================================================
# EXTENDED OPTIONAL SODAS PATHFINDING API
# =============================================================================
#
# The definitions below intentionally come AFTER the historical implementation.
# We first retain references to the original public functions, then redefine the
# public names with backward-compatible wrappers. Existing callers therefore
# continue to get the legacy behavior unless they explicitly select a new mode.
#

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

_generate_latent_space_path_legacy = generate_latent_space_path
_assign_gammas_legacy = assign_gammas


def _nonself_knn_table(X, n_neighbors, algorithm='auto'):
    """Return exact sklearn kNN tables with each query point itself removed."""
    X = _as_2d_float_array(X)
    n_points = len(X)
    if n_points < 2:
        return (
            np.empty((n_points, 0), dtype=int),
            np.empty((n_points, 0), dtype=float),
        )

    n_neighbors = max(1, min(int(n_neighbors), n_points - 1))
    query_k = min(n_points, n_neighbors + 1)
    model = NearestNeighbors(n_neighbors=query_k, algorithm=str(algorithm))
    model.fit(X)
    dists, ids = model.kneighbors(X, n_neighbors=query_k)

    out_ids = np.empty((n_points, n_neighbors), dtype=int)
    out_dists = np.empty((n_points, n_neighbors), dtype=float)

    for i in range(n_points):
        keep = ids[i] != i
        row_ids = ids[i][keep][:n_neighbors]
        row_dists = dists[i][keep][:n_neighbors]
        if len(row_ids) < n_neighbors:
            # Possible with duplicate coordinates / ambiguous self ordering.
            full_dists, full_ids = model.kneighbors(X[i:i + 1], n_neighbors=n_points)
            keep = full_ids[0] != i
            row_ids = full_ids[0][keep][:n_neighbors]
            row_dists = full_dists[0][keep][:n_neighbors]
        out_ids[i] = row_ids
        out_dists[i] = row_dists

    return out_ids, out_dists


def _local_density_scales(knn_dists, density_k):
    """Distance to the density_k-th non-self neighbor as a local sparsity scale."""
    knn_dists = np.asarray(knn_dists, dtype=float)
    if knn_dists.ndim != 2 or knn_dists.shape[1] == 0:
        return np.ones(knn_dists.shape[0], dtype=float)

    density_k = max(1, min(int(density_k), knn_dists.shape[1]))
    scales = knn_dists[:, density_k - 1].copy()
    finite_positive = scales[np.isfinite(scales) & (scales > 0.0)]
    fallback = float(np.median(finite_positive)) if len(finite_positive) else 1.0
    bad = ~np.isfinite(scales) | (scales <= 0.0)
    scales[bad] = fallback
    return scales


def _build_sparse_knn_graph_from_table(
    X,
    knn_ids,
    knn_dists,
    k,
    graph_mode='symmetric_knn',
    density_weight_alpha=0.0,
    density_scales=None,
):
    """Build an undirected sparse kNN graph from a precomputed neighbor table."""
    X = _as_2d_float_array(X)
    n_points = len(X)
    available = int(knn_ids.shape[1])
    if available < 1:
        raise ValueError("kNN table has no non-self neighbors.")
    k = max(1, min(int(k), available))

    ids = np.asarray(knn_ids[:, :k], dtype=int)
    dists = np.asarray(knn_dists[:, :k], dtype=float)
    rows = np.repeat(np.arange(n_points, dtype=int), k)
    cols = ids.reshape(-1)
    vals = dists.reshape(-1)

    keep = rows != cols
    directed = coo_matrix(
        (vals[keep], (rows[keep], cols[keep])),
        shape=(n_points, n_points),
        dtype=float,
    ).tocsr()
    directed.eliminate_zeros()

    mode = str(graph_mode).lower()
    if mode in {'symmetric_knn', 'symmetric', 'union'}:
        graph = directed.maximum(directed.T).tocsr()
    elif mode in {'mutual_knn', 'mutual', 'reciprocal'}:
        reverse_present = directed.T.copy()
        reverse_present.data = np.ones_like(reverse_present.data)
        graph = directed.multiply(reverse_present).tocsr()
        graph = ((graph + graph.T) * 0.5).tocsr()
    else:
        raise ValueError("graph_mode must be 'symmetric_knn' or 'mutual_knn'.")

    graph.setdiag(0.0)
    graph.eliminate_zeros()

    alpha = float(density_weight_alpha)
    if alpha < 0.0:
        raise ValueError("density_weight_alpha must be >= 0.")
    if alpha > 0.0:
        if density_scales is None:
            raise ValueError("density_scales are required for density weighting.")
        density_scales = np.asarray(density_scales, dtype=float)
        finite_positive = density_scales[
            np.isfinite(density_scales) & (density_scales > 0.0)
        ]
        reference_scale = float(np.median(finite_positive)) if len(finite_positive) else 1.0
        reference_scale = max(reference_scale, np.finfo(float).eps)

        coo = graph.tocoo()
        local_ratio = (
            density_scales[coo.row] + density_scales[coo.col]
        ) / (2.0 * reference_scale)
        # No discount in dense regions; only sparse regions receive a penalty.
        penalty = 1.0 + alpha * np.maximum(local_ratio - 1.0, 0.0)
        graph = coo_matrix(
            (coo.data * penalty, (coo.row, coo.col)),
            shape=graph.shape,
        ).tocsr()

    return graph


def _nodes_connected(graph, start, sink):
    if int(start) == int(sink):
        return True
    _n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return int(labels[int(start)]) == int(labels[int(sink)])


def _reconstruct_predecessor_path(predecessors, start, sink):
    start = int(start)
    sink = int(sink)
    if start == sink:
        return [start]

    path = [sink]
    current = sink
    while current != start:
        current = int(predecessors[current])
        if current == -9999:
            raise RuntimeError(
                f"No predecessor chain exists between start={start} and sink={sink}."
            )
        path.append(current)
        if len(path) > len(predecessors) + 1:
            raise RuntimeError("Predecessor reconstruction exceeded graph size.")
    path.reverse()
    return path


def _rdp_keep_indices(points, tolerance):
    """Ramer-Douglas-Peucker simplification, returning indices into points."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    tolerance = float(tolerance)
    if n <= 2 or tolerance <= 0.0:
        return np.arange(n, dtype=int)

    def recurse(start, end, keep):
        if end <= start + 1:
            return
        a = points[start]
        b = points[end]
        ab = b - a
        denom = float(np.dot(ab, ab))
        middle = points[start + 1:end]
        if len(middle) == 0:
            return
        if denom <= np.finfo(float).eps:
            distances = np.linalg.norm(middle - a, axis=1)
        else:
            t = ((middle - a) @ ab) / denom
            projection = a + t[:, None] * ab
            distances = np.linalg.norm(middle - projection, axis=1)
        rel = int(np.argmax(distances))
        max_distance = float(distances[rel])
        split = start + 1 + rel
        if max_distance > tolerance:
            keep.add(split)
            recurse(start, split, keep)
            recurse(split, end, keep)

    keep = {0, n - 1}
    recurse(0, n - 1, keep)
    return np.asarray(sorted(keep), dtype=int)


def _build_dijkstra_path_output(
    X,
    path_indices,
    graph_sparse,
    version,
    reduction,
    *,
    gamma_distance='euclidean',
    return_networkx_graph=False,
    path_simplify_tolerance=0.0,
):
    """Convert Dijkstra path indices into Catalyst-style path_data."""
    X = _as_2d_float_array(X)
    path_indices = np.asarray(path_indices, dtype=int)
    raw_path_indices = path_indices.copy()

    tolerance = float(path_simplify_tolerance)
    if tolerance > 0.0 and len(path_indices) > 2:
        keep = _rdp_keep_indices(X[path_indices], tolerance)
        path_indices = path_indices[keep]

    if version == '1':
        path_coords = X[path_indices]
        if str(gamma_distance).lower() == 'graph_cost':
            if not np.array_equal(path_indices, raw_path_indices):
                raise ValueError(
                    "gamma_distance='graph_cost' cannot be combined with path simplification."
                )
            edge_lengths = []
            for u, v in zip(path_indices[:-1], path_indices[1:]):
                edge_lengths.append(float(graph_sparse[int(u), int(v)]))
            travelled = _normalized_cumulative_distances_from_edge_lengths(edge_lengths)
        else:
            travelled = _normalized_cumulative_distances_from_coords(path_coords).tolist()

        x_to_path, x_to_path_dist = nearest_path_node(X, path_indices)
        path_edges = list(zip(path_indices.tolist(), path_indices[1:].tolist()))
        return {
            'path': x_to_path,
            'path_dist': x_to_path_dist,
            'graph': nx.from_scipy_sparse_array(graph_sparse) if return_networkx_graph else None,
            'graph_sparse': graph_sparse,
            'weighted_path': path_coords,
            'd': travelled,
            'path_edges': path_edges,
            'path_indices': path_indices.tolist(),
            'raw_path_indices': raw_path_indices.tolist(),
        }

    if version == '2':
        new_weighted_path = [int(path_indices[0])]
        take = max(1, math.ceil(reduction * len(path_indices)))
        for i in range(len(path_indices)):
            if i % take == 0:
                new_weighted_path.append(int(path_indices[i]))
        new_weighted_path.append(int(path_indices[-1]))
        dedup = []
        for value in new_weighted_path:
            if not dedup or value != dedup[-1]:
                dedup.append(value)
        binned_data = bin_data_by_nearest_node(X, dedup)
        weighted_path = np.asarray(get_cluster_centroids(X, binned_data), dtype=float)
        return {
            'weighted_path': weighted_path,
            'd': _normalized_cumulative_distances_from_coords(weighted_path),
            'path_indices': path_indices.tolist(),
            'raw_path_indices': raw_path_indices.tolist(),
            'graph_sparse': graph_sparse,
        }

    if version == '3':
        take = max(1, math.ceil(reduction * len(path_indices)))
        new_weighted_path = []
        accumulated = []
        for i, node_id in enumerate(path_indices):
            accumulated.append(int(node_id))
            if i % take == 0:
                new_weighted_path.append(X[np.asarray(accumulated, dtype=int)].mean(axis=0))
                accumulated = []
        if accumulated:
            new_weighted_path.append(X[np.asarray(accumulated, dtype=int)].mean(axis=0))
        binned_data = bin_data_by_nearest_node(X, new_weighted_path, version='3')
        weighted_path = np.asarray(get_cluster_centroids(X, binned_data), dtype=float)
        return {
            'weighted_path': weighted_path,
            'd': _normalized_cumulative_distances_from_coords(weighted_path),
            'path_indices': path_indices.tolist(),
            'raw_path_indices': raw_path_indices.tolist(),
            'graph_sparse': graph_sparse,
        }

    raise ValueError("version must be one of '1', '2', or '3'.")


def _generate_latent_space_path_dijkstra(
    X,
    k=7,
    boundaries=[0, -1],
    version='1',
    reduction=1.0,
    *,
    graph_mode='symmetric_knn',
    auto_expand_k=True,
    k_step=2,
    max_k=None,
    knn_algorithm='auto',
    density_weight_alpha=0.0,
    density_k=None,
    gamma_distance='euclidean',
    return_networkx_graph=False,
    path_simplify_tolerance=0.0,
):
    """Sparse kNN + globally optimal Dijkstra endpoint-to-endpoint path."""
    X = _apply_boundaries(X, boundaries)
    n_points = len(X)
    if n_points < 2:
        raise ValueError("At least two latent points are required.")

    start = 0
    sink = n_points - 1
    initial_k = max(1, min(int(k), n_points - 1))
    k_step = max(1, int(k_step))

    if max_k is None:
        max_k = min(n_points - 1, max(initial_k, initial_k * 4, initial_k + 32))
    else:
        max_k = max(initial_k, min(int(max_k), n_points - 1))
    if not auto_expand_k:
        max_k = initial_k

    if density_k is None:
        density_k = max(initial_k, 7)
    density_k = max(1, min(int(density_k), n_points - 1))

    cache_k = max(max_k, density_k)
    knn_ids, knn_dists = _nonself_knn_table(
        X,
        n_neighbors=cache_k,
        algorithm=knn_algorithm,
    )
    density_scales = None
    if float(density_weight_alpha) > 0.0:
        density_scales = _local_density_scales(knn_dists, density_k)

    used_k = initial_k
    while True:
        graph_sparse = _build_sparse_knn_graph_from_table(
            X,
            knn_ids,
            knn_dists,
            used_k,
            graph_mode=graph_mode,
            density_weight_alpha=density_weight_alpha,
            density_scales=density_scales,
        )
        if _nodes_connected(graph_sparse, start, sink):
            break
        if used_k >= max_k:
            raise RuntimeError(
                "Sparse SODAS endpoints remain disconnected after global k expansion "
                f"to k={used_k}. Increase max_k or switch graph_mode."
            )
        used_k = min(max_k, used_k + k_step)

    distances, predecessors = sparse_dijkstra(
        graph_sparse,
        directed=False,
        indices=start,
        return_predecessors=True,
    )
    if not np.isfinite(distances[sink]):
        raise RuntimeError("Dijkstra could not connect the requested endpoints.")

    path_indices = _reconstruct_predecessor_path(predecessors, start, sink)
    gamma_distance = str(gamma_distance).lower()
    if gamma_distance not in {'euclidean', 'graph_cost'}:
        raise ValueError("gamma_distance must be 'euclidean' or 'graph_cost'.")

    data = _build_dijkstra_path_output(
        X,
        path_indices,
        graph_sparse,
        version,
        reduction,
        gamma_distance=gamma_distance,
        return_networkx_graph=bool(return_networkx_graph),
        path_simplify_tolerance=float(path_simplify_tolerance),
    )
    data.update({
        'pathfinder': 'dijkstra',
        'graph_mode': str(graph_mode),
        'initial_k': int(initial_k),
        'used_k': int(used_k),
        'max_k': int(max_k),
        'auto_expand_k': bool(auto_expand_k),
        'density_weight_alpha': float(density_weight_alpha),
        'density_k': int(density_k),
        'gamma_distance': gamma_distance,
        'dijkstra_total_cost': float(distances[sink]),
    })
    return data


def generate_latent_space_path(
    X,
    k=7,
    boundaries=[0, -1],
    version='1',
    reduction=1.0,
    random_neighbors=False,
    knn_cache_k=None,
    knn_expand_by=1,
    *,
    pathfinder='legacy',
    graph_mode='symmetric_knn',
    auto_expand_k=True,
    k_step=2,
    max_k=None,
    knn_algorithm='auto',
    density_weight_alpha=0.0,
    density_k=None,
    gamma_distance='euclidean',
    return_networkx_graph=False,
    path_simplify_tolerance=0.0,
):
    """
    Generate a latent-space path with either the historical Catalyst algorithm
    or the new sparse Dijkstra algorithm.

    pathfinder='legacy' (DEFAULT)
        Calls the existing Catalyst greedy pathfinder unchanged.

    pathfinder='dijkstra'
        Builds a sparse kNN graph and computes the true shortest path with
        scipy.sparse.csgraph.dijkstra.

    Dijkstra-only options
    ---------------------
    graph_mode : {'symmetric_knn', 'mutual_knn'}
    auto_expand_k : bool
        Increase k globally until the endpoints are connected.
    k_step, max_k : int
        Control global connectivity expansion.
    knn_algorithm : {'auto', 'kd_tree', 'ball_tree', 'brute'}
        Passed to sklearn NearestNeighbors.
    density_weight_alpha : float >= 0
        Penalize edges in sparsely sampled latent regions.
    density_k : int
        Neighbor rank used for the local density/sparsity scale.
    gamma_distance : {'euclidean', 'graph_cost'}
        Recommended default is Euclidean arclength even when density weighting
        is used, so gamma is not distorted by the path penalty.
    path_simplify_tolerance : float
        Optional RDP path simplification in latent-space units.
    return_networkx_graph : bool
        Materialize a NetworkX graph in addition to graph_sparse.

    Existing calls require no changes because the default is pathfinder='legacy'.
    """
    mode = str(pathfinder).lower()
    if mode in {'legacy', 'greedy', 'original'}:
        return _generate_latent_space_path_legacy(
            X=X,
            k=k,
            boundaries=boundaries,
            version=version,
            reduction=reduction,
            random_neighbors=random_neighbors,
            knn_cache_k=knn_cache_k,
            knn_expand_by=knn_expand_by,
        )

    if mode in {'dijkstra', 'sparse_dijkstra', 'shortest_path'}:
        if random_neighbors:
            raise ValueError("random_neighbors is available only in legacy mode.")
        return _generate_latent_space_path_dijkstra(
            X=X,
            k=k,
            boundaries=boundaries,
            version=version,
            reduction=reduction,
            graph_mode=graph_mode,
            auto_expand_k=auto_expand_k,
            k_step=k_step,
            max_k=max_k,
            knn_algorithm=knn_algorithm,
            density_weight_alpha=density_weight_alpha,
            density_k=density_k,
            gamma_distance=gamma_distance,
            return_networkx_graph=return_networkx_graph,
            path_simplify_tolerance=path_simplify_tolerance,
        )

    raise ValueError("pathfinder must be 'legacy' or 'dijkstra'.")


# =============================================================================
# OPTIONAL ROBUST ENDPOINT SELECTION
# =============================================================================

def _exact_diameter_blockwise(X, block_size=1024, candidate_indices=None):
    """Exact Euclidean diameter with bounded memory."""
    X = _as_2d_float_array(X)
    if candidate_indices is None:
        candidate_indices = np.arange(len(X), dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
    if len(candidate_indices) < 2:
        raise ValueError("At least two endpoint candidates are required.")

    Y = X[candidate_indices]
    norms = np.sum(Y * Y, axis=1)
    block_size = max(1, int(block_size))
    best_d2 = -np.inf
    best_i, best_j = 0, 1
    for i0 in range(0, len(Y), block_size):
        i1 = min(i0 + block_size, len(Y))
        d2 = norms[i0:i1, None] + norms[None, :] - 2.0 * Y[i0:i1] @ Y.T
        np.maximum(d2, 0.0, out=d2)
        flat = int(np.argmax(d2))
        ii, jj = np.unravel_index(flat, d2.shape)
        value = float(d2[ii, jj])
        if value > best_d2:
            best_d2 = value
            best_i = i0 + int(ii)
            best_j = int(jj)
    return (
        int(candidate_indices[best_i]),
        int(candidate_indices[best_j]),
        float(math.sqrt(max(best_d2, 0.0))),
    )


def _convex_hull_diameter_2d(X, candidate_indices=None):
    """Exact 2-D diameter using convex hull followed by rotating calipers."""
    X = _as_2d_float_array(X)
    if X.shape[1] != 2:
        raise ValueError("convex_hull_2d requires exactly two latent dimensions.")
    if candidate_indices is None:
        candidate_indices = np.arange(len(X), dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
    if len(candidate_indices) < 2:
        raise ValueError("At least two endpoint candidates are required.")

    Y = X[candidate_indices]
    if len(Y) <= 3:
        return _exact_diameter_blockwise(X, candidate_indices=candidate_indices)

    try:
        hull = ConvexHull(Y)
    except QhullError:
        return _exact_diameter_blockwise(X, candidate_indices=candidate_indices)

    hull_local = np.asarray(hull.vertices, dtype=int)
    P = Y[hull_local]
    H = len(P)
    if H <= 3:
        return _exact_diameter_blockwise(
            X,
            candidate_indices=candidate_indices[hull_local],
        )

    def area2(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def dist2(a, b):
        d = a - b
        return float(np.dot(d, d))

    best_d2 = -1.0
    best_pair = (0, 1)
    j = 1
    for i in range(H):
        ni = (i + 1) % H
        while True:
            nj = (j + 1) % H
            if area2(P[i], P[ni], P[nj]) > area2(P[i], P[ni], P[j]) + 1.0e-15:
                j = nj
            else:
                break
        for a, b in ((i, j), (ni, j)):
            value = dist2(P[a], P[b])
            if value > best_d2:
                best_d2 = value
                best_pair = (a, b)

    local_a = int(hull_local[best_pair[0]])
    local_b = int(hull_local[best_pair[1]])
    return (
        int(candidate_indices[local_a]),
        int(candidate_indices[local_b]),
        float(math.sqrt(max(best_d2, 0.0))),
    )


def find_latent_space_endpoints(
    X,
    method='auto',
    *,
    density_k=10,
    density_quantile=0.02,
    block_size=1024,
    knn_algorithm='auto',
):
    """
    Select two ACTUAL latent points as path endpoints.

    method='auto'
        Exact convex-hull diameter in 2-D; exact blockwise diameter otherwise.
    method='exact'
        Exact blockwise Euclidean diameter in any dimension.
    method='convex_hull_2d'
        Exact 2-D hull + rotating-calipers diameter.
    method='density_supported'
        Reject the sparsest density_quantile fraction first, then compute the
        diameter among the remaining actual points.
    """
    X = _as_2d_float_array(X)
    if len(X) < 2:
        raise ValueError("At least two latent points are required.")

    mode = str(method).lower()
    candidate_mask = np.ones(len(X), dtype=bool)
    density_scale = None

    if mode == 'density_supported':
        density_k = max(1, min(int(density_k), len(X) - 1))
        _ids, dists = _nonself_knn_table(X, density_k, algorithm=knn_algorithm)
        density_scale = _local_density_scales(dists, density_k)
        q = float(density_quantile)
        if not (0.0 <= q < 0.5):
            raise ValueError("density_quantile must satisfy 0 <= q < 0.5.")
        threshold = float(np.quantile(density_scale, 1.0 - q)) if q > 0.0 else np.inf
        candidate_mask = density_scale <= threshold
        candidates = np.flatnonzero(candidate_mask)
        if len(candidates) < 2:
            raise RuntimeError("Density filtering left fewer than two endpoint candidates.")
        if X.shape[1] == 2:
            a, b, distance = _convex_hull_diameter_2d(X, candidates)
            resolved = 'density_supported_convex_hull_2d'
        else:
            a, b, distance = _exact_diameter_blockwise(X, block_size, candidates)
            resolved = 'density_supported_exact'
    elif mode == 'auto':
        if X.shape[1] == 2:
            a, b, distance = _convex_hull_diameter_2d(X)
            resolved = 'convex_hull_2d'
        else:
            a, b, distance = _exact_diameter_blockwise(X, block_size)
            resolved = 'exact'
    elif mode in {'convex_hull_2d', 'hull', 'hull_2d'}:
        a, b, distance = _convex_hull_diameter_2d(X)
        resolved = 'convex_hull_2d'
    elif mode in {'exact', 'blockwise_exact'}:
        a, b, distance = _exact_diameter_blockwise(X, block_size)
        resolved = 'exact'
    else:
        raise ValueError(
            "method must be 'auto', 'exact', 'convex_hull_2d', or 'density_supported'."
        )

    return {
        'start_index': int(a),
        'end_index': int(b),
        'distance': float(distance),
        'method': resolved,
        'candidate_mask': candidate_mask,
        'local_density_scale': density_scale,
    }


# =============================================================================
# OPTIONAL CONTINUOUS GAMMA ASSIGNMENT
# =============================================================================

def project_onto_piecewise_linear_path(new_data, path_coords, path_d=None, *, chunk_size=2048):
    """
    Project points onto the nearest piecewise-linear path segment.

    Returns continuous gamma, perpendicular distance-to-path, segment index,
    within-segment interpolation t, and projected coordinates.
    """
    X = _as_2d_float_array(new_data)
    P = _as_2d_float_array(path_coords)
    if len(P) == 0:
        raise ValueError("path_coords must contain at least one point.")

    if len(P) == 1:
        return {
            'gamma': np.zeros(len(X), dtype=float),
            'path_distance': np.linalg.norm(X - P[0], axis=1),
            'segment_index': np.zeros(len(X), dtype=int),
            'segment_t': np.zeros(len(X), dtype=float),
            'projected_points': np.repeat(P[0:1], len(X), axis=0),
        }

    if path_d is None:
        D = _normalized_cumulative_distances_from_coords(P)
    else:
        D = np.asarray(path_d, dtype=float)
        if D.ndim != 1 or len(D) != len(P):
            raise ValueError("path_d must contain one cumulative coordinate per path point.")

    A = P[:-1]
    V = P[1:] - P[:-1]
    VV = np.sum(V * V, axis=1)
    safe_VV = np.where(VV > np.finfo(float).eps, VV, 1.0)

    n = len(X)
    chunk_size = max(1, int(chunk_size))
    best_gamma = np.empty(n, dtype=float)
    best_distance = np.empty(n, dtype=float)
    best_segment = np.empty(n, dtype=int)
    best_t = np.empty(n, dtype=float)
    best_projected = np.empty_like(X, dtype=float)

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        C = X[i0:i1]
        delta = C[:, None, :] - A[None, :, :]
        t = np.sum(delta * V[None, :, :], axis=2) / safe_VV[None, :]
        t = np.clip(t, 0.0, 1.0)
        projected = A[None, :, :] + t[:, :, None] * V[None, :, :]
        residual = C[:, None, :] - projected
        dist2 = np.sum(residual * residual, axis=2)
        seg = np.argmin(dist2, axis=1)
        row = np.arange(len(C), dtype=int)
        chosen_t = t[row, seg]
        best_gamma[i0:i1] = D[seg] + chosen_t * (D[seg + 1] - D[seg])
        best_distance[i0:i1] = np.sqrt(np.maximum(dist2[row, seg], 0.0))
        best_segment[i0:i1] = seg
        best_t[i0:i1] = chosen_t
        best_projected[i0:i1] = projected[row, seg]

    return {
        'gamma': best_gamma,
        'path_distance': best_distance,
        'segment_index': best_segment,
        'segment_t': best_t,
        'projected_points': best_projected,
    }


def assign_gammas(
    ref_data,
    new_data,
    path_data,
    smearing='sum',
    k=1,
    iterations=1,
    cutoff=1000.0,
    scale=10.0,
    *,
    assignment_mode='nearest_node',
    projection_chunk_size=2048,
    return_path_distance=False,
):
    """
    Assign gamma using either historical nearest-node behavior or continuous
    piecewise-linear segment projection.

    assignment_mode='nearest_node' is the DEFAULT and preserves old behavior.
    assignment_mode='segment_projection' returns continuous arclength gamma.
    """
    mode = str(assignment_mode).lower()
    if mode in {'nearest_node', 'legacy', 'original'}:
        if return_path_distance:
            raise ValueError(
                "return_path_distance=True is available only for segment_projection."
            )
        return _assign_gammas_legacy(
            ref_data=ref_data,
            new_data=new_data,
            path_data=path_data,
            smearing=smearing,
            k=k,
            iterations=iterations,
            cutoff=cutoff,
            scale=scale,
        )

    if mode in {'segment_projection', 'continuous', 'piecewise_linear'}:
        if int(iterations) != 1:
            raise ValueError("iterations > 1 is not defined for segment_projection mode.")
        result = project_onto_piecewise_linear_path(
            new_data,
            np.asarray(path_data['weighted_path'], dtype=float),
            np.asarray(path_data['d'], dtype=float),
            chunk_size=projection_chunk_size,
        )
        return result if return_path_distance else result['gamma'].tolist()

    raise ValueError("assignment_mode must be 'nearest_node' or 'segment_projection'.")


# Public export list extension without changing older import behavior.
for _name in (
    'find_latent_space_endpoints',
    'project_onto_piecewise_linear_path',
    'assign_gammas',
):
    if _name not in __all__:
        __all__.append(_name)



# =============================================================================
# K-MEANS / COMPRESSED-MANIFOLD EXTENSION
# =============================================================================
#
# This extension is intentionally appended after the existing backward-compatible
# SODAS wrappers. It adds new options while keeping every previous call valid.
#
# New public capabilities:
#   cluster_latent_space_kmeans(...)
#   find_latent_space_endpoints(..., method="kmeans_diameter")
#   generate_latent_space_path(..., path_space="raw")
#   generate_latent_space_path(..., path_space="kmeans_centroids")
#   generate_latent_space_path(..., path_space="kmeans_raw_topology")
#
# Existing defaults remain:
#   endpoint method: existing behavior
#   path_space="raw"
#   pathfinder="legacy"
#

from sklearn.cluster import KMeans, MiniBatchKMeans

_find_latent_space_endpoints_pre_kmeans = find_latent_space_endpoints
_generate_latent_space_path_pre_kmeans = generate_latent_space_path


def _resolve_kmeans_cluster_count(n_points, n_clusters=None, max_clusters=512):
    """
    Resolve K for latent-space clustering.

    None or "sqrt" uses round(sqrt(N)), bounded to [2, max_clusters].
    """
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("K-means requires at least two points.")

    max_allowed = max(2, min(int(max_clusters), n_points))

    if n_clusters is None or (
        isinstance(n_clusters, str) and str(n_clusters).lower() == "sqrt"
    ):
        k = int(round(math.sqrt(n_points)))
    else:
        k = int(n_clusters)

    return max(2, min(k, max_allowed))


def cluster_latent_space_kmeans(
    X,
    n_clusters=None,
    *,
    random_state=112358,
    n_init=10,
    max_iter=300,
    algorithm="auto",
    max_clusters=512,
    minibatch_threshold=50000,
    minibatch_size=2048,
):
    """
    Cluster a latent cloud and return reusable cluster information.

    Parameters
    ----------
    X : array-like, shape (N, D)
    n_clusters : int, "sqrt", or None
        None/"sqrt" uses approximately sqrt(N).
    algorithm : {"auto", "kmeans", "minibatch"}
        "auto" uses standard KMeans below minibatch_threshold and
        MiniBatchKMeans above it.

    Returns
    -------
    dict
        labels, centroids, counts, representative_indices, inertia, and metadata.

    representative_indices[k] is the ACTUAL observed point in cluster k nearest
    its centroid. This is useful when a downstream routine requires real
    structures rather than synthetic centroid coordinates.
    """
    X = _as_2d_float_array(X)
    n_points = len(X)
    k = _resolve_kmeans_cluster_count(
        n_points,
        n_clusters=n_clusters,
        max_clusters=max_clusters,
    )

    mode = str(algorithm).lower()
    if mode == "auto":
        mode = "minibatch" if n_points >= int(minibatch_threshold) else "kmeans"

    if mode == "kmeans":
        model = KMeans(
            n_clusters=k,
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
        )
    elif mode in {"minibatch", "mini_batch", "minibatchkmeans"}:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
            batch_size=max(k, int(minibatch_size)),
        )
    else:
        raise ValueError("algorithm must be 'auto', 'kmeans', or 'minibatch'.")

    labels = np.asarray(model.fit_predict(X), dtype=int)
    centroids = np.asarray(model.cluster_centers_, dtype=float)
    counts = np.bincount(labels, minlength=k).astype(int)

    representative_indices = np.full(k, -1, dtype=int)
    representative_distances = np.full(k, np.nan, dtype=float)

    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id)
        if len(members) == 0:
            continue
        delta = X[members] - centroids[cluster_id]
        d2 = np.sum(delta * delta, axis=1)
        local = int(np.argmin(d2))
        representative_indices[cluster_id] = int(members[local])
        representative_distances[cluster_id] = float(math.sqrt(max(float(d2[local]), 0.0)))

    return {
        "labels": labels,
        "centroids": centroids,
        "counts": counts,
        "representative_indices": representative_indices,
        "representative_distances": representative_distances,
        "n_clusters": int(k),
        "algorithm": mode,
        "random_state": int(random_state),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "inertia": float(model.inertia_),
    }


def _validate_kmeans_result(X, result):
    X = _as_2d_float_array(X)
    if result is None:
        raise ValueError("kmeans_result cannot be None.")
    required = {
        "labels",
        "centroids",
        "counts",
        "representative_indices",
        "n_clusters",
    }
    missing = required - set(result)
    if missing:
        raise KeyError(f"kmeans_result is missing keys: {sorted(missing)}")

    labels = np.asarray(result["labels"], dtype=int)
    centroids = np.asarray(result["centroids"], dtype=float)
    counts = np.asarray(result["counts"], dtype=int)
    representatives = np.asarray(result["representative_indices"], dtype=int)

    if len(labels) != len(X):
        raise ValueError(
            f"kmeans_result has {len(labels)} labels for {len(X)} input points."
        )
    if centroids.ndim != 2 or centroids.shape[1] != X.shape[1]:
        raise ValueError("kmeans_result centroid dimensionality does not match X.")
    if len(counts) != len(centroids):
        raise ValueError("kmeans_result counts do not match centroid count.")
    if len(representatives) != len(centroids):
        raise ValueError("kmeans_result representatives do not match centroid count.")

    return labels, centroids, counts, representatives


def _centroid_diameter_pair(centroids, candidate_ids):
    """Exact centroid-diameter pair for a normally modest number of clusters."""
    centroids = _as_2d_float_array(centroids)
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    if len(candidate_ids) < 2:
        raise ValueError("Need at least two supported clusters.")

    Y = centroids[candidate_ids]
    a_local, b_local, distance = _exact_diameter_blockwise(Y, block_size=1024)
    return int(candidate_ids[a_local]), int(candidate_ids[b_local]), float(distance)


def find_latent_space_endpoints(
    X,
    method="auto",
    *,
    density_k=10,
    density_quantile=0.02,
    block_size=1024,
    knn_algorithm="auto",
    kmeans_result=None,
    kmeans_n_clusters=None,
    kmeans_min_cluster_size=5,
    kmeans_random_state=112358,
    kmeans_n_init=10,
    kmeans_algorithm="auto",
    kmeans_max_clusters=512,
):
    """
    Extended endpoint selector.

    Existing methods are unchanged:
        auto
        exact
        convex_hull_2d
        density_supported

    New method:
        kmeans_diameter

    kmeans_diameter clusters the entire latent cloud, filters endpoint candidate
    clusters by kmeans_min_cluster_size, finds the maximally separated pair of
    supported centroids, then returns the ACTUAL data point nearest each selected
    centroid as start_index/end_index.

    The selected cluster centroids and IDs are also returned so a compressed
    path can use the centroid states directly.
    """
    mode = str(method).lower()

    if mode not in {
        "kmeans_diameter",
        "kmeans",
        "cluster_diameter",
        "centroid_diameter",
    }:
        return _find_latent_space_endpoints_pre_kmeans(
            X,
            method=method,
            density_k=density_k,
            density_quantile=density_quantile,
            block_size=block_size,
            knn_algorithm=knn_algorithm,
        )

    X = _as_2d_float_array(X)

    if kmeans_result is None:
        kmeans_result = cluster_latent_space_kmeans(
            X,
            n_clusters=kmeans_n_clusters,
            random_state=kmeans_random_state,
            n_init=kmeans_n_init,
            algorithm=kmeans_algorithm,
            max_clusters=kmeans_max_clusters,
        )

    labels, centroids, counts, representatives = _validate_kmeans_result(
        X,
        kmeans_result,
    )

    min_size = max(1, int(kmeans_min_cluster_size))
    supported_clusters = np.flatnonzero(
        (counts >= min_size) & (representatives >= 0)
    )
    if len(supported_clusters) < 2:
        raise RuntimeError(
            "Fewer than two K-means clusters satisfy "
            f"kmeans_min_cluster_size={min_size}."
        )

    start_cluster, end_cluster, centroid_distance = _centroid_diameter_pair(
        centroids,
        supported_clusters,
    )

    start_index = int(representatives[start_cluster])
    end_index = int(representatives[end_cluster])

    candidate_mask = np.isin(labels, supported_clusters)
    actual_distance = float(np.linalg.norm(X[start_index] - X[end_index]))

    return {
        "start_index": start_index,
        "end_index": end_index,
        "distance": actual_distance,
        "method": "kmeans_diameter",
        "candidate_mask": candidate_mask,
        "local_density_scale": None,
        "start_cluster": int(start_cluster),
        "end_cluster": int(end_cluster),
        "start_centroid": centroids[start_cluster].copy(),
        "end_centroid": centroids[end_cluster].copy(),
        "centroid_distance": float(centroid_distance),
        "cluster_counts": counts.copy(),
        "supported_cluster_ids": supported_clusters.copy(),
        "kmeans_result": kmeans_result,
    }


def _endpoint_reorder(X, start_index, end_index):
    """Move requested actual endpoint rows to positions 0 and -1."""
    X = _as_2d_float_array(X)
    n = len(X)

    start = 0 if start_index is None else int(start_index)
    end = n - 1 if end_index is None else int(end_index)

    start %= n
    end %= n

    if start == end:
        raise ValueError("start_index and end_index must identify different points.")

    order = [start]
    order.extend(i for i in range(n) if i not in {start, end})
    order.append(end)
    order = np.asarray(order, dtype=int)
    return X[order], order


def _reorder_kmeans_result(kmeans_result, order):
    """Reorder only point-level labels while preserving centroid-level fields."""
    if kmeans_result is None:
        return None
    result = dict(kmeans_result)
    labels = np.asarray(result["labels"], dtype=int)
    result["labels"] = labels[np.asarray(order, dtype=int)]
    return result


def _cluster_crossing_graph(
    raw_graph,
    labels,
    centroids,
    *,
    edge_weight="median_crossing",
):
    """
    Collapse a raw sparse neighbor graph into a cluster graph.

    Two clusters are adjacent only when at least one raw graph edge crosses
    between them. This preserves sampled manifold topology rather than inventing
    centroid-to-centroid edges solely from Euclidean proximity.
    """
    labels = np.asarray(labels, dtype=int)
    centroids = _as_2d_float_array(centroids)
    n_clusters = len(centroids)

    coo = raw_graph.tocoo()
    crossings = {}

    for i, j, value in zip(coo.row, coo.col, coo.data):
        if int(i) >= int(j):
            continue
        ci = int(labels[int(i)])
        cj = int(labels[int(j)])
        if ci == cj:
            continue
        a, b = (ci, cj) if ci < cj else (cj, ci)
        crossings.setdefault((a, b), []).append(float(value))

    rows = []
    cols = []
    values = []

    mode = str(edge_weight).lower()

    for (a, b), raw_values in crossings.items():
        if mode == "centroid_distance":
            weight = float(np.linalg.norm(centroids[a] - centroids[b]))
        elif mode == "min_crossing":
            weight = float(np.min(raw_values))
        elif mode == "mean_crossing":
            weight = float(np.mean(raw_values))
        elif mode == "median_crossing":
            weight = float(np.median(raw_values))
        else:
            raise ValueError(
                "cluster_edge_weight must be 'centroid_distance', "
                "'min_crossing', 'mean_crossing', or 'median_crossing'."
            )

        if weight <= 0.0:
            weight = float(np.finfo(float).eps)

        rows.extend([a, b])
        cols.extend([b, a])
        values.extend([weight, weight])

    graph = coo_matrix(
        (np.asarray(values, dtype=float), (rows, cols)),
        shape=(n_clusters, n_clusters),
        dtype=float,
    ).tocsr()
    graph.eliminate_zeros()
    return graph


def _generate_kmeans_raw_topology_path(
    X,
    *,
    start_index,
    end_index,
    k=7,
    version="1",
    reduction=1.0,
    graph_mode="symmetric_knn",
    auto_expand_k=True,
    k_step=2,
    max_k=None,
    knn_algorithm="auto",
    density_weight_alpha=0.0,
    density_k=None,
    gamma_distance="euclidean",
    return_networkx_graph=False,
    path_simplify_tolerance=0.0,
    kmeans_result=None,
    kmeans_n_clusters=None,
    kmeans_random_state=112358,
    kmeans_n_init=10,
    kmeans_algorithm="auto",
    kmeans_max_clusters=512,
    cluster_edge_weight="median_crossing",
):
    """
    K-means-compressed Dijkstra path whose topology is inherited from raw-point
    kNN adjacency.
    """
    X = _as_2d_float_array(X)
    n = len(X)

    if kmeans_result is None:
        kmeans_result = cluster_latent_space_kmeans(
            X,
            n_clusters=kmeans_n_clusters,
            random_state=kmeans_random_state,
            n_init=kmeans_n_init,
            algorithm=kmeans_algorithm,
            max_clusters=kmeans_max_clusters,
        )

    labels, centroids, counts, representatives = _validate_kmeans_result(
        X,
        kmeans_result,
    )
    _ = representatives

    start = int(start_index) % n
    end = int(end_index) % n
    start_cluster = int(labels[start])
    end_cluster = int(labels[end])

    if start_cluster == end_cluster:
        raise RuntimeError(
            "Selected path endpoints fall in the same K-means cluster. "
            "Increase kmeans_n_clusters or choose a different endpoint method."
        )

    initial_k = max(1, min(int(k), n - 1))
    k_step = max(1, int(k_step))
    if max_k is None:
        max_k = min(n - 1, max(initial_k + 32, initial_k * 4))
    else:
        max_k = max(initial_k, min(int(max_k), n - 1))
    if not auto_expand_k:
        max_k = initial_k

    if density_k is None:
        density_k = max(initial_k, 7)
    density_k = max(1, min(int(density_k), n - 1))

    cache_k = max(max_k, density_k)
    knn_ids, knn_dists = _nonself_knn_table(
        X,
        n_neighbors=cache_k,
        algorithm=knn_algorithm,
    )

    density_scales = None
    if float(density_weight_alpha) > 0.0:
        density_scales = _local_density_scales(knn_dists, density_k)

    used_k = initial_k
    cluster_graph = None

    while True:
        raw_graph = _build_sparse_knn_graph_from_table(
            X,
            knn_ids,
            knn_dists,
            used_k,
            graph_mode=graph_mode,
            density_weight_alpha=density_weight_alpha,
            density_scales=density_scales,
        )

        cluster_graph = _cluster_crossing_graph(
            raw_graph,
            labels,
            centroids,
            edge_weight=cluster_edge_weight,
        )

        if _nodes_connected(cluster_graph, start_cluster, end_cluster):
            break

        if used_k >= max_k:
            raise RuntimeError(
                "K-means compressed SODAS endpoint clusters remain disconnected "
                f"after raw-graph k expansion to k={used_k}."
            )

        used_k = min(max_k, used_k + k_step)

    distances, predecessors = sparse_dijkstra(
        cluster_graph,
        directed=False,
        indices=start_cluster,
        return_predecessors=True,
    )

    if not np.isfinite(distances[end_cluster]):
        raise RuntimeError("Compressed cluster Dijkstra could not connect endpoints.")

    cluster_path = _reconstruct_predecessor_path(
        predecessors,
        start_cluster,
        end_cluster,
    )

    data = _build_dijkstra_path_output(
        centroids,
        cluster_path,
        cluster_graph,
        version,
        reduction,
        gamma_distance=gamma_distance,
        return_networkx_graph=return_networkx_graph,
        path_simplify_tolerance=path_simplify_tolerance,
    )

    data.update({
        "pathfinder": "dijkstra",
        "path_space": "kmeans_raw_topology",
        "graph_mode": str(graph_mode),
        "initial_k": int(initial_k),
        "used_k": int(used_k),
        "max_k": int(max_k),
        "density_weight_alpha": float(density_weight_alpha),
        "density_k": int(density_k),
        "gamma_distance": str(gamma_distance),
        "dijkstra_total_cost": float(distances[end_cluster]),
        "kmeans_n_clusters": int(len(centroids)),
        "kmeans_start_cluster": int(start_cluster),
        "kmeans_end_cluster": int(end_cluster),
        "kmeans_counts": counts.copy(),
        "kmeans_labels": labels.copy(),
        "cluster_edge_weight": str(cluster_edge_weight),
    })
    return data


def generate_latent_space_path(
    X,
    k=7,
    boundaries=[0, -1],
    version="1",
    reduction=1.0,
    random_neighbors=False,
    knn_cache_k=None,
    knn_expand_by=1,
    *,
    pathfinder="legacy",
    graph_mode="symmetric_knn",
    auto_expand_k=True,
    k_step=2,
    max_k=None,
    knn_algorithm="auto",
    density_weight_alpha=0.0,
    density_k=None,
    gamma_distance="euclidean",
    return_networkx_graph=False,
    path_simplify_tolerance=0.0,
    path_space="raw",
    start_index=None,
    end_index=None,
    kmeans_result=None,
    kmeans_n_clusters=None,
    kmeans_random_state=112358,
    kmeans_n_init=10,
    kmeans_algorithm="auto",
    kmeans_max_clusters=512,
    cluster_edge_weight="median_crossing",
):
    """
    Extended latent-space path generator.

    Existing behavior
    -----------------
    path_space="raw" (DEFAULT)
        Uses the original point cloud. pathfinder may be "legacy" or "dijkstra".

    New compressed path spaces
    --------------------------
    path_space="kmeans_centroids"
        K-means compresses X to centroid states. The selected endpoint POINTS are
        mapped to their clusters; those centroid states become the compressed
        source/sink. The chosen legacy or Dijkstra pathfinder then operates
        directly on the centroid cloud.

        This is the simplest interpretation of:
            raw latent cloud -> K centroids -> path over centroids

    path_space="kmeans_raw_topology"
        K-means still supplies the coarse states, but centroid adjacency is
        allowed only when raw kNN edges actually cross between the two clusters.
        Dijkstra then runs on this topology-preserving compressed graph.

    start_index/end_index
    ---------------------
    Optional actual-point endpoint indices. Defaults remain first/last point.
    For path_space="raw" and "kmeans_centroids", points/centroids are reordered
    internally so the historical pathfinders can retain their first/last
    endpoint convention.

    kmeans_result
    -------------
    A reusable result from cluster_latent_space_kmeans(). Supplying this avoids
    reclustering the same latent cloud across many comparison paths.
    """
    X = _as_2d_float_array(X)
    n = len(X)
    if n < 2:
        raise ValueError("At least two latent points are required.")

    start = 0 if start_index is None else int(start_index) % n
    end = n - 1 if end_index is None else int(end_index) % n
    if start == end:
        raise ValueError("start_index and end_index must differ.")

    space = str(path_space).lower()

    if space in {"raw", "points", "full"}:
        reordered, order = _endpoint_reorder(X, start, end)
        data = _generate_latent_space_path_pre_kmeans(
            reordered,
            k=k,
            boundaries=boundaries,
            version=version,
            reduction=reduction,
            random_neighbors=random_neighbors,
            knn_cache_k=knn_cache_k,
            knn_expand_by=knn_expand_by,
            pathfinder=pathfinder,
            graph_mode=graph_mode,
            auto_expand_k=auto_expand_k,
            k_step=k_step,
            max_k=max_k,
            knn_algorithm=knn_algorithm,
            density_weight_alpha=density_weight_alpha,
            density_k=density_k,
            gamma_distance=gamma_distance,
            return_networkx_graph=return_networkx_graph,
            path_simplify_tolerance=path_simplify_tolerance,
        )
        data["path_space"] = "raw"
        data["input_order"] = order
        data["start_index_original"] = int(start)
        data["end_index_original"] = int(end)

        if "path_indices" in data:
            p = np.asarray(data["path_indices"], dtype=int)
            data["path_indices_original"] = order[p].tolist()
        if "raw_path_indices" in data:
            p = np.asarray(data["raw_path_indices"], dtype=int)
            data["raw_path_indices_original"] = order[p].tolist()
        return data

    if kmeans_result is None:
        kmeans_result = cluster_latent_space_kmeans(
            X,
            n_clusters=kmeans_n_clusters,
            random_state=kmeans_random_state,
            n_init=kmeans_n_init,
            algorithm=kmeans_algorithm,
            max_clusters=kmeans_max_clusters,
        )

    labels, centroids, counts, representatives = _validate_kmeans_result(
        X,
        kmeans_result,
    )
    _ = representatives

    start_cluster = int(labels[start])
    end_cluster = int(labels[end])

    if start_cluster == end_cluster:
        raise RuntimeError(
            "Selected path endpoints map to the same K-means cluster. "
            "Increase kmeans_n_clusters or use another endpoint definition."
        )

    if space in {
        "kmeans_centroids",
        "centroids",
        "kmeans_centroid_knn",
        "kmeans",
    }:
        cluster_order = [start_cluster]
        cluster_order.extend(
            i for i in range(len(centroids))
            if i not in {start_cluster, end_cluster}
        )
        cluster_order.append(end_cluster)
        cluster_order = np.asarray(cluster_order, dtype=int)
        centroid_cloud = centroids[cluster_order]

        data = _generate_latent_space_path_pre_kmeans(
            centroid_cloud,
            k=min(int(k), max(1, len(centroid_cloud) - 1)),
            boundaries=boundaries,
            version=version,
            reduction=reduction,
            random_neighbors=random_neighbors,
            knn_cache_k=knn_cache_k,
            knn_expand_by=knn_expand_by,
            pathfinder=pathfinder,
            graph_mode=graph_mode,
            auto_expand_k=auto_expand_k,
            k_step=k_step,
            max_k=(
                None
                if max_k is None
                else min(int(max_k), max(1, len(centroid_cloud) - 1))
            ),
            knn_algorithm=knn_algorithm,
            density_weight_alpha=density_weight_alpha,
            density_k=(
                None
                if density_k is None
                else min(int(density_k), max(1, len(centroid_cloud) - 1))
            ),
            gamma_distance=gamma_distance,
            return_networkx_graph=return_networkx_graph,
            path_simplify_tolerance=path_simplify_tolerance,
        )

        data.update({
            "path_space": "kmeans_centroids",
            "kmeans_n_clusters": int(len(centroids)),
            "kmeans_start_cluster": int(start_cluster),
            "kmeans_end_cluster": int(end_cluster),
            "kmeans_cluster_order": cluster_order,
            "kmeans_counts": counts.copy(),
            "kmeans_labels": labels.copy(),
            "start_index_original": int(start),
            "end_index_original": int(end),
        })

        if "path_indices" in data:
            p = np.asarray(data["path_indices"], dtype=int)
            data["path_cluster_indices"] = cluster_order[p].tolist()
        if "raw_path_indices" in data:
            p = np.asarray(data["raw_path_indices"], dtype=int)
            data["raw_path_cluster_indices"] = cluster_order[p].tolist()
        return data

    if space in {
        "kmeans_raw_topology",
        "cluster_topology",
        "kmeans_topology",
    }:
        mode = str(pathfinder).lower()
        if mode not in {"dijkstra", "sparse_dijkstra", "shortest_path"}:
            raise ValueError(
                "path_space='kmeans_raw_topology' currently requires "
                "pathfinder='dijkstra'."
            )
        if boundaries != [0, -1] and tuple(boundaries) != (0, -1):
            raise ValueError(
                "Custom boundary averaging is not supported with "
                "kmeans_raw_topology."
            )
        return _generate_kmeans_raw_topology_path(
            X,
            start_index=start,
            end_index=end,
            k=k,
            version=version,
            reduction=reduction,
            graph_mode=graph_mode,
            auto_expand_k=auto_expand_k,
            k_step=k_step,
            max_k=max_k,
            knn_algorithm=knn_algorithm,
            density_weight_alpha=density_weight_alpha,
            density_k=density_k,
            gamma_distance=gamma_distance,
            return_networkx_graph=return_networkx_graph,
            path_simplify_tolerance=path_simplify_tolerance,
            kmeans_result=kmeans_result,
            kmeans_n_clusters=kmeans_n_clusters,
            kmeans_random_state=kmeans_random_state,
            kmeans_n_init=kmeans_n_init,
            kmeans_algorithm=kmeans_algorithm,
            kmeans_max_clusters=kmeans_max_clusters,
            cluster_edge_weight=cluster_edge_weight,
        )

    raise ValueError(
        "path_space must be 'raw', 'kmeans_centroids', or "
        "'kmeans_raw_topology'."
    )


# Public export list extension.
for _name in (
    "cluster_latent_space_kmeans",
    "find_latent_space_endpoints",
    "generate_latent_space_path",
):
    if _name not in __all__:
        __all__.append(_name)
