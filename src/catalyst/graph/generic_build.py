from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..properties.structure_properties import get_3body_angle
from .graph import Generic_Graph_Data
from .graph import line_graph


ArrayLike = Any
FeatureFn = Callable[[ArrayLike, np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
# Public dispatcher
# =============================================================================


def check_generic_params(target_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize generic graph-generation parameters.

    This mirrors the style of ``alignnd.check_params``: callers can provide only
    the fields they care about, and the dispatcher fills in safe defaults.

    Supported types
    ---------------
    generic_pairwise
        Builds a normal generic graph from a kNN/cutoff neighbor table.

    generic_pairwise_atomic
        Builds a center-node star graph from one local neighborhood.
    """
    source_dict: Dict[str, Any] = {
        "type": None,
        "raw_data": None,
        "params": {},
        "line_graph": True,
        "include_angs": True,
        "include_self_edges": False,
        "source_from_neighbor_table": True,
        "directed": True,
        "dtype": np.float32,
        "body_order": 3,
        "max_body_order": 3,
        "feature_fns": None,
        "strict": True,
        "include_equivariant_fields": True,
    }

    updated = dict(target_dict)
    for key, value in source_dict.items():
        updated.setdefault(key, value)
    return updated


def generic_graph_gen(data: Mapping[str, Any]):
    """Dispatch generic graph generation by ``data['type']``.

    This is the generic-graph analogue of ``alignn_gen``.
    """
    data = check_generic_params(data)
    graph_type = data["type"]

    if graph_type == "generic_pairwise":
        return generic_pairwise(
            data=data["raw_data"],
            data_params=data["params"],
            gen_line_graph=bool(data["line_graph"] and data["include_angs"]),
            include_self_edges=bool(data["include_self_edges"]),
            source_from_neighbor_table=bool(data["source_from_neighbor_table"]),
            dtype=data["dtype"],
            strict=bool(data["strict"]),
            include_equivariant_fields=bool(data["include_equivariant_fields"]),
        )

    if graph_type == "generic_pairwise_atomic":
        return generic_pairwise_atomic(
            data=data["raw_data"],
            data_params=data["params"],
            include_self_edges=bool(data["include_self_edges"]),
            dtype=data["dtype"],
            strict=bool(data["strict"]),
            include_equivariant_fields=bool(data["include_equivariant_fields"]),
        )



    raise ValueError(
        f"Unsupported generic graph type {graph_type!r}. Supported types are "
        "'generic_pairwise' and 'generic_pairwise_atomic'."
    )


# =============================================================================
# Shared utilities
# =============================================================================


def _normalize_edge_index(edge_index: Any, dtype=np.int64) -> np.ndarray:
    """Return edge index with canonical shape ``(2, n_edges)``."""
    edge_index = np.asarray(edge_index, dtype=dtype)

    if edge_index.size == 0:
        return np.empty((2, 0), dtype=dtype)

    if edge_index.ndim == 1:
        if edge_index.size != 2:
            raise ValueError(
                f"A one-dimensional edge index must contain exactly two values; "
                f"received shape {edge_index.shape}."
            )
        return edge_index.reshape(2, 1)

    if edge_index.ndim != 2:
        raise ValueError(
            f"An edge index must be two-dimensional; received shape {edge_index.shape}."
        )

    if edge_index.shape[0] == 2:
        return edge_index

    if edge_index.shape[1] == 2:
        return edge_index.T

    raise ValueError(
        f"An edge index must have shape (2, n_edges) or (n_edges, 2); "
        f"received shape {edge_index.shape}."
    )


def _as_float_array(values: Any, dtype=np.float32) -> np.ndarray:
    """Convert values to a flat NumPy float array."""
    if values is None:
        return np.empty((0,), dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _as_node_features(values: Any, dtype=np.float32) -> np.ndarray:
    """Convert node features to a 2D NumPy array."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Node features must be 1D or 2D; received shape {arr.shape}.")
    return arr


def _amount_tensor(length: int) -> torch.Tensor:
    """Compatibility amount tensor used by Generic_Graph_Data.

    The legacy code stored ``len(values) - 1``.  Keep that convention, while
    protecting empty arrays from becoming negative in a surprising way.
    """
    return torch.tensor(max(int(length) - 1, 0), dtype=torch.long)


def _maybe_tensor(value: Any, dtype=None):
    """Convert optional user-supplied graph fields into tensors."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)




def _looks_like_positions(value: Any, n_nodes: int) -> bool:
    """Return True if value can safely be interpreted as [n_nodes, 3] positions."""
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.ndim == 2 and arr.shape[0] == int(n_nodes) and arr.shape[1] == 3


def _as_cell_tensor(value: Any, dtype=torch.float) -> torch.Tensor:
    """Return a single-graph cell tensor with shape [3, 3]."""
    if value is None:
        return torch.eye(3, dtype=dtype)

    tensor = _maybe_tensor(value, dtype=dtype)

    if tensor.dim() == 2 and tuple(tensor.shape) == (3, 3):
        return tensor

    if tensor.dim() == 3 and tensor.size(0) == 1 and tuple(tensor.shape[-2:]) == (3, 3):
        return tensor[0]

    raise ValueError(
        "Generic graph cell must have shape [3, 3] for a single graph "
        f"or [1, 3, 3]; received {tuple(tensor.shape)}."
    )


def _as_pbc_tensor(value: Any) -> torch.Tensor:
    """Return a single-graph PBC tensor with shape [3]."""
    if value is None:
        return torch.zeros(3, dtype=torch.bool)

    tensor = _maybe_tensor(value, dtype=torch.bool)

    if tensor.dim() == 1 and tensor.numel() == 3:
        return tensor.reshape(3)

    if tensor.dim() == 2 and tensor.size(0) == 1 and tensor.size(1) == 3:
        return tensor[0]

    raise ValueError(
        "Generic graph pbc must have shape [3] for a single graph "
        f"or [1, 3]; received {tuple(tensor.shape)}."
    )


def _finalize_generic_graph_metadata(
    graph: Generic_Graph_Data,
    *,
    data_params: Optional[Mapping[str, Any]] = None,
    raw_data: Any = None,
    edge_index_G: Optional[np.ndarray] = None,
    dtype=np.float32,
    include_equivariant_fields: bool = True,
) -> Generic_Graph_Data:
    """Finalize PyG/equivariant metadata for a Generic_Graph_Data object.

    This makes the generic graph builder consistent with the updated graph.py
    batching contract.

    It explicitly sets:
        num_nodes
        edge_index
        cell
        pbc
        shifts

    when enough equivariant information is present. It also optionally derives
    pos from raw_data when raw_data has shape [N, 3], which is useful for generic
    coordinate graphs built from kNN tables.
    """
    data_params = dict(data_params or {})

    # ------------------------------------------------------------------
    # 1. Always make PyG node count explicit.
    # ------------------------------------------------------------------
    if getattr(graph, "z", None) is not None:
        graph.num_nodes = int(graph.z.size(0))
    elif getattr(graph, "pos", None) is not None:
        graph.num_nodes = int(graph.pos.size(0))
    elif getattr(graph, "node_G", None) is not None:
        graph.num_nodes = int(graph.node_G.size(0))

    n_nodes = int(getattr(graph, "num_nodes", 0))

    # ------------------------------------------------------------------
    # 2. Ensure PyG-standard edge_index exists.
    # ------------------------------------------------------------------
    if getattr(graph, "edge_index", None) is None:
        if "edge_index" in data_params and data_params["edge_index"] is not None:
            graph.edge_index = _maybe_tensor(data_params["edge_index"], dtype=torch.long)
        elif edge_index_G is not None:
            graph.edge_index = torch.tensor(
                _normalize_edge_index(edge_index_G, dtype=np.int64),
                dtype=torch.long,
            )
        elif getattr(graph, "edge_index_G", None) is not None:
            graph.edge_index = graph.edge_index_G

    if getattr(graph, "edge_index", None) is not None:
        if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
            raise ValueError(
                "Generic graph edge_index must have shape [2, E], "
                f"got {tuple(graph.edge_index.shape)}."
            )

    # ------------------------------------------------------------------
    # 3. Optionally infer pos from raw_data for coordinate-like generic graphs.
    # ------------------------------------------------------------------
    if include_equivariant_fields and getattr(graph, "pos", None) is None:
        use_raw_data_as_pos = bool(data_params.get("use_raw_data_as_pos", True))
        if use_raw_data_as_pos and _looks_like_positions(raw_data, n_nodes):
            graph.pos = torch.as_tensor(np.asarray(raw_data, dtype=dtype), dtype=torch.float)

    # ------------------------------------------------------------------
    # 4. If z is absent but pos exists, create a generic single-species z.
    #    This gives equivariant encoders something stable to embed.
    # ------------------------------------------------------------------
    if include_equivariant_fields and getattr(graph, "pos", None) is not None:
        if getattr(graph, "z", None) is None:
            graph.z = torch.ones(n_nodes, dtype=torch.long, device=graph.pos.device)

        if getattr(graph, "global_atom_indices", None) is None:
            graph.global_atom_indices = torch.arange(
                n_nodes,
                dtype=torch.long,
                device=graph.pos.device,
            )

    # ------------------------------------------------------------------
    # 5. If pos exists, make the single-graph PBC fields explicit.
    # ------------------------------------------------------------------
    if include_equivariant_fields and getattr(graph, "pos", None) is not None:
        graph.cell = _as_cell_tensor(getattr(graph, "cell", None), dtype=torch.float)
        graph.pbc = _as_pbc_tensor(getattr(graph, "pbc", None))

        n_edges = int(graph.edge_index.size(1)) if getattr(graph, "edge_index", None) is not None else 0

        if getattr(graph, "shifts", None) is None:
            graph.shifts = torch.zeros(
                (n_edges, 3),
                dtype=torch.long,
                device=graph.pos.device,
            )
        else:
            graph.shifts = _maybe_tensor(graph.shifts, dtype=torch.long)
            if graph.shifts.dim() != 2 or graph.shifts.size(1) != 3:
                raise ValueError(
                    "Generic graph shifts must have shape [E, 3], "
                    f"got {tuple(graph.shifts.shape)}."
                )
            if graph.shifts.size(0) != n_edges:
                raise ValueError(
                    "Generic graph shifts length does not match edge_index: "
                    f"{graph.shifts.size(0)} versus {n_edges}."
                )

        # If precomputed edge geometry is absent, build nonperiodic/local
        # geometry from pos and shifts. For conservative force models, the
        # processor should still recompute geometry inside forward().
        if n_edges > 0 and getattr(graph, "edge_vec", None) is None:
            src, dst = graph.edge_index
            cell_per_edge = graph.cell.unsqueeze(0).expand(n_edges, -1, -1)
            shift_vec = torch.einsum(
                "ei,eij->ej",
                graph.shifts.to(dtype=graph.pos.dtype, device=graph.pos.device),
                cell_per_edge.to(dtype=graph.pos.dtype, device=graph.pos.device),
            )
            graph.edge_vec = graph.pos[dst] + shift_vec - graph.pos[src]

        if n_edges > 0 and getattr(graph, "edge_dist", None) is None and getattr(graph, "edge_vec", None) is not None:
            graph.edge_dist = torch.linalg.norm(graph.edge_vec, dim=-1)

        if getattr(graph, "edge_vec", None) is not None and graph.edge_vec.size(0) != n_edges:
            raise ValueError(
                "Generic graph edge_vec length does not match edge_index: "
                f"{graph.edge_vec.size(0)} versus {n_edges}."
            )

        if getattr(graph, "edge_dist", None) is not None:
            graph.edge_dist = graph.edge_dist.reshape(-1)
            if graph.edge_dist.size(0) != n_edges:
                raise ValueError(
                    "Generic graph edge_dist length does not match edge_index: "
                    f"{graph.edge_dist.size(0)} versus {n_edges}."
                )

    return graph

def _attach_optional_equivariant_fields(
    graph: Generic_Graph_Data,
    data_params: Mapping[str, Any],
    edge_index_G: np.ndarray,
    *,
    dtype=np.float32,
    include_equivariant_fields: bool = True,
) -> Generic_Graph_Data:
    """Attach optional standardized equivariant fields to generic graphs.

    Generic graphs are not necessarily atomistic, so this helper does not try to
    infer MIC vectors or periodic shifts.  It simply passes through fields when
    the caller provides them in ``data_params``:

        z, pos, edge_index, edge_vec, edge_dist, cell, pbc, shifts

    If ``edge_index`` is not supplied but position-like fields are present, the
    primary graph connectivity ``edge_index_G`` is used.
    """
    if not include_equivariant_fields:
        return graph

    present = {key for key in (
        "z", "pos", "edge_index", "edge_vec", "edge_dist",
        "cell", "pbc", "shifts", "global_atom_indices",
    ) if key in data_params and data_params[key] is not None}

    if not present:
        return graph

    if "z" in data_params:
        graph.z = _maybe_tensor(data_params.get("z"), dtype=torch.long)

    if "pos" in data_params:
        graph.pos = _maybe_tensor(data_params.get("pos"), dtype=torch.float)

    if "edge_index" in data_params:
        edge_index = _normalize_edge_index(data_params.get("edge_index"), dtype=np.int64)
        graph.edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        graph.edge_index = torch.tensor(
            _normalize_edge_index(edge_index_G, dtype=np.int64),
            dtype=torch.long,
        )

    if "edge_vec" in data_params:
        graph.edge_vec = _maybe_tensor(data_params.get("edge_vec"), dtype=torch.float)

    if "edge_dist" in data_params:
        graph.edge_dist = _maybe_tensor(data_params.get("edge_dist"), dtype=torch.float)
    elif getattr(graph, "node_A", None) is not None:
        # node_A is the edge scalar produced after self-edge filtering. This is
        # safer than raw data_params["dist"], whose shape may include self entries.
        try:
            if graph.node_A.numel() == graph.edge_index.size(1):
                graph.edge_dist = graph.node_A.reshape(-1)
        except Exception:
            pass

    if "cell" in data_params:
        graph.cell = _maybe_tensor(data_params.get("cell"), dtype=torch.float)

    if "pbc" in data_params:
        graph.pbc = _maybe_tensor(data_params.get("pbc"), dtype=torch.bool)

    if "shifts" in data_params:
        graph.shifts = _maybe_tensor(data_params.get("shifts"), dtype=torch.long)

    if "global_atom_indices" in data_params:
        graph.global_atom_indices = _maybe_tensor(
            data_params.get("global_atom_indices"),
            dtype=torch.long,
        )

    return graph


def _empty_generic_graph(g_nodes: np.ndarray, dtype=np.float32) -> Generic_Graph_Data:
    graph = Generic_Graph_Data(
        node_G=torch.tensor(g_nodes, dtype=torch.float),
        edge_index_G=torch.empty((2, 0), dtype=torch.long),
        node_A=torch.empty((0,), dtype=torch.float),
        edge_index_A=None,
        edge_A=None,
        node_G_amounts=_amount_tensor(len(g_nodes)),
        node_A_amounts=_amount_tensor(0),
        edge_A_amounts=None,
    )
    graph = _finalize_generic_graph_metadata(
        graph,
        data_params=None,
        raw_data=None,
        edge_index_G=np.empty((2, 0), dtype=np.int64),
        dtype=dtype,
        include_equivariant_fields=False,
    )
    graph.generate_gid()
    return graph


def _build_pair_edges_from_neighbor_table(
    ind: Any,
    dist: Any,
    *,
    include_self_edges: bool = False,
    source_from_neighbor_table: bool = True,
    dtype=np.float32,
    strict: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build directed pair edges from a neighbor table.

    Parameters
    ----------
    ind, dist
        Arrays like those returned by ``sklearn.neighbors.KDTree.query`` with
        shape ``(n_nodes, n_neighbors)``.
    include_self_edges
        Keep self edges if present.  The legacy generic graph skipped column 0,
        assuming it was the self-neighbor.  This implementation instead removes
        self edges explicitly unless requested.
    source_from_neighbor_table
        If True, use ``ind[row, 0]`` as the source node.  This preserves the
        legacy convention.  If False, use the row index as the source node.
    """
    ind_arr = np.asarray(ind, dtype=np.int64)
    dist_arr = np.asarray(dist, dtype=dtype)

    if ind_arr.ndim == 1:
        ind_arr = ind_arr.reshape(1, -1)
    if dist_arr.ndim == 1:
        dist_arr = dist_arr.reshape(1, -1)

    if ind_arr.shape != dist_arr.shape:
        raise ValueError(
            "Neighbor index and distance arrays must have the same shape; "
            f"received {ind_arr.shape} and {dist_arr.shape}."
        )

    src_list = []
    dst_list = []
    edge_values = []

    for row_idx in range(ind_arr.shape[0]):
        source = int(ind_arr[row_idx, 0]) if source_from_neighbor_table else int(row_idx)

        for col_idx in range(ind_arr.shape[1]):
            target = int(ind_arr[row_idx, col_idx])
            value = float(dist_arr[row_idx, col_idx])

            if not np.isfinite(value):
                continue
            if not include_self_edges and source == target:
                continue
            if target < 0:
                if strict:
                    raise ValueError(f"Negative neighbor index found: {target}.")
                continue

            src_list.append(source)
            dst_list.append(target)
            edge_values.append(value)

    if not src_list:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=dtype),
        )

    return (
        np.asarray([src_list, dst_list], dtype=np.int64),
        np.asarray(edge_values, dtype=dtype),
    )


def _safe_line_graph(edge_index: np.ndarray) -> np.ndarray:
    return _normalize_edge_index(line_graph(_normalize_edge_index(edge_index)), dtype=np.int64)


def _safe_get_3body_angle(
    data: Any,
    edge_index_G: np.ndarray,
    edge_index_A: np.ndarray,
    *,
    dtype=np.float32,
) -> np.ndarray:
    edge_index_A = _normalize_edge_index(edge_index_A, dtype=np.int64)
    if edge_index_A.shape[1] == 0:
        return np.empty((0,), dtype=dtype)
    values = get_3body_angle(data, edge_index_G, edge_index_A)
    return np.asarray(values, dtype=dtype).reshape(-1)


# =============================================================================
# Current Generic_Graph_Data-compatible builders
# =============================================================================


def generic_pairwise(
    data: Any,
    data_params: Mapping[str, Any],
    gen_line_graph: bool = True,
    *,
    include_self_edges: bool = False,
    source_from_neighbor_table: bool = True,
    dtype=np.float32,
    strict: bool = True,
    include_equivariant_fields: bool = True,
) -> Generic_Graph_Data:
    """Build a generic pairwise graph, optionally with 3-body angular edges.

    Expected params
    ---------------
    ind : array, shape (n_nodes, n_neighbors)
        Neighbor indices.
    dist : array, shape (n_nodes, n_neighbors)
        Neighbor distances/features corresponding to ``ind``.
    g_nodes : array, shape (n_nodes, n_features)
        Global graph node features.
    """
    required = ("ind", "dist", "g_nodes")
    missing = [key for key in required if key not in data_params]
    if missing:
        raise KeyError(f"generic_pairwise missing required params: {missing}")

    g_nodes = _as_node_features(data_params["g_nodes"], dtype=dtype)
    edge_index_G, node_A = _build_pair_edges_from_neighbor_table(
        data_params["ind"],
        data_params["dist"],
        include_self_edges=include_self_edges,
        source_from_neighbor_table=source_from_neighbor_table,
        dtype=dtype,
        strict=strict,
    )

    if edge_index_G.shape[1] == 0:
        if strict:
            raise ValueError("generic_pairwise produced no graph edges.")
        return _empty_generic_graph(g_nodes, dtype=dtype)

    if gen_line_graph:
        edge_index_A = _safe_line_graph(edge_index_G)
        edge_A = _safe_get_3body_angle(data, edge_index_G, edge_index_A, dtype=dtype)

        graph = Generic_Graph_Data(
            node_G=torch.tensor(g_nodes, dtype=torch.float),
            edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
            node_A=torch.tensor(node_A, dtype=torch.float),
            edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
            edge_A=torch.tensor(edge_A, dtype=torch.float),
            node_G_amounts=_amount_tensor(len(g_nodes)),
            node_A_amounts=_amount_tensor(len(node_A)),
            edge_A_amounts=_amount_tensor(len(edge_A)),
        )
    else:
        graph = Generic_Graph_Data(
            node_G=torch.tensor(g_nodes, dtype=torch.float),
            edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
            node_A=torch.tensor(node_A, dtype=torch.float),
            edge_index_A=None,
            edge_A=None,
            node_G_amounts=_amount_tensor(len(g_nodes)),
            node_A_amounts=_amount_tensor(len(node_A)),
            edge_A_amounts=None,
        )

    graph = _attach_optional_equivariant_fields(
        graph,
        data_params,
        edge_index_G,
        dtype=dtype,
        include_equivariant_fields=include_equivariant_fields,
    )
    graph = _finalize_generic_graph_metadata(
        graph,
        data_params=data_params,
        raw_data=data,
        edge_index_G=edge_index_G,
        dtype=dtype,
        include_equivariant_fields=include_equivariant_fields,
    )
    graph.generate_gid()
    return graph


def generic_pairwise_atomic(
    data: Any,
    data_params: Mapping[str, Any],
    *,
    include_self_edges: bool = False,
    dtype=np.float32,
    strict: bool = True,
    include_equivariant_fields: bool = True,
) -> Optional[Generic_Graph_Data]:
    """Build the legacy center-node local generic graph.

    This preserves the old behavior: node 0 is a dummy/center node and each
    supplied local neighbor becomes node ``i + 1``.  Edges are bidirectional
    center-neighbor links and ``node_A`` stores the corresponding distances.
    """
    del data  # Retained for dispatcher/API symmetry.

    required = ("ind", "dist", "g_nodes", "all_labels")
    missing = [key for key in required if key not in data_params]
    if missing:
        raise KeyError(f"generic_pairwise_atomic missing required params: {missing}")

    dist = _as_float_array(data_params["dist"], dtype=dtype)
    g_nodes_in = list(data_params["g_nodes"])
    all_labels = list(data_params["all_labels"])

    if len(dist) == 0:
        if strict:
            raise ValueError("generic_pairwise_atomic received no distances/neighbors.")
        return None

    if len(g_nodes_in) != len(dist):
        raise ValueError(
            "generic_pairwise_atomic expects one node label per distance; "
            f"received {len(g_nodes_in)} labels and {len(dist)} distances."
        )

    symbol_to_id = {symbol: idx + 1 for idx, symbol in enumerate(all_labels)}

    # Dummy/center label followed by one-hot labels for local neighbors.
    g_nodes = [np.eye(len(all_labels) + 1, dtype=dtype)[0]]
    for node_label in g_nodes_in:
        if node_label not in symbol_to_id:
            raise ValueError(
                f"Unknown node label {node_label!r}; available labels are {all_labels}."
            )
        g_nodes.append(np.eye(len(all_labels) + 1, dtype=dtype)[symbol_to_id[node_label]])
    g_nodes_arr = np.asarray(g_nodes, dtype=dtype)

    src = []
    dst = []
    node_A = []

    for local_idx, value in enumerate(dist):
        neighbor_id = local_idx + 1
        if include_self_edges or neighbor_id != 0:
            src.append(0)
            dst.append(neighbor_id)
            node_A.append(float(value))

    for local_idx, value in enumerate(dist):
        neighbor_id = local_idx + 1
        if include_self_edges or neighbor_id != 0:
            src.append(neighbor_id)
            dst.append(0)
            node_A.append(float(value))

    edge_index_G = np.asarray([src, dst], dtype=np.int64)
    node_A_arr = np.asarray(node_A, dtype=dtype)

    graph = Generic_Graph_Data(
        node_G=torch.tensor(g_nodes_arr, dtype=torch.float),
        edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
        node_A=torch.tensor(node_A_arr, dtype=torch.float),
        node_G_amounts=_amount_tensor(len(g_nodes_arr)),
        node_A_amounts=_amount_tensor(len(node_A_arr)),
    )

    graph = _attach_optional_equivariant_fields(
        graph,
        data_params,
        edge_index_G,
        dtype=dtype,
        include_equivariant_fields=include_equivariant_fields,
    )
    graph = _finalize_generic_graph_metadata(
        graph,
        data_params=data_params,
        raw_data=data,
        edge_index_G=edge_index_G,
        dtype=dtype,
        include_equivariant_fields=include_equivariant_fields,
    )
    graph.generate_gid()
    return graph
