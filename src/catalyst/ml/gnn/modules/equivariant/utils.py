"""
Equivariant geometry utilities for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/equivariant/utils.py

These utilities operate on PyG Data/Batch objects carrying the standardized
Catalyst equivariant graph fields:

    z, pos, edge_index, cell, pbc, shifts, batch

The key idea is that edge geometry should usually be recomputed inside the
forward pass from pos/cell/shifts, rather than relying on detached precomputed
edge_vec/edge_dist. This is required for conservative energy-gradient forces.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def validate_equivariant_data(data, *, require_batch: bool = True) -> None:
    """
    Validate that a Data/Batch object has the fields required by an
    equivariant processor.
    """
    required = ["z", "pos", "edge_index", "cell", "pbc", "shifts"]

    if require_batch:
        required.append("batch")

    missing = [name for name in required if not has_attr(data, name)]
    if missing:
        raise AttributeError(
            "Equivariant graph data is missing required fields: "
            + ", ".join(missing)
        )

    if data.pos.dim() != 2 or data.pos.size(-1) != 3:
        raise ValueError(f"data.pos must have shape [N, 3], got {tuple(data.pos.shape)}")

    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError(
            f"data.edge_index must have shape [2, E], got {tuple(data.edge_index.shape)}"
        )

    if data.shifts.dim() != 2 or data.shifts.size(-1) != 3:
        raise ValueError(
            f"data.shifts must have shape [E, 3], got {tuple(data.shifts.shape)}"
        )

    if data.shifts.size(0) != data.edge_index.size(1):
        raise ValueError(
            "data.shifts and data.edge_index disagree on number of edges: "
            f"{data.shifts.size(0)} versus {data.edge_index.size(1)}"
        )

    if data.cell.dim() != 3 or data.cell.size(-2) != 3 or data.cell.size(-1) != 3:
        raise ValueError(
            f"data.cell must have shape [B, 3, 3], got {tuple(data.cell.shape)}"
        )

    if data.pbc.dim() != 2 or data.pbc.size(-1) != 3:
        raise ValueError(f"data.pbc must have shape [B, 3], got {tuple(data.pbc.shape)}")

    if require_batch:
        if data.batch.dim() != 1:
            raise ValueError(f"data.batch must have shape [N], got {tuple(data.batch.shape)}")
        if data.batch.size(0) != data.pos.size(0):
            raise ValueError(
                "data.batch and data.pos disagree on number of nodes: "
                f"{data.batch.size(0)} versus {data.pos.size(0)}"
            )


def ensure_batch(data):
    """
    Ensure data.batch exists.

    For a single unbatched graph, PyG may not define data.batch. In that case,
    create a zero batch vector.
    """
    if not has_attr(data, "batch"):
        data.batch = torch.zeros(
            data.pos.size(0),
            dtype=torch.long,
            device=data.pos.device,
        )

    return data


def get_edge_batch(data) -> torch.Tensor:
    """
    Return graph id for each edge.

    Uses the source node batch assignment.
    """
    data = ensure_batch(data)
    src = data.edge_index[0]
    return data.batch[src]


def compute_shift_vectors(
    shifts: torch.Tensor,
    cell_per_edge: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert integer periodic image shifts to Cartesian vectors.

    Parameters
    ----------
    shifts
        Integer shift tensor with shape [E, 3].
    cell_per_edge
        Cell tensor with shape [E, 3, 3].

    Returns
    -------
    shift_vec
        Cartesian shift vectors with shape [E, 3].
    """
    if dtype is None:
        dtype = cell_per_edge.dtype

    shifts = shifts.to(device=cell_per_edge.device, dtype=dtype)
    return torch.einsum("ei,eij->ej", shifts, cell_per_edge)


def get_edge_geometry(
    data,
    *,
    eps: float = 1.0e-9,
    validate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute differentiable edge geometry from positions, cell, and shifts.

    Convention
    ----------
    For edge i -> j:

        edge_vec = pos[j] + shifts @ cell - pos[i]

    Returns
    -------
    edge_vec
        Edge vectors with shape [E, 3].
    edge_dist
        Edge distances with shape [E, 1].
    edge_unit
        Unit edge vectors with shape [E, 3].
    """
    data = ensure_batch(data)

    if validate:
        validate_equivariant_data(data, require_batch=True)

    src, dst = data.edge_index
    edge_batch = get_edge_batch(data)

    cell_per_edge = data.cell[edge_batch]
    shift_vec = compute_shift_vectors(
        data.shifts,
        cell_per_edge,
        dtype=data.pos.dtype,
    )

    edge_vec = data.pos[dst] + shift_vec - data.pos[src]
    edge_dist = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
    edge_unit = edge_vec / edge_dist.clamp_min(eps)

    return edge_vec, edge_dist, edge_unit


def get_precomputed_edge_geometry(data):
    """
    Return precomputed edge geometry if present.

    This is useful for debugging or direct-force models. For conservative
    energy-gradient force models, prefer get_edge_geometry(data).
    """
    if not has_attr(data, "edge_vec") or not has_attr(data, "edge_dist"):
        raise AttributeError("data.edge_vec and/or data.edge_dist are missing.")

    edge_vec = data.edge_vec
    edge_dist = data.edge_dist

    if edge_dist.dim() == 1:
        edge_dist = edge_dist.view(-1, 1)

    edge_unit = edge_vec / edge_dist.clamp_min(1.0e-9)
    return edge_vec, edge_dist, edge_unit


def enable_pos_grad(data):
    """
    Enable gradients with respect to positions.

    Needed for energy-gradient force training/inference.
    """
    data.pos = data.pos.detach().clone().requires_grad_(True)
    return data


def compute_forces_from_energy(
    energy: torch.Tensor,
    pos: torch.Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    Compute conservative forces from an energy prediction.

    Parameters
    ----------
    energy
        Graph energy tensor. Can be shape [B], [B, 1], or scalar.
    pos
        Position tensor with requires_grad=True.

    Returns
    -------
    forces
        Force tensor with shape [N, 3].
    """
    grad = torch.autograd.grad(
        energy.sum(),
        pos,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]

    return -grad


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Minimal scatter-sum helper without requiring torch_scatter.

    Parameters
    ----------
    src
        Source tensor with shape [N, F] or [N].
    index
        Destination indices with shape [N].
    dim_size
        Number of output groups.

    Returns
    -------
    out
        Summed tensor with shape [dim_size, ...].
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(out_shape)
    out.index_add_(0, index, src)
    return out


def graph_sum(
    node_values: torch.Tensor,
    batch: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Sum node values into graph-level values using data.batch.
    """
    return scatter_sum(node_values, batch, dim_size=dim_size)


def summarize_equivariant_data(data) -> Dict[str, object]:
    """
    Return a compact shape summary for debugging.
    """
    summary = {}

    for key in [
        "z",
        "pos",
        "edge_index",
        "edge_vec",
        "edge_dist",
        "cell",
        "pbc",
        "shifts",
        "batch",
        "ptr",
    ]:
        value = getattr(data, key, None)
        summary[key] = None if value is None else tuple(value.shape)

    summary["num_nodes"] = getattr(data, "num_nodes", None)
    summary["num_edges"] = (
        int(data.edge_index.size(1))
        if has_attr(data, "edge_index")
        else None
    )

    return summary