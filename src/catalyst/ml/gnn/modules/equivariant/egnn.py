"""
Minimal but production-usable EGNN layers for Catalyst.

Recommended location:
    catalyst/ml/gnn/modules/equivariant/egnn.py

This implementation consumes the standardized Catalyst equivariant graph fields:

    pos, edge_index, cell, shifts, batch

and hidden scalar features:

    h_scalar

It recomputes edge geometry from pos/cell/shifts inside forward(), which is the
safe path for energy-gradient force models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter-sum along the first dimension using torch.index_add_."""
    if src.numel() == 0:
        if dim_size is None:
            dim_size = 0
        return src.new_zeros((dim_size,) + tuple(src.shape[1:]))

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def ensure_batch(data):
    """Ensure data.batch exists for single-graph and batched inputs."""
    if not has_attr(data, "batch"):
        data.batch = torch.zeros(
            data.pos.size(0),
            dtype=torch.long,
            device=data.pos.device,
        )
    return data


def get_edge_batch(data) -> torch.Tensor:
    """Return graph id for each edge using source-node batch assignment."""
    data = ensure_batch(data)
    src = data.edge_index[0]
    return data.batch[src]


def _cell_per_edge(data, edge_batch: torch.Tensor) -> torch.Tensor:
    """Return cell tensor with shape [E, 3, 3]."""
    n_edges = int(edge_batch.numel())

    if not has_attr(data, "cell"):
        eye = torch.eye(3, dtype=data.pos.dtype, device=data.pos.device)
        return eye.unsqueeze(0).expand(n_edges, -1, -1)

    cell = data.cell.to(device=data.pos.device, dtype=data.pos.dtype)

    if cell.dim() == 2:
        if tuple(cell.shape) != (3, 3):
            raise ValueError(f"data.cell must be [3, 3] or [B, 3, 3], got {tuple(cell.shape)}.")
        return cell.unsqueeze(0).expand(n_edges, -1, -1)

    if cell.dim() == 3:
        if tuple(cell.shape[-2:]) != (3, 3):
            raise ValueError(f"data.cell must end with [3, 3], got {tuple(cell.shape)}.")
        return cell[edge_batch]

    raise ValueError(f"data.cell must be [3, 3] or [B, 3, 3], got {tuple(cell.shape)}.")


def get_edge_geometry(
    data,
    *,
    pos: Optional[torch.Tensor] = None,
    eps: float = 1.0e-9,
    use_precomputed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable edge geometry.

    Convention for edge i -> j:

        edge_vec = pos[j] + shifts @ cell - pos[i]

    Returns
    -------
    edge_vec
        Tensor with shape [E, 3].
    edge_dist
        Tensor with shape [E, 1].
    edge_unit
        Tensor with shape [E, 3].
    """
    if pos is None:
        pos = data.pos

    if use_precomputed and has_attr(data, "edge_vec") and has_attr(data, "edge_dist"):
        edge_vec = data.edge_vec.to(device=pos.device, dtype=pos.dtype)
        edge_dist = data.edge_dist.to(device=pos.device, dtype=pos.dtype)
        if edge_dist.dim() == 1:
            edge_dist = edge_dist.view(-1, 1)
        edge_unit = edge_vec / edge_dist.clamp_min(eps)
        return edge_vec, edge_dist, edge_unit

    data = ensure_batch(data)
    src, dst = data.edge_index
    edge_batch = get_edge_batch(data)

    if has_attr(data, "shifts"):
        shifts = data.shifts.to(device=pos.device, dtype=pos.dtype)
    else:
        shifts = torch.zeros((data.edge_index.size(1), 3), dtype=pos.dtype, device=pos.device)

    cell = _cell_per_edge(data, edge_batch)
    shift_vec = torch.einsum("ei,eij->ej", shifts, cell)

    edge_vec = pos[dst] + shift_vec - pos[src]
    edge_dist = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
    edge_unit = edge_vec / edge_dist.clamp_min(eps)

    return edge_vec, edge_dist, edge_unit


class MLP(nn.Module):
    """Small configurable MLP used by EGNN blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        act: Optional[nn.Module] = None,
        dropout: float = 0.0,
        final_act: bool = False,
    ):
        super().__init__()

        if act is None:
            act = nn.SiLU()

        layers = [
            nn.Linear(int(in_dim), int(hidden_dim)),
            act,
        ]

        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))

        layers.append(nn.Linear(int(hidden_dim), int(out_dim)))

        if final_act:
            layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianRadialBasis(nn.Module):
    """Gaussian radial basis expansion for edge distances."""

    def __init__(
        self,
        num_basis: int = 32,
        cutoff: Optional[float] = None,
        *,
        r_min: float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()

        self.num_basis = int(num_basis)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.r_min = float(r_min)

        if self.num_basis < 1:
            raise ValueError("num_basis must be >= 1.")

        r_max = self.cutoff if self.cutoff is not None else 6.0
        centers = torch.linspace(self.r_min, r_max, self.num_basis)
        if self.num_basis > 1:
            spacing = float(centers[1] - centers[0])
        else:
            spacing = max(r_max - self.r_min, 1.0)
        gamma = torch.tensor(1.0 / max(spacing, 1.0e-6) ** 2)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("gamma", gamma)

    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        if edge_dist.dim() == 1:
            edge_dist = edge_dist.view(-1, 1)

        diff = edge_dist - self.centers.view(1, -1).to(device=edge_dist.device, dtype=edge_dist.dtype)
        gamma = self.gamma.to(device=edge_dist.device, dtype=edge_dist.dtype)
        return torch.exp(-gamma * diff.pow(2))


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: Optional[float]):
        super().__init__()
        self.cutoff = None if cutoff is None else float(cutoff)

    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        if self.cutoff is None:
            return torch.ones_like(edge_dist)

        x = edge_dist / self.cutoff
        envelope = 0.5 * (torch.cos(torch.pi * x.clamp(min=0.0, max=1.0)) + 1.0)
        return envelope * (x <= 1.0).to(edge_dist.dtype)


class EGNNLayer(nn.Module):
    """One EGNN message-passing layer.

    The default mode does not update physical positions.  It updates scalar
    hidden states using distance/geometric messages, which is appropriate for
    invariant energy models whose forces are obtained by energy gradients.

    Set update_pos=True only if you explicitly want coordinate updates.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: Optional[int] = None,
        rbf_dim: int = 32,
        edge_attr_dim: int = 0,
        act: Optional[nn.Module] = None,
        dropout: float = 0.0,
        residual: bool = True,
        norm: bool = True,
        update_pos: bool = False,
        coord_update_scale: float = 0.1,
        normalize_coord_update: bool = True,
    ):
        super().__init__()

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim or dim)
        self.rbf_dim = int(rbf_dim)
        self.edge_attr_dim = int(edge_attr_dim or 0)
        self.residual = bool(residual)
        self.update_pos = bool(update_pos)
        self.coord_update_scale = float(coord_update_scale)
        self.normalize_coord_update = bool(normalize_coord_update)

        if act is None:
            act = nn.SiLU()

        edge_in_dim = 2 * self.dim + self.rbf_dim + self.edge_attr_dim
        self.edge_mlp = MLP(
            edge_in_dim,
            self.hidden_dim,
            self.hidden_dim,
            act=act,
            dropout=dropout,
            final_act=True,
        )

        self.node_mlp = MLP(
            self.dim + self.hidden_dim,
            self.hidden_dim,
            self.dim,
            act=act,
            dropout=dropout,
            final_act=False,
        )

        self.node_norm = nn.LayerNorm(self.dim) if norm else nn.Identity()

        if self.update_pos:
            self.coord_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                act,
                nn.Linear(self.hidden_dim, 1),
            )
            # Stable start: no coordinate update at initialization.
            nn.init.zeros_(self.coord_mlp[-1].weight)
            nn.init.zeros_(self.coord_mlp[-1].bias)
        else:
            self.coord_mlp = None

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
        radial: torch.Tensor,
        *,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n_nodes = h.size(0)

        pieces = [h[src], h[dst], radial]
        if self.edge_attr_dim > 0:
            if edge_attr is None:
                edge_attr = h.new_zeros((edge_index.size(1), self.edge_attr_dim))
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError(
                    "edge_attr length does not match edge count: "
                    f"{edge_attr.size(0)} versus {edge_index.size(1)}."
                )
            pieces.append(edge_attr.to(device=h.device, dtype=h.dtype))

        edge_input = torch.cat(pieces, dim=-1)
        messages = self.edge_mlp(edge_input)

        aggregated = scatter_sum(messages, src, dim_size=n_nodes)
        dh = self.node_mlp(torch.cat([h, aggregated], dim=-1))

        if self.residual:
            h_out = h + dh
        else:
            h_out = dh

        h_out = self.node_norm(h_out)

        if self.update_pos:
            coord_gate = self.coord_mlp(messages)
            if self.normalize_coord_update:
                direction = edge_vec / edge_dist.clamp_min(1.0e-9)
            else:
                direction = edge_vec
            delta_pos = scatter_sum(direction * coord_gate, src, dim_size=n_nodes)
            pos_out = pos + self.coord_update_scale * delta_pos
        else:
            pos_out = pos

        return h_out, pos_out


class EGNNStack(nn.Module):
    """Stack of EGNN layers operating on Catalyst Data/Batch objects."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int = 4,
        hidden_dim: Optional[int] = None,
        rbf_dim: int = 32,
        cutoff: Optional[float] = None,
        edge_attr_dim: int = 0,
        edge_attr_key: Optional[str] = None,
        act: Optional[nn.Module] = None,
        dropout: float = 0.0,
        residual: bool = True,
        norm: bool = True,
        update_pos: bool = False,
        update_data_pos: bool = False,
        use_precomputed_geometry: bool = False,
        trainable_rbf: bool = False,
        coord_update_scale: float = 0.1,
    ):
        super().__init__()

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.edge_attr_dim = int(edge_attr_dim or 0)
        self.edge_attr_key = edge_attr_key
        self.update_pos = bool(update_pos)
        self.update_data_pos = bool(update_data_pos)
        self.use_precomputed_geometry = bool(use_precomputed_geometry)

        self.rbf = GaussianRadialBasis(
            num_basis=rbf_dim,
            cutoff=cutoff,
            trainable=trainable_rbf,
        )
        self.cutoff_envelope = CosineCutoff(cutoff)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    dim=self.dim,
                    hidden_dim=hidden_dim,
                    rbf_dim=rbf_dim,
                    edge_attr_dim=edge_attr_dim,
                    act=act,
                    dropout=dropout,
                    residual=residual,
                    norm=norm,
                    update_pos=update_pos,
                    coord_update_scale=coord_update_scale,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _get_edge_attr(self, data) -> Optional[torch.Tensor]:
        if self.edge_attr_dim <= 0:
            return None

        if self.edge_attr_key is not None and has_attr(data, self.edge_attr_key):
            value = getattr(data, self.edge_attr_key)
        elif has_attr(data, "edge_attr"):
            value = data.edge_attr
        elif has_attr(data, "x_bnd"):
            value = data.x_bnd
        elif has_attr(data, "x_2"):
            value = data.x_2
        else:
            return None

        if value.dim() == 1:
            value = value.view(-1, 1)

        if value.size(-1) != self.edge_attr_dim:
            raise ValueError(
                f"Expected edge_attr_dim={self.edge_attr_dim}, "
                f"but got edge attributes with shape {tuple(value.shape)}."
            )

        return value

    def forward(self, data):
        if not has_attr(data, "h_scalar"):
            raise AttributeError(
                "EGNNStack requires data.h_scalar. Run EquivariantAtomicEncoder first."
            )

        if not has_attr(data, "pos"):
            raise AttributeError("EGNNStack requires data.pos.")

        if not has_attr(data, "edge_index"):
            raise AttributeError("EGNNStack requires data.edge_index.")

        data = ensure_batch(data)

        h = data.h_scalar
        pos = data.pos
        edge_attr = self._get_edge_attr(data)

        for layer in self.layers:
            edge_vec, edge_dist, _ = get_edge_geometry(
                data,
                pos=pos,
                use_precomputed=self.use_precomputed_geometry,
            )
            radial = self.rbf(edge_dist) * self.cutoff_envelope(edge_dist)
            h, pos = layer(
                h,
                pos,
                data.edge_index,
                edge_vec,
                edge_dist,
                radial,
                edge_attr=edge_attr,
            )

        data.h_scalar = h
        data.h_1 = h

        if self.update_pos:
            data.pos_updated = pos
            if self.update_data_pos:
                data.pos = pos

        return data


# Convenient aliases.
EGNN = EGNNStack
EGNNProcessorCore = EGNNStack
