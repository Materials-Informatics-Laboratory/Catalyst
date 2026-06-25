"""
Generic equivariant decoders for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/decoders/equivariant_decoders.py

The decoder names are intentionally generic.  They describe tensor behavior,
not a particular physics interpretation:

    output_type="scalar"
        Predict invariant scalar quantities from scalar node features.

    output_type="vector"
        Predict equivariant vector quantities using geometry-aware vector
        aggregation.  This avoids producing arbitrary XYZ vectors from scalar
        features alone.

    output_type="scalar_gradient"
        Predict a scalar quantity and return its gradient with respect to a
        selected input tensor, usually data.pos.  In atomistic force-field use,
        users may interpret this as energy + force, but the core decoder keeps
        the generic names "scalar" and "gradient".

Expected processor output
-------------------------
The decoder expects an equivariant processor to have created:

    data.h_scalar     [N, hidden_dim]

and graph data to contain:

    data.pos          [N, 3]
    data.edge_index   [2, E]
    data.cell         [B, 3, 3] or [3, 3]
    data.shifts       [E, 3]
    data.batch        [N], optional for single graphs

Return convention
-----------------
By default the decoder returns dictionaries:

    scalar:
        {"scalar": tensor}

    vector:
        {"vector": tensor}

    scalar_gradient:
        {"scalar": tensor, "gradient": tensor}

For scalar_gradient, the gradient is computed as:

    gradient = sign * d scalar.sum() / d data.<gradient_input_key>

where sign is controlled by gradient_sign:

    "positive" -> +grad
    "negative" -> -grad
    "none"     -> +grad

Important scalar_gradient note
------------------------------
For a scalar_gradient output to track geometric dependence, data.pos must have
requires_grad=True *before* the encoder/processor forward pass recomputes edge
geometry from positions.  Calling requires_grad_ inside the decoder is too late
if h_scalar has already been computed from detached positions.

Use prepare_scalar_gradient_input(data) at the very beginning of a model forward
or in a wrapper before encoder -> processor -> decoder.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn


# =============================================================================
# Generic helpers
# =============================================================================


def has_attr(data, name: str) -> bool:
    """Return True if data has a non-None attribute."""
    return hasattr(data, name) and getattr(data, name) is not None


def first_existing_attr(data, names: Sequence[str]):
    """Return the first non-None attribute from a list of candidate names."""
    for name in names:
        if has_attr(data, name):
            return getattr(data, name)
    return None


def ensure_batch(data):
    """Ensure a PyG-style data.batch vector exists.

    For an unbatched single graph, create an all-zero batch vector.
    """
    if not has_attr(data, "batch"):
        if not has_attr(data, "pos"):
            raise AttributeError("Cannot create data.batch because data.pos is missing.")
        data.batch = torch.zeros(
            data.pos.size(0),
            dtype=torch.long,
            device=data.pos.device,
        )
    return data


def infer_num_graphs(data, batch: Optional[torch.Tensor] = None) -> int:
    """Infer number of graphs in a Data/Batch object."""
    if has_attr(data, "num_graphs"):
        return int(data.num_graphs)

    if has_attr(data, "ptr"):
        return int(data.ptr.numel() - 1)

    if batch is None:
        if has_attr(data, "batch"):
            batch = data.batch
        else:
            return 1

    return int(batch.max().item()) + 1 if batch.numel() > 0 else 1


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter-sum along dim 0 without requiring torch_scatter."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(out_shape)

    if src.numel() > 0:
        out.index_add_(0, index, src)

    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter-mean along dim 0 without requiring torch_scatter."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    summed = scatter_sum(src, index, dim_size=dim_size)

    count = src.new_zeros((dim_size,))
    if index.numel() > 0:
        ones = torch.ones_like(index, dtype=src.dtype)
        count.index_add_(0, index, ones)

    view_shape = (dim_size,) + (1,) * (src.dim() - 1)
    return summed / count.clamp_min(1.0).view(view_shape)


def reduce_nodes(
    node_values: torch.Tensor,
    batch: torch.Tensor,
    *,
    reduce: str = "sum",
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Reduce node-level values to graph-level values."""
    reduce = str(reduce).lower()

    if reduce == "sum":
        return scatter_sum(node_values, batch, dim_size=dim_size)

    if reduce == "mean":
        return scatter_mean(node_values, batch, dim_size=dim_size)

    if reduce in {"none", "node", "identity"}:
        return node_values

    raise ValueError(f"Unsupported reduce={reduce!r}. Supported: sum, mean, none.")


def normalize_output_type(output_type: str) -> str:
    """Normalize decoder output type aliases."""
    key = str(output_type).lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "scalar": "scalar",
        "node_scalar": "scalar",
        "graph_scalar": "scalar",
        "invariant": "scalar",

        "vector": "vector",
        "node_vector": "vector",
        "graph_vector": "vector",
        "equivariant_vector": "vector",

        "scalar_gradient": "scalar_gradient",
        "scalar_grad": "scalar_gradient",
        "gradient": "scalar_gradient",
        "grad": "scalar_gradient",
    }

    if key not in aliases:
        raise ValueError(
            f"Unsupported output_type={output_type!r}. "
            "Supported: scalar, vector, scalar_gradient."
        )

    return aliases[key]


def normalize_output_level(output_level: str) -> str:
    """Normalize output-level aliases."""
    key = str(output_level).lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "graph": "graph",
        "global": "graph",
        "structure": "graph",
        "system": "graph",

        "node": "node",
        "local": "node",
        "atom": "node",
        "site": "node",
    }

    if key not in aliases:
        raise ValueError(
            f"Unsupported output_level={output_level!r}. "
            "Supported: graph, node."
        )

    return aliases[key]


def gradient_sign_multiplier(gradient_sign: str) -> float:
    """Map a gradient sign string to a multiplier."""
    key = str(gradient_sign).lower().replace("-", "_").replace(" ", "_")

    if key in {"positive", "plus", "+", "raw", "none", "identity"}:
        return 1.0

    if key in {"negative", "minus", "-", "force_like", "descent"}:
        return -1.0

    raise ValueError(
        f"Unsupported gradient_sign={gradient_sign!r}. "
        "Supported: positive, negative, raw/none."
    )


def get_scalar_features(
    data,
    feature_names: Sequence[str] = ("h_scalar", "h_1", "h_node"),
) -> torch.Tensor:
    """Return scalar node features from a Data/Batch object."""
    h = first_existing_attr(data, feature_names)
    if h is None:
        raise AttributeError(
            "Could not find scalar node features. Expected one of: "
            + ", ".join(feature_names)
        )

    if h.dim() != 2:
        raise ValueError(f"Scalar node features must have shape [N, F], got {tuple(h.shape)}.")

    return h


def _cell_per_edge(data, edge_batch: torch.Tensor) -> torch.Tensor:
    """Return per-edge cell tensor with shape [E, 3, 3]."""
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute differentiable edge geometry from pos/cell/shifts.

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
        if not has_attr(data, "pos"):
            raise AttributeError("data.pos is required to compute edge geometry.")
        pos = data.pos

    if not has_attr(data, "edge_index"):
        raise AttributeError("data.edge_index is required to compute edge geometry.")

    data = ensure_batch(data)

    src, dst = data.edge_index
    edge_batch = data.batch[src]

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


def prepare_scalar_gradient_input(
    data,
    *,
    gradient_input_key: str = "pos",
    clone: bool = False,
):
    """Enable gradients on the input tensor used by scalar_gradient decoders.

    This must be called before encoder -> processor -> decoder if the scalar
    depends on geometry computed inside the processor.
    """
    if not has_attr(data, gradient_input_key):
        raise AttributeError(f"data.{gradient_input_key} is required for scalar_gradient output.")

    value = getattr(data, gradient_input_key)

    if clone:
        value = value.detach().clone()

    value.requires_grad_(True)
    setattr(data, gradient_input_key, value)
    return data


def compute_scalar_gradient(
    scalar: torch.Tensor,
    wrt: torch.Tensor,
    *,
    sign: Union[str, float] = "positive",
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Compute signed gradient of scalar.sum() with respect to a tensor."""
    if isinstance(sign, str):
        multiplier = gradient_sign_multiplier(sign)
    else:
        multiplier = float(sign)

    grad = torch.autograd.grad(
        scalar.sum(),
        wrt,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )[0]

    if grad is None:
        raise RuntimeError(
            "Scalar-gradient computation returned None. This usually means the "
            "scalar output is not connected to the requested gradient input. "
            "For position gradients, ensure data.pos.requires_grad=True before "
            "the encoder/processor forward pass."
        )

    return multiplier * grad


# =============================================================================
# Modules
# =============================================================================


class MLP(nn.Module):
    """Small configurable MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        act: Optional[nn.Module] = None,
        dropout: float = 0.0,
        final_act: bool = False,
    ):
        super().__init__()

        in_dim = int(in_dim)
        hidden_dim = int(hidden_dim)
        out_dim = int(out_dim)
        num_layers = int(num_layers)

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        if act is None:
            act = nn.SiLU()

        if num_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), act]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))

            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))

            layers.append(nn.Linear(hidden_dim, out_dim))

        if final_act:
            layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EquivariantDecoder(nn.Module):
    """Generic decoder for equivariant GNN outputs.

    Parameters
    ----------
    dim
        Hidden scalar feature dimension.
    output_type
        One of: "scalar", "vector", "scalar_gradient".
    output_level
        "node" or "graph".
    out_dim
        Number of scalar channels for scalar outputs, or number of vector
        channels for vector outputs.
    reduce
        Node-to-graph reduction for graph-level outputs.  Supports "sum" and
        "mean".
    gradient_input_key
        Data attribute to differentiate with respect to for scalar_gradient.
    gradient_sign
        "positive" for +grad, "negative" for -grad.
    squeeze_last_dim
        If True and out_dim=1, removes the last scalar channel dimension.
    squeeze_vector_channels
        If True and output_type="vector" and out_dim=1, returns [N, 3] or
        [B, 3] instead of [N, 1, 3] or [B, 1, 3].
    return_dict
        If True, returns dictionaries with generic keys.  If False, returns
        the primary tensor for scalar/vector and a tuple for scalar_gradient.
    """

    def __init__(
        self,
        dim: int,
        *,
        output_type: str = "scalar",
        output_level: str = "graph",
        out_dim: int = 1,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        act: Optional[nn.Module] = None,
        dropout: float = 0.0,
        reduce: str = "sum",
        scalar_feature_names: Sequence[str] = ("h_scalar", "h_1", "h_node"),
        gradient_input_key: str = "pos",
        gradient_sign: str = "negative",
        create_graph: bool = True,
        retain_graph: bool = True,
        allow_unused_gradient: bool = False,
        vector_hidden_dim: Optional[int] = None,
        vector_num_layers: Optional[int] = None,
        vector_use_edge_geometry: bool = True,
        vector_reduce: Optional[str] = None,
        squeeze_last_dim: bool = False,
        squeeze_vector_channels: bool = True,
        return_dict: bool = True,
        scalar_key: str = "scalar",
        vector_key: str = "vector",
        gradient_key: str = "gradient",
    ):
        super().__init__()

        self.dim = int(dim)
        self.output_type = normalize_output_type(output_type)
        self.output_level = normalize_output_level(output_level)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim or dim)
        self.num_layers = int(num_layers)
        self.reduce = str(reduce).lower()
        self.scalar_feature_names = tuple(scalar_feature_names)

        self.gradient_input_key = str(gradient_input_key)
        self.gradient_sign = str(gradient_sign)
        self.create_graph = bool(create_graph)
        self.retain_graph = bool(retain_graph)
        self.allow_unused_gradient = bool(allow_unused_gradient)

        self.vector_use_edge_geometry = bool(vector_use_edge_geometry)
        self.vector_reduce = str(vector_reduce or reduce).lower()

        self.squeeze_last_dim = bool(squeeze_last_dim)
        self.squeeze_vector_channels = bool(squeeze_vector_channels)
        self.return_dict = bool(return_dict)

        self.scalar_key = str(scalar_key)
        self.vector_key = str(vector_key)
        self.gradient_key = str(gradient_key)

        if self.dim < 1:
            raise ValueError("dim must be >= 1.")
        if self.out_dim < 1:
            raise ValueError("out_dim must be >= 1.")

        if act is None:
            act = nn.SiLU()

        # Scalar head is used by scalar and scalar_gradient.
        if self.output_type in {"scalar", "scalar_gradient"}:
            self.scalar_mlp = MLP(
                self.dim,
                self.hidden_dim,
                self.out_dim,
                num_layers=self.num_layers,
                act=act,
                dropout=dropout,
                final_act=False,
            )
        else:
            self.scalar_mlp = None

        # Vector head predicts edge scalar gates and aggregates them along
        # geometric directions, yielding an equivariant vector field.
        if self.output_type == "vector":
            v_hidden = int(vector_hidden_dim or self.hidden_dim)
            v_layers = int(vector_num_layers or self.num_layers)
            self.vector_edge_mlp = MLP(
                2 * self.dim + 1,
                v_hidden,
                self.out_dim,
                num_layers=v_layers,
                act=act,
                dropout=dropout,
                final_act=False,
            )
        else:
            self.vector_edge_mlp = None

    def _get_h_scalar(self, data) -> torch.Tensor:
        return get_scalar_features(data, self.scalar_feature_names)

    def _get_batch(self, data, h: torch.Tensor) -> torch.Tensor:
        data = ensure_batch(data)
        if data.batch.size(0) != h.size(0):
            raise ValueError(
                "data.batch and h_scalar disagree on number of nodes: "
                f"{data.batch.size(0)} versus {h.size(0)}."
            )
        return data.batch

    def _format_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        if self.squeeze_last_dim and scalar.dim() >= 2 and scalar.size(-1) == 1:
            scalar = scalar.squeeze(-1)
        return scalar

    def _format_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if self.squeeze_vector_channels and vector.dim() >= 3 and vector.size(-2) == 1:
            vector = vector.squeeze(-2)
        return vector

    def decode_scalar(self, data) -> torch.Tensor:
        """Decode invariant scalar output."""
        h = self._get_h_scalar(data)
        node_scalar = self.scalar_mlp(h)

        if self.output_level == "node":
            return self._format_scalar(node_scalar)

        batch = self._get_batch(data, h)
        n_graphs = infer_num_graphs(data, batch)
        graph_scalar = reduce_nodes(node_scalar, batch, reduce=self.reduce, dim_size=n_graphs)
        return self._format_scalar(graph_scalar)

    def decode_vector(self, data) -> torch.Tensor:
        """Decode equivariant vector output through edge-direction aggregation."""
        h = self._get_h_scalar(data)

        if not has_attr(data, "edge_index"):
            raise AttributeError("Vector decoding requires data.edge_index.")

        if not has_attr(data, "pos"):
            raise AttributeError("Vector decoding requires data.pos.")

        src, dst = data.edge_index

        if self.vector_use_edge_geometry:
            _, edge_dist, edge_unit = get_edge_geometry(data)
        else:
            if not has_attr(data, "edge_vec"):
                raise AttributeError(
                    "vector_use_edge_geometry=False requires precomputed data.edge_vec."
                )
            edge_vec = data.edge_vec.to(device=h.device, dtype=h.dtype)
            edge_dist = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
            edge_unit = edge_vec / edge_dist.clamp_min(1.0e-9)

        edge_input = torch.cat([h[src], h[dst], edge_dist.to(device=h.device, dtype=h.dtype)], dim=-1)
        edge_gates = self.vector_edge_mlp(edge_input)  # [E, out_dim]

        edge_vectors = edge_gates.unsqueeze(-1) * edge_unit.unsqueeze(1)  # [E, out_dim, 3]
        node_vector = scatter_sum(edge_vectors, src, dim_size=h.size(0))

        if self.output_level == "node":
            return self._format_vector(node_vector)

        batch = self._get_batch(data, h)
        n_graphs = infer_num_graphs(data, batch)
        graph_vector = reduce_nodes(node_vector, batch, reduce=self.vector_reduce, dim_size=n_graphs)
        return self._format_vector(graph_vector)

    def decode_scalar_gradient(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode scalar and compute signed gradient with respect to an input."""
        if not has_attr(data, self.gradient_input_key):
            raise AttributeError(
                f"scalar_gradient output requires data.{self.gradient_input_key}."
            )

        wrt = getattr(data, self.gradient_input_key)

        if not wrt.requires_grad:
            raise RuntimeError(
                f"data.{self.gradient_input_key}.requires_grad is False. "
                "For scalar_gradient output, call "
                "prepare_scalar_gradient_input(data, gradient_input_key="
                f"{self.gradient_input_key!r}) before encoder/processor forward."
            )

        scalar = self.decode_scalar(data)
        gradient = compute_scalar_gradient(
            scalar,
            wrt,
            sign=self.gradient_sign,
            create_graph=self.create_graph,
            retain_graph=self.retain_graph,
            allow_unused=self.allow_unused_gradient,
        )

        return scalar, gradient

    def forward(self, data):
        if self.output_type == "scalar":
            scalar = self.decode_scalar(data)
            if self.return_dict:
                return {self.scalar_key: scalar}
            return scalar

        if self.output_type == "vector":
            vector = self.decode_vector(data)
            if self.return_dict:
                return {self.vector_key: vector}
            return vector

        if self.output_type == "scalar_gradient":
            scalar, gradient = self.decode_scalar_gradient(data)
            if self.return_dict:
                return {
                    self.scalar_key: scalar,
                    self.gradient_key: gradient,
                }
            return scalar, gradient

        raise RuntimeError(f"Unhandled output_type={self.output_type!r}.")


# Backwards-compatible and convenient aliases.
Decoder = EquivariantDecoder
GenericEquivariantDecoder = EquivariantDecoder
ScalarGradientDecoder = EquivariantDecoder
