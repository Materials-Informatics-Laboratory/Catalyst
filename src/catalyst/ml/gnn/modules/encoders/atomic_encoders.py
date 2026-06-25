"""
Catalyst atomic and generic graph encoders.

Recommended location:
    catalyst/ml/gnn/modules/encoders/atomic_encoders.py

Purpose
-------
This file pulls the existing encoder logic out of the old ALIGNN model file and
makes it plug-and-play with the new GenericGNN / OrderProcessor format.

It writes both:

    New order-style hidden names:
        h_1, h_2, h_3

and legacy Catalyst hidden names:

    Atomic_Graph_Data:
        h_atm, h_bnd, h_ang

    Generic_Graph_Data:
        h_g_node, h_a_node, h_a_edge

so old decoders/processors and the new generic framework can both consume the
same encoded graph.

Backward-compatible aliases are provided:

    Encoder_atomic  = AtomicGraphEncoder
    Encoder_generic = GenericFeatureEncoder
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Union

import torch
from torch import nn

from ....nn.mlp import MLP
from ..utils.basis import gaussian, bessel, scalar2basis
from .....graph.graph import Atomic_Graph_Data, Generic_Graph_Data


# =============================================================================
# Helpers
# =============================================================================


def _has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def _first_attr(data, *names: str, required: bool = True):
    for name in names:
        if _has_attr(data, name):
            return getattr(data, name)

    if required:
        raise AttributeError(f"Could not find any of these graph attributes: {names}")

    return None


def _device_from_data(data):
    for name in ("x_atm", "node_G", "x_1", "x_bnd", "node_A", "x_2"):
        if _has_attr(data, name):
            return getattr(data, name).device
    return torch.device("cpu")


def _as_column_if_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize scalar features to [N, 1].

    Vector features are left as [N, F]. Higher-rank singleton scalar features
    such as [N, 1, 1] are flattened to [N, 1].
    """
    if x.dim() == 1:
        return x.view(-1, 1)

    if x.dim() > 2 and all(size == 1 for size in x.shape[1:]):
        return x.reshape(x.size(0), 1)

    return x


def _safe_as_tensor(x, *, device, dtype=torch.float):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def attach_input_order_aliases(data):
    """
    Attach order-style raw feature aliases without deleting existing fields.

    Atomic:
        x_atm -> x_1
        x_bnd -> x_2
        x_ang -> x_3

    Generic:
        node_G -> x_1
        node_A -> x_2
        edge_A -> x_3

    Edges:
        edge_index_G -> edge_index_2
        edge_index_A -> edge_index_3
    """
    if not _has_attr(data, "x_1"):
        if _has_attr(data, "x_atm"):
            data.x_1 = data.x_atm
        elif _has_attr(data, "node_G"):
            data.x_1 = data.node_G

    if not _has_attr(data, "x_2"):
        if _has_attr(data, "x_bnd"):
            data.x_2 = data.x_bnd
        elif _has_attr(data, "node_A"):
            data.x_2 = data.node_A

    if not _has_attr(data, "x_3"):
        if _has_attr(data, "x_ang"):
            data.x_3 = data.x_ang
        elif _has_attr(data, "edge_A"):
            data.x_3 = data.edge_A

    if not _has_attr(data, "edge_index_2") and _has_attr(data, "edge_index_G"):
        data.edge_index_2 = data.edge_index_G

    if not _has_attr(data, "edge_index_3") and _has_attr(data, "edge_index_A"):
        data.edge_index_3 = data.edge_index_A

    return data


def backfill_hidden_aliases(data):
    """
    Backfill legacy hidden names from h_1/h_2/h_3.
    """
    if _has_attr(data, "h_1"):
        if isinstance(data, Atomic_Graph_Data) or _has_attr(data, "x_atm"):
            data.h_atm = data.h_1
        if isinstance(data, Generic_Graph_Data) or _has_attr(data, "node_G"):
            data.h_g_node = data.h_1

    if _has_attr(data, "h_2"):
        if isinstance(data, Atomic_Graph_Data) or _has_attr(data, "x_bnd"):
            data.h_bnd = data.h_2
        if isinstance(data, Generic_Graph_Data) or _has_attr(data, "node_A"):
            data.h_a_node = data.h_2

    if _has_attr(data, "h_3"):
        if isinstance(data, Atomic_Graph_Data) or _has_attr(data, "x_ang"):
            data.h_ang = data.h_3
        if isinstance(data, Generic_Graph_Data) or _has_attr(data, "edge_A"):
            data.h_a_edge = data.h_3

    return data


# =============================================================================
# Atomic/materials encoder
# =============================================================================


class AtomicGraphEncoder(nn.Module):
    """
    Atomistic encoder for Atomic_Graph_Data.

    This is the extracted and modernized version of the old Encoder_atomic.

    Expected raw fields:
        x_atm : atom/species features, typically one-hot [N_atom, num_species]
        x_bnd : bond distances [N_bond] or [N_bond, 1]
        x_ang : optional angles/dihedrals [N_angle] or [N_angle, 1]

    Output hidden fields:
        h_1 / h_atm : atom hidden features
        h_2 / h_bnd : bond hidden features
        h_3 / h_ang : optional angle hidden features
    """

    def __init__(
        self,
        num_species: int,
        cutoff: float,
        act=nn.SiLU(),
        dim: int = 128,
        dihedral: bool = False,
        params_group=None,
    ):
        super().__init__()

        del params_group  # kept for compatibility with older constructor calls

        self.num_species = num_species
        self.cutoff = cutoff
        self.dim = dim
        self.dihedral = dihedral
        self.act_func = act

        self.embed_g_node = nn.Sequential(
            MLP([num_species, dim, dim], act=self.act_func),
            nn.LayerNorm(dim),
        )

        self.embed_a_node = partial(
            bessel,
            start=0,
            end=cutoff,
            num_basis=dim,
        )

        self.embed_a_edge = (
            self.embed_ang_with_dihedral
            if dihedral
            else self.embed_ang_without_dihedral
        )

    def embed_ang_with_dihedral(self, x_ang: torch.Tensor, mask_dih_ang: torch.Tensor) -> torch.Tensor:
        x_ang = x_ang.view(-1)
        mask_dih_ang = mask_dih_ang.bool().view(-1)

        cos_ang = torch.cos(x_ang)
        sin_ang = torch.sin(x_ang)

        h_ang = torch.zeros(
            [x_ang.numel(), self.dim],
            dtype=x_ang.dtype,
            device=x_ang.device,
        )

        # Regular bond angles use cos(theta) basis.
        if (~mask_dih_ang).any():
            h_ang[~mask_dih_ang, : self.dim // 2] = gaussian(
                cos_ang[~mask_dih_ang],
                start=-1,
                end=1,
                num_basis=self.dim // 2,
            )

        # Dihedral angles use cos/sin components.
        if mask_dih_ang.any():
            h_cos_ang = gaussian(
                cos_ang[mask_dih_ang],
                start=-1,
                end=1,
                num_basis=self.dim // 4,
            )
            h_sin_ang = gaussian(
                sin_ang[mask_dih_ang],
                start=-1,
                end=1,
                num_basis=self.dim // 4,
            )
            h_ang[mask_dih_ang, self.dim // 2 :] = torch.cat(
                [h_cos_ang, h_sin_ang],
                dim=-1,
            )

        return h_ang

    def embed_ang_without_dihedral(self, x_ang: torch.Tensor) -> torch.Tensor:
        x_ang = x_ang.view(-1)
        cos_ang = torch.cos(x_ang)
        return gaussian(cos_ang, start=-1, end=1, num_basis=self.dim)

    def forward(self, data):
        data = attach_input_order_aliases(data)

        if not isinstance(data, Atomic_Graph_Data) and not _has_attr(data, "x_atm"):
            raise TypeError(
                "AtomicGraphEncoder expects Atomic_Graph_Data or a graph with x_atm/x_1 fields."
            )

        x_atm = _first_attr(data, "x_atm", "x_1")
        x_bnd = _first_attr(data, "x_bnd", "x_2")

        data.h_1 = self.embed_g_node(x_atm)

        # bessel/scalar basis functions are safest when given flat scalar distances.
        x_bnd_scalar = x_bnd.reshape(-1) if x_bnd.dim() > 1 and x_bnd.size(-1) == 1 else x_bnd
        data.h_2 = self.embed_a_node(x_bnd_scalar)
        if data.h_2.dim() > 2:
            data.h_2 = data.h_2.reshape(data.h_2.size(0), -1)

        x_ang = _first_attr(data, "x_ang", "x_3", required=False)
        if x_ang is not None:
            if self.dihedral:
                mask_dih_ang = _first_attr(data, "mask_dih_ang", required=False)
                if mask_dih_ang is None:
                    # If no mask is supplied, treat all x_ang entries as ordinary bond angles.
                    mask_dih_ang = torch.zeros(
                        x_ang.numel(),
                        dtype=torch.bool,
                        device=x_ang.device,
                    )
                data.h_3 = self.embed_ang_with_dihedral(x_ang, mask_dih_ang)
            else:
                data.h_3 = self.embed_ang_without_dihedral(x_ang)

        return backfill_hidden_aliases(data)


# =============================================================================
# Generic feature encoder
# =============================================================================


class GenericFeatureEncoder(nn.Module):
    """
    Generic feature encoder for Generic_Graph_Data or order-style graph data.

    This is the extracted and modernized version of the old Encoder_generic.

    It is intentionally less atomic-specific than AtomicGraphEncoder. It treats
    1-body features as arbitrary vectors. For 2-body and 3-body features, it can
    either:
        - basis-expand scalar features, or
        - directly MLP-embed vector features.

    Lazy initialization is supported to remain compatible with the old code,
    where the graph feature dimensions were inferred on first forward pass.
    """

    def __init__(
        self,
        act=nn.SiLU(),
        dim: int = 128,
        basis: str = "gaussian",
        x1_dim: Optional[int] = None,
        x2_dim: Optional[int] = None,
        x3_dim: Optional[int] = None,
        basis_for_scalar_edges: bool = True,
        scalar_basis_start: float = -1.0,
        scalar_basis_end: float = 1.0,
        params_group=None,
    ):
        super().__init__()

        del params_group  # kept for compatibility with older constructor calls

        self.dim = dim
        self.act_func = act
        self.basis = basis
        self.basis_for_scalar_edges = basis_for_scalar_edges
        self.scalar_basis_start = scalar_basis_start
        self.scalar_basis_end = scalar_basis_end

        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim

        self.embed_g_node = None
        self.embed_a_node = None
        self.embed_a_edge = None

        if x1_dim is not None:
            self.embed_g_node = self._make_mlp(x1_dim)
        if x2_dim is not None:
            self.embed_a_node = self._make_mlp(x2_dim)
        if x3_dim is not None:
            self.embed_a_edge = self._make_mlp(x3_dim)

    def _make_mlp(self, in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            MLP([in_dim, self.dim, self.dim], act=self.act_func),
            nn.LayerNorm(self.dim),
        )

    def _ensure_mlp(self, attr_name: str, in_dim: int, device) -> nn.Sequential:
        module = getattr(self, attr_name)
        if module is None:
            module = self._make_mlp(in_dim).to(device)
            setattr(self, attr_name, module)
        return module

    def _basis_expand_if_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """
        Basis-expand scalar features to [N, dim].

        The old generic workflow may store node_A/edge_A as [N], [N, 1], or
        occasionally a singleton higher-rank tensor after batching. Passing
        [N, 1] directly into scalar2basis can produce [N, 1, dim], which then
        breaks conv layers expecting 2D edge attributes. Therefore scalar inputs
        are flattened to [N] before basis expansion.
        """
        x = _as_column_if_scalar(x)

        if not self.basis_for_scalar_edges:
            return x

        if x.size(-1) != 1:
            return x

        x_scalar = x.reshape(-1)

        basis_x = scalar2basis(
            x_scalar,
            start=self.scalar_basis_start,
            end=self.scalar_basis_end,
            num_basis=self.dim,
            basis=self.basis,
        )

        basis_x = _safe_as_tensor(basis_x, device=x.device, dtype=torch.float)

        if basis_x.dim() > 2:
            basis_x = basis_x.reshape(basis_x.size(0), -1)

        if basis_x.size(-1) != self.dim:
            raise RuntimeError(
                "Expected scalar2basis to return [N, dim] after scalar flattening, "
                f"but got shape {tuple(basis_x.shape)} with dim={self.dim}."
            )

        return basis_x

    def forward(self, data):
        data = attach_input_order_aliases(data)
        device = _device_from_data(data)

        x_1 = _first_attr(data, "node_G", "x_1")
        x_2 = _first_attr(data, "node_A", "x_2")
        x_3 = _first_attr(data, "edge_A", "x_3", required=False)

        x_1 = _as_column_if_scalar(x_1)
        x_2 = self._basis_expand_if_scalar(x_2)

        embed_1 = self._ensure_mlp("embed_g_node", x_1.size(-1), device=device)
        embed_2 = self._ensure_mlp("embed_a_node", x_2.size(-1), device=device)

        data.h_1 = embed_1(x_1)
        data.h_2 = embed_2(x_2)

        if data.h_1.dim() != 2:
            raise RuntimeError(f"GenericFeatureEncoder produced h_1 with shape {tuple(data.h_1.shape)}; expected [N, dim].")
        if data.h_2.dim() != 2:
            raise RuntimeError(f"GenericFeatureEncoder produced h_2 with shape {tuple(data.h_2.shape)}; expected [E, dim].")

        if x_3 is not None:
            x_3 = self._basis_expand_if_scalar(x_3)
            embed_3 = self._ensure_mlp("embed_a_edge", x_3.size(-1), device=device)
            data.h_3 = embed_3(x_3)
            if data.h_3.dim() != 2:
                raise RuntimeError(f"GenericFeatureEncoder produced h_3 with shape {tuple(data.h_3.shape)}; expected [A, dim].")

        return backfill_hidden_aliases(data)


# Backward-compatible names from the old alignn.py file.
Encoder_atomic = AtomicGraphEncoder
Encoder_generic = GenericFeatureEncoder


__all__ = [
    "AtomicGraphEncoder",
    "GenericFeatureEncoder",
    "Encoder_atomic",
    "Encoder_generic",
    "attach_input_order_aliases",
    "backfill_hidden_aliases",
]
