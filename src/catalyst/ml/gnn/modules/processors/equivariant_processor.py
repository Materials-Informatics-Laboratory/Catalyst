"""
Equivariant processor for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/processors/equivariant_processor.py

This processor is the architecture-level wrapper parallel to ScalarProcessor and
OrderProcessor.  It dispatches to equivariant backends such as EGNN while keeping
the public model-building API stable.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..equivariant.egnn import EGNNStack, ensure_batch


class EquivariantProcessor(nn.Module):
    """Processor branch for equivariant GNNs.

    Parameters
    ----------
    dim
        Hidden scalar feature dimension.
    num_convs, num_layers
        Number of equivariant message-passing layers.  ``num_convs`` is accepted
        for consistency with existing Catalyst model builders.
    equivariant_type
        Backend type.  Currently supports ``"egnn"``.
    cutoff
        Optional radial cutoff used by the EGNN radial basis envelope.
    update_pos
        If True, EGNN coordinate updates are enabled.  For most energy-gradient
        force models, keep this False and let forces come from ``-dE/dpos``.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_convs: Optional[int] = None,
        num_layers: Optional[int] = None,
        equivariant_type: str = "egnn",
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
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            # Keep this nonfatal but visible during development.
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown EquivariantProcessor keyword argument(s): {unknown}")

        self.dim = int(dim)
        self.num_layers = int(num_layers if num_layers is not None else (num_convs if num_convs is not None else 4))
        self.equivariant_type = str(equivariant_type).lower()

        if act is None:
            act = nn.SiLU()

        if self.equivariant_type in {"egnn", "e_gnn", "equivariant_gnn"}:
            self.processor = EGNNStack(
                dim=self.dim,
                num_layers=self.num_layers,
                hidden_dim=hidden_dim,
                rbf_dim=rbf_dim,
                cutoff=cutoff,
                edge_attr_dim=edge_attr_dim,
                edge_attr_key=edge_attr_key,
                act=act,
                dropout=dropout,
                residual=residual,
                norm=norm,
                update_pos=update_pos,
                update_data_pos=update_data_pos,
                use_precomputed_geometry=use_precomputed_geometry,
                trainable_rbf=trainable_rbf,
                coord_update_scale=coord_update_scale,
            )
        else:
            raise ValueError(
                f"Unsupported equivariant_type={equivariant_type!r}. "
                "Currently supported: 'egnn'."
            )

    def _ensure_hidden_aliases(self, data):
        if not hasattr(data, "h_scalar") or data.h_scalar is None:
            # Allow explicit h_1/h_node to flow into the equivariant processor
            # for compatibility with other Catalyst model branches.
            if hasattr(data, "h_1") and data.h_1 is not None:
                data.h_scalar = data.h_1
            elif hasattr(data, "h_node") and data.h_node is not None:
                data.h_scalar = data.h_node
            else:
                raise AttributeError(
                    "EquivariantProcessor requires data.h_scalar. "
                    "Run EquivariantAtomicEncoder before the processor."
                )

        data.h_1 = data.h_scalar

        if hasattr(data, "pos") and data.pos is not None:
            if not hasattr(data, "num_nodes") or data.num_nodes is None:
                data.num_nodes = int(data.pos.size(0))

        return data

    def forward(self, data):
        data = ensure_batch(data)
        data = self._ensure_hidden_aliases(data)
        data = self.processor(data)

        # Compatibility aliases for downstream decoders.
        data.h_1 = data.h_scalar
        data.h_node = data.h_scalar

        return data


# Backwards-compatible alias, matching the old processor files.
Processor = EquivariantProcessor
