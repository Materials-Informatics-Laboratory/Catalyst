"""
Equivariant encoders for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/encoders/equivariant_encoders.py

These encoders initialize the hidden fields consumed by equivariant processors:

    data.h_scalar    invariant node features, shape [N, dim]
    data.h_vector    optional vector node features, shape [N, dim, 3]

The encoder is intentionally compatible with the updated Catalyst graph fields:

    z, pos, x_atm, x_1, node_G

For atomistic graphs, ``z`` is usually the atomic number.  For generic graphs,
``z`` can also be a species/type id.  To avoid Al z=13 failing when a caller
passes num_species=1, the default embedding table always covers atomic numbers
up to max_atomic_number.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch import nn


def _has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def _first_existing_attr(data, names: Sequence[str]):
    for name in names:
        if _has_attr(data, name):
            return getattr(data, name)
    return None


class EquivariantAtomicEncoder(nn.Module):
    """Initialize scalar/vector hidden states for equivariant atomistic GNNs.

    Parameters
    ----------
    dim
        Hidden scalar feature dimension.
    num_species
        Optional number of species/type ids.  The embedding table size is chosen
        as ``max(num_species + 1, max_atomic_number + 1)`` so atomic-number z
        values also work.
    max_atomic_number
        Highest atomic number expected when ``data.z`` stores atomic numbers.
    use_atom_features
        If True, also project existing atom/node features such as ``x_atm`` or
        ``x_1`` and add them to the z embedding.
    atom_feature_names
        Candidate graph attributes to use as existing atom/node features.
    initialize_vector
        If True, initializes ``data.h_vector`` as zeros with shape [N, dim, 3].
        EGNN does not require vector channels, but PaiNN-like models will.
    z_feature_name
        Attribute used for species ids / atomic numbers.
    allow_missing_z
        If True and no z field exists, creates a single-species z vector.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_species: Optional[int] = None,
        max_atomic_number: int = 118,
        use_atom_features: bool = True,
        atom_feature_names: Sequence[str] = ("x_atm", "x_1", "x", "node_G"),
        initialize_vector: bool = True,
        z_feature_name: str = "z",
        allow_missing_z: bool = True,
        dropout: float = 0.0,
        norm: bool = False,
    ):
        super().__init__()

        self.dim = int(dim)
        self.num_species = num_species
        self.max_atomic_number = int(max_atomic_number)
        self.use_atom_features = bool(use_atom_features)
        self.atom_feature_names = tuple(atom_feature_names)
        self.initialize_vector = bool(initialize_vector)
        self.z_feature_name = str(z_feature_name)
        self.allow_missing_z = bool(allow_missing_z)

        if self.dim < 1:
            raise ValueError("dim must be >= 1.")

        if self.max_atomic_number < 0:
            raise ValueError("max_atomic_number must be nonnegative.")

        species_size = 0 if num_species is None else int(num_species) + 1
        embedding_size = max(species_size, self.max_atomic_number + 1, 2)
        self.embedding_size = int(embedding_size)

        self.z_embedding = nn.Embedding(self.embedding_size, self.dim)

        if self.use_atom_features:
            self.atom_feature_projection = nn.LazyLinear(self.dim)
        else:
            self.atom_feature_projection = None

        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.dim) if norm else nn.Identity()

    def _get_num_nodes(self, data) -> int:
        if _has_attr(data, self.z_feature_name):
            return int(getattr(data, self.z_feature_name).numel())

        if _has_attr(data, "pos"):
            return int(data.pos.size(0))

        atom_features = _first_existing_attr(data, self.atom_feature_names)
        if atom_features is not None:
            return int(atom_features.size(0))

        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            return int(data.num_nodes)

        raise AttributeError(
            "Cannot infer number of nodes. Expected one of: z, pos, x_atm, x_1, x, node_G."
        )

    def _get_z(self, data) -> torch.Tensor:
        if _has_attr(data, self.z_feature_name):
            z = getattr(data, self.z_feature_name).long().reshape(-1)
            return z

        if not self.allow_missing_z:
            raise AttributeError(f"Graph data is missing required field {self.z_feature_name!r}.")

        n_nodes = self._get_num_nodes(data)
        device = data.pos.device if _has_attr(data, "pos") else None
        z = torch.ones(n_nodes, dtype=torch.long, device=device)
        setattr(data, self.z_feature_name, z)
        return z

    def _get_atom_features(self, data, n_nodes: int) -> Optional[torch.Tensor]:
        if not self.use_atom_features:
            return None

        x = _first_existing_attr(data, self.atom_feature_names)
        if x is None:
            return None

        if x.dim() == 1:
            x = x.view(-1, 1)

        if x.dim() != 2:
            raise ValueError(f"Atom/node features must be 1D or 2D, got shape {tuple(x.shape)}.")

        if x.size(0) != n_nodes:
            raise ValueError(
                "Atom/node feature length does not match number of nodes: "
                f"{x.size(0)} versus {n_nodes}."
            )

        return x.float()

    def forward(self, data):
        z = self._get_z(data)

        if z.numel() == 0:
            raise ValueError("Cannot encode an empty graph with zero nodes.")

        max_z = int(z.max().item())
        min_z = int(z.min().item())
        if min_z < 0:
            raise ValueError(f"z contains negative species/atomic ids. Minimum z={min_z}.")

        if max_z >= self.embedding_size:
            raise ValueError(
                f"z contains id {max_z}, but embedding_size={self.embedding_size}. "
                "Increase max_atomic_number or num_species."
            )

        h = self.z_embedding(z)

        atom_features = self._get_atom_features(data, n_nodes=z.numel())
        if atom_features is not None:
            h = h + self.atom_feature_projection(atom_features.to(device=h.device, dtype=h.dtype))

        h = self.norm(self.dropout(h))

        data.h_scalar = h

        # Compatibility alias with the scalar/order processor ecosystem.
        data.h_1 = h

        if self.initialize_vector:
            data.h_vector = torch.zeros(
                h.size(0),
                h.size(1),
                3,
                device=h.device,
                dtype=h.dtype,
            )

        if not hasattr(data, "num_nodes") or data.num_nodes is None:
            data.num_nodes = int(h.size(0))

        return data


# Backwards-compatible / convenient aliases.
EquivariantEncoder = EquivariantAtomicEncoder
AtomicEquivariantEncoder = EquivariantAtomicEncoder
