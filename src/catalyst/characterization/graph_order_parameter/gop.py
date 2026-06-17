"""Optimized Graph Order Parameter (GOP) descriptor.

    gop = GOP(params)
    predictions = gop.predict(data, flatten=False)
    predictions, flat = gop.predict(data, flatten=True)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from ase.atoms import Atoms

from ...graph.alignnd import alignn_gen
from ...graph.graph import Atomic_Graph_Data, Generic_Graph_Data


class GOP():
    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__()

        default_params = {
            "cutoffs": [],       # [3.0, 4.0, 5.0, ...]
            "interactions": [],  # [['Al', 'Al'], ...] or one-hot/int labels
            "k": 1,
            "with_gini": False,
        }
        if params is None:
            params = default_params
        else:
            merged = default_params.copy()
            merged.update(params)
            params = merged

        self.params = params
        self._interaction_labels = self._prepare_interaction_labels(
            self.params["interactions"]
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def calc_gop(self, G):
        """Calculate the GOP value for a filtered weighted NetworkX graph."""
        op = 0.0

        for sg in nx.connected_components(G):
            degrees = np.array(
                [degree for _, degree in G.degree(sg, weight="weight")],
                dtype=float,
            )
            if degrees.size == 0:
                continue

            unique_degrees, counts = np.unique(degrees, return_counts=True)
            probs = counts.astype(float) / counts.sum()

            # Entropy of weighted-degree classes.
            sg_op = -float(np.sum(probs * np.log(probs)))

            if self.params["with_gini"]:
                denom = 2.0 * len(unique_degrees) * float(np.sum(unique_degrees))
                if denom > 0.0:
                    sg_op += float(
                        np.abs(
                            np.subtract.outer(unique_degrees, unique_degrees)
                        ).sum()
                        / denom
                    )
            else:
                # Original behavior: add expectation value over unique weighted degrees.
                sg_op += float(np.sum(unique_degrees * probs))

            op += math.pow(sg_op, self.params["k"])

        return op

    def build_graph(self, snapshot):
        self._validate_params(require_cutoffs=True, require_interactions=False)

        data = {
            "type": "alignnd",
            "neighbor_params": [max(self.params["cutoffs"]), -1],
            "raw_data": snapshot,
            "is_dihedral": False,
            "include_angs": False,
        }
        return alignn_gen(data=data)

    def predict(self, data, flatten=False):
        """Predict GOP descriptors for one or more snapshots.

        Parameters
        ----------
        data
            A single ASE ``Atoms`` object, a single graph object, or an iterable of
            ASE ``Atoms`` / graph objects.
        flatten : bool, default=False
            If True, also return a flattened feature matrix with one row per
            snapshot.
        """
        self._validate_params(require_cutoffs=True, require_interactions=True)

        snapshots = self._as_snapshot_list(data)
        predictions = []

        for snapshot in snapshots:
            predictions.append([])

            if isinstance(snapshot, Atoms):
                graph = self.build_graph(snapshot)
            else:
                graph = snapshot

            graph_arrays = self._extract_graph_arrays(graph)
            species_labels = graph_arrays["species_labels"]
            edge_src = graph_arrays["edge_src"]
            edge_dst = graph_arrays["edge_dst"]
            edge_distances = graph_arrays["edge_distances"]

            interactions = [
                (
                    self._adapt_interaction_label(i_label, species_labels),
                    self._adapt_interaction_label(j_label, species_labels),
                )
                for i_label, j_label in self._interaction_labels
            ]

            self._validate_interactions_possible(interactions, species_labels)

            for rc in self.params["cutoffs"]:
                predictions[-1].append([])

                for i_label, j_label in interactions:
                    predictions[-1][-1].append([])

                    edge_list = self._build_filtered_edge_list(
                        edge_src=edge_src,
                        edge_dst=edge_dst,
                        edge_distances=edge_distances,
                        species_labels=species_labels,
                        i_label=i_label,
                        j_label=j_label,
                        cutoff=rc,
                    )

                    G = nx.Graph(edge_list)
                    predictions[-1][-1][-1].append(self.calc_gop(G))

        if flatten:
            flattened_predictions = []
            for pred in predictions:
                row = []
                for cutoff_block in pred:
                    for interaction_block in cutoff_block:
                        row.extend(interaction_block)
                flattened_predictions.append(row)
            return predictions, np.array(flattened_predictions)

        return predictions

    # -------------------------------------------------------------------------
    # Parameter and input handling
    # -------------------------------------------------------------------------
    def _validate_params(self, require_cutoffs=True, require_interactions=True):
        if require_cutoffs and len(self.params.get("cutoffs", [])) == 0:
            raise ValueError("GOP requires at least one cutoff in params['cutoffs'].")
        if require_interactions and len(self.params.get("interactions", [])) == 0:
            raise ValueError(
                "GOP requires at least one interaction in params['interactions']."
            )

    @staticmethod
    def _as_snapshot_list(data):
        if isinstance(data, (Atoms, Atomic_Graph_Data, Generic_Graph_Data)):
            return [data]

        # Some graph-like classes may not pass isinstance checks but still expose
        # the expected graph attributes. Treat them as single snapshots.
        if GOP._has_graph_attr(data, "edge_index_G"):
            return [data]

        if isinstance(data, Iterable):
            return list(data)

        raise TypeError(
            "data must be an ASE Atoms object, a graph object, or an iterable of "
            "ASE Atoms / graph objects."
        )

    @staticmethod
    def _has_graph_attr(graph, name: str) -> bool:
        if hasattr(graph, name):
            return True
        try:
            graph[name]
            return True
        except Exception:
            return False

    @staticmethod
    def _get_graph_attr(graph, name: str):
        try:
            return graph[name]
        except Exception:
            if hasattr(graph, name):
                return getattr(graph, name)
            raise AttributeError(f"Graph object does not contain required field {name!r}.")

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    # -------------------------------------------------------------------------
    # Graph extraction and normalization
    # -------------------------------------------------------------------------
    def _extract_graph_arrays(self, graph):
        if isinstance(graph, (Atomic_Graph_Data, Generic_Graph_Data)) or self._has_graph_attr(graph, "edge_index_G"):
            if self._has_graph_attr(graph, "x_atm"):
                symbols = self._get_graph_attr(graph, "x_atm")
                edge_distances = self._get_graph_attr(graph, "x_bnd")
            elif self._has_graph_attr(graph, "node_G"):
                symbols = self._get_graph_attr(graph, "node_G")
                edge_distances = self._get_graph_attr(graph, "node_A")
            else:
                raise TypeError(
                    "Unsupported graph object: expected either 'x_atm'/'x_bnd' "
                    "or 'node_G'/'node_A' fields."
                )
        else:
            raise TypeError(
                f"Unsupported snapshot type {type(graph)}. Expected ASE Atoms or "
                "Atomic_Graph_Data / Generic_Graph_Data-like graph object."
            )

        edge_index = self._to_numpy(self._get_graph_attr(graph, "edge_index_G")).astype(int)
        if edge_index.ndim != 2:
            raise ValueError("edge_index_G must be a 2D array with shape (2, E) or (E, 2).")
        if edge_index.shape[0] == 2:
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
        elif edge_index.shape[1] == 2:
            edge_src = edge_index[:, 0]
            edge_dst = edge_index[:, 1]
        else:
            raise ValueError("edge_index_G must have shape (2, E) or (E, 2).")

        edge_distances = self._to_numpy(edge_distances)
        edge_distances = self._normalize_edge_distances(edge_distances, len(edge_src))

        species_labels = self._normalize_species_labels(symbols)
        if len(species_labels) <= int(max(edge_src.max(initial=0), edge_dst.max(initial=0))):
            raise ValueError(
                "edge_index_G references node indices beyond the available species labels."
            )

        if len(edge_distances) != len(edge_src):
            raise ValueError(
                "Number of edge distances does not match number of graph edges: "
                f"{len(edge_distances)} distances vs {len(edge_src)} edges."
            )

        return {
            "species_labels": species_labels,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_distances": edge_distances,
        }

    @staticmethod
    def _normalize_edge_distances(edge_distances, expected_edges: int):
        edge_distances = np.asarray(edge_distances)
        edge_distances = np.squeeze(edge_distances)

        if edge_distances.ndim == 0:
            edge_distances = edge_distances.reshape(1)
        elif edge_distances.ndim > 1:
            # Preserve the original assumption that each edge has a scalar distance.
            # If the edge feature array has a singleton last dimension, flatten it;
            # otherwise the caller needs to provide/select a scalar distance channel.
            if edge_distances.shape[-1] == 1:
                edge_distances = edge_distances.reshape(-1)
            elif edge_distances.shape[0] == expected_edges:
                edge_distances = edge_distances[:, 0]
            else:
                raise ValueError(
                    "Expected scalar edge distances. Received an edge feature array "
                    f"with shape {edge_distances.shape}."
                )

        return edge_distances.astype(float, copy=False).reshape(-1)

    def _normalize_species_labels(self, symbols):
        symbols = self._to_numpy(symbols)

        # One-hot or dense feature representation: use argmax to recover species ID,
        # preserving the original implementation's behavior.
        if symbols.ndim >= 2 and not self._is_string_array(symbols):
            return np.argmax(symbols, axis=1).astype(int)

        symbols = np.asarray(symbols).reshape(-1)
        if self._is_string_array(symbols):
            return symbols.astype(str)

        # Numeric scalar species labels.
        if np.issubdtype(symbols.dtype, np.integer):
            return symbols.astype(int)

        # Float scalar labels that are effectively integers.
        if np.issubdtype(symbols.dtype, np.floating):
            rounded = np.rint(symbols)
            if np.allclose(symbols, rounded):
                return rounded.astype(int)

        raise ValueError(
            "Could not infer species labels from graph node data. Expected one-hot "
            "vectors, string symbols, or integer species IDs."
        )

    @staticmethod
    def _is_string_array(array) -> bool:
        arr = np.asarray(array)
        return arr.dtype.kind in {"U", "S"} or any(
            isinstance(x, str) for x in arr.reshape(-1)[: min(arr.size, 8)]
        )

    # -------------------------------------------------------------------------
    # Interaction handling
    # -------------------------------------------------------------------------
    def _prepare_interaction_labels(self, interactions):
        prepared = []
        for interaction in interactions:
            if len(interaction) != 2:
                raise ValueError(
                    "Each interaction must contain exactly two species labels. "
                    f"Received: {interaction!r}"
                )
            prepared.append(
                (
                    self._normalize_interaction_label_static(interaction[0]),
                    self._normalize_interaction_label_static(interaction[1]),
                )
            )
        return prepared

    @staticmethod
    def _normalize_interaction_label_static(label):
        if isinstance(label, str):
            return label

        arr = np.asarray(label)
        if arr.ndim == 0:
            scalar = arr.item()
            if isinstance(scalar, str):
                return scalar
            if isinstance(scalar, (int, np.integer)):
                return int(scalar)
            if isinstance(scalar, (float, np.floating)) and float(scalar).is_integer():
                return int(scalar)
            return scalar

        # Original behavior: one-hot vector -> nonzero/argmax index.
        nonzero = np.where(arr != 0)[0]
        if nonzero.size == 0:
            raise ValueError(f"Interaction label {label!r} contains no nonzero entry.")
        return int(nonzero[0])

    def _adapt_interaction_label(self, label, species_labels):
        # If graph species labels are strings, keep string labels as strings.
        # Numeric labels can optionally be mapped to strings if params contains a
        # species/elements list.
        if self._is_string_array(species_labels):
            if isinstance(label, str):
                return label
            species_list = self._species_name_list()
            if species_list is not None and isinstance(label, (int, np.integer)):
                return species_list[int(label)]
            return label

        # If graph species labels are numeric, map string labels using an optional
        # species/elements list when available; otherwise leave the error for the
        # interaction validation step, which will report the unsupported label.
        if isinstance(label, str):
            species_list = self._species_name_list()
            if species_list is not None:
                try:
                    return int(species_list.index(label))
                except ValueError:
                    pass
        return label

    def _species_name_list(self):
        for key in ("species", "elements", "element_symbols", "species_symbols"):
            if key in self.params and self.params[key] is not None:
                return list(self.params[key])
        return None

    @staticmethod
    def _validate_interactions_possible(interactions, species_labels):
        available = set(np.asarray(species_labels).reshape(-1).tolist())
        for i_label, j_label in interactions:
            if i_label not in available or j_label not in available:
                raise ValueError(
                    "Requested interaction is not possible with the given graph object: "
                    f"({i_label!r}, {j_label!r}). Available labels: "
                    f"{sorted(available, key=str)}"
                )

    # -------------------------------------------------------------------------
    # Sparse edge filtering
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_filtered_edge_list(
        edge_src,
        edge_dst,
        edge_distances,
        species_labels,
        i_label,
        j_label,
        cutoff,
    ):
        src_species = species_labels[edge_src]
        dst_species = species_labels[edge_dst]

        if i_label == j_label:
            mask = (
                (src_species == i_label)
                & (dst_species == i_label)
                & (edge_src != edge_dst)
                & (edge_distances < cutoff)
                & (edge_distances > 0.0)
            )
        else:
            # Preserve the original directional interaction behavior. If both
            # directions exist in edge_index_G, NetworkX will collapse duplicate
            # undirected edges when nx.Graph(edge_list) is constructed.
            mask = (
                (src_species == i_label)
                & (dst_species == j_label)
                & (edge_src != edge_dst)
                & (edge_distances < cutoff)
                & (edge_distances > 0.0)
            )

        selected_src = edge_src[mask]
        selected_dst = edge_dst[mask]
        selected_dist = edge_distances[mask]

        return [
            (int(u), int(v), {"weight": float(1.0 / d)})
            for u, v, d in zip(selected_src, selected_dst, selected_dist)
        ]
