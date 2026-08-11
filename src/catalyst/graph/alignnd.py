from ..properties.physics_database import Physics_data
from ..properties.structure_properties import *
from ..utilities.data_tools import remove_duplicate_list_pairs
from .graph import Atomic_Graph_Data
from .graph import atoms2graph
from .graph import build_equivariant_atomic_fields
from .graph import line_graph
from .graph import dihedral_graph
from joblib import Parallel, delayed
import warnings
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------

def check_params(target_dict):
    source_dict = {
        'type': None,
        'raw_data': None,
        'node_labels': None,
        'element_list': None,
        'neighbor_params': [5.0, -1],
        'is_dihedral': False,
        'store_raw_data': False,
        'use_pt': False,
        'include_angs': False,
        'cpu_cores': -1,
        'store_atoms_type': 'ase-atoms',
        'include_equivariant_fields': True,

        # Universal graph retry controls. These are passed through alignn_gen to
        # all graph builders that can actually change cutoff/k.
        'auto_retry_graph': True,
        'max_graph_attempts': 8,
        'k_step': 4,
        'cutoff_scale': 1.25,
        'max_k': None,
        'max_cutoff': None,
        'require_bonds': True,
        'require_angles': True,
        'require_dihedrals': False,
        'retry_verbose': True,
    }
    updated_dict = target_dict.copy()

    for key, source_value in source_dict.items():
        if key not in updated_dict:
            updated_dict[key] = source_value

    return updated_dict


def alignn_gen(data):
    data = check_params(data)

    retry_kwargs = dict(
        auto_retry_graph=data['auto_retry_graph'],
        max_graph_attempts=data['max_graph_attempts'],
        k_step=data['k_step'],
        cutoff_scale=data['cutoff_scale'],
        max_k=data['max_k'],
        max_cutoff=data['max_cutoff'],
        require_bonds=data['require_bonds'],
        require_angles=data['require_angles'],
        require_dihedrals=data['require_dihedrals'],
        retry_verbose=data['retry_verbose'],
    )

    if data['type'] == 'alignnd':
        graphs = alignnd(
            atoms=data['raw_data'],
            neighbor_params=data['neighbor_params'],
            dihedral=data['is_dihedral'],
            store_atoms=data['store_raw_data'],
            use_pt=data['use_pt'],
            include_angs=data['include_angs'],
            include_equivariant_fields=data['include_equivariant_fields'],
            atom_labels=data['node_labels'],
            all_elements=data['element_list'],
            **retry_kwargs,
        )
    elif data['type'] == 'realignnd':
        graphs = realignnd(
            structures=data['raw_data'],
            neighbor_params=data['neighbor_params'],
            dihedral=data['is_dihedral'],
            store_atoms=data['store_raw_data'],
            use_pt=data['use_pt'],
            include_angs=data['include_angs'],
            include_equivariant_fields=data['include_equivariant_fields'],
            atom_labels=data['node_labels'],
            all_elements=data['element_list'],
            **retry_kwargs,
        )
    elif data['type'] == 'atomic_alignnd':
        graphs = atomic_alignnd(
            atoms=data['raw_data'],
            neighbor_params=data['neighbor_params'],
            dihedral=data['is_dihedral'],
            store_atoms=data['store_raw_data'],
            use_pt=data['use_pt'],
            include_angs=data['include_angs'],
            include_equivariant_fields=data['include_equivariant_fields'],
            atom_labels=data['node_labels'],
            all_elements=data['element_list'],
            store_atoms_type=data['store_atoms_type'],
            cpu_cores=data['cpu_cores'],
            **retry_kwargs,
        )
    elif data['type'] == 'atomic_alignnd_from_global_graph':
        # There is no cutoff/k to expand for this pathway because the input is
        # already a graph. We still pass validation controls so empty local
        # bond/angle graphs fail clearly instead of crashing later.
        graphs = atomic_alignnd_from_global_graph(
            global_graph=data['raw_data'],
            dihedral=data['is_dihedral'],
            store_atoms=data['store_raw_data'],
            include_angs=data['include_angs'],
            include_equivariant_fields=data['include_equivariant_fields'],
            store_atoms_type=data['store_atoms_type'],
            require_bonds=data['require_bonds'],
            require_angles=data['require_angles'],
            require_dihedrals=data['require_dihedrals'],
        )
    else:
        raise ValueError(f"Unknown graph generation type: {data['type']!r}")

    return graphs


# -----------------------------------------------------------------------------
# Shared retry / validation helpers
# -----------------------------------------------------------------------------

def _normalize_edge_index(edge_index, dtype=np.int64):
    """Return an edge-index array with canonical shape ``(2, N)``."""
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
        f"An edge index must have shape (2, N) or (N, 2); "
        f"received shape {edge_index.shape}."
    )


def _empty_edge_index(dtype=np.int64):
    return np.empty((2, 0), dtype=dtype)


def _hstack_edge_indices(edge_indices):
    nonempty = [
        _normalize_edge_index(edge_index, dtype=np.int64)
        for edge_index in edge_indices
        if _normalize_edge_index(edge_index, dtype=np.int64).shape[1] > 0
    ]
    if not nonempty:
        return _empty_edge_index(dtype=np.int64)
    return np.hstack(nonempty)


def _to_numpy_1d(values, dtype=np.float32):
    if values is None:
        return np.empty((0,), dtype=dtype)
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype).reshape(-1)


def _extract_numpy_graph_field(graph, key, dtype=None):
    value = graph[key]
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)




def _finalize_equivariant_graph_metadata(data):
    """Finalize PyG/equivariant metadata after constructing Atomic_Graph_Data.

    This is intentionally redundant with graph.py.  The Data container should
    already set num_nodes and handle batching, but setting it here as well makes
    every ALIGNN-generation pathway explicit and safer for equivariant models.

    The most important field is num_nodes.  PyG can infer this from a normal
    ``x`` field, but Catalyst atomic graphs use ``x_atm``/``x_1`` instead.
    Equivariant processors also consume ``z`` and ``pos`` directly, so setting
    num_nodes explicitly avoids ambiguous batching behavior.
    """
    z = getattr(data, "z", None)
    pos = getattr(data, "pos", None)
    x_atm = getattr(data, "x_atm", None)

    if z is not None:
        data.num_nodes = int(z.size(0))
    elif pos is not None:
        data.num_nodes = int(pos.size(0))
    elif x_atm is not None:
        data.num_nodes = int(x_atm.size(0))

    # If equivariant fields are present, ensure the PyG-standard edge_index is
    # present too.  This should already be handled by graph.py, but the alias is
    # useful for downstream equivariant processors and generic PyG utilities.
    if getattr(data, "edge_index", None) is None and getattr(data, "edge_index_G", None) is not None:
        data.edge_index = data.edge_index_G

    # Graph builders now preserve periodic image shifts even when optional
    # equivariant geometry is disabled.  The zero-shift fallback below is kept
    # only for legacy/nonperiodic graph objects assembled outside these builders.
    edge_index = getattr(data, "edge_index", None)
    if getattr(data, "shifts", None) is None and edge_index is not None:
        n_edges = int(edge_index.size(1))
        data.shifts = torch.zeros((n_edges, 3), dtype=torch.long, device=edge_index.device)

    return data

def _normalize_shifts(shifts, n_edges):
    if shifts is None:
        return np.zeros((int(n_edges), 3), dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    if shifts.shape[0] != int(n_edges):
        raise ValueError(
            "shifts and edge_index_G contain different numbers of edges: "
            f"{shifts.shape[0]} versus {n_edges}."
        )
    return shifts


def _deduplicate_bond_graph(edge_index_G, x_bnd, shifts=None):
    """Remove only exact periodic-image duplicates.

    Physical neighbors are identified by ``(src, dst, Sx, Sy, Sz)``.  Distinct
    periodic images sharing the same atom indices must remain separate edges.
    """
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)
    shifts = _normalize_shifts(shifts, edge_index_G.shape[1])

    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges before "
            f"duplicate removal: {edge_index_G.shape[1]} versus {len(x_bnd)}."
        )

    if edge_index_G.shape[1] == 0:
        return (
            _empty_edge_index(dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )

    seen = set()
    keep = []
    for edge_id, (src, dst) in enumerate(edge_index_G.T):
        shift = shifts[edge_id]
        key = (int(src), int(dst), int(shift[0]), int(shift[1]), int(shift[2]))
        if key not in seen:
            seen.add(key)
            keep.append(edge_id)

    keep = np.asarray(keep, dtype=np.int64)
    return (
        edge_index_G[:, keep],
        x_bnd[keep].astype(np.float32, copy=False),
        shifts[keep],
    )


def _edge_vectors_from_shifts(atoms, edge_index_G, shifts):
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    shifts = _normalize_shifts(shifts, edge_index_G.shape[1])
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64).reshape(3, 3)
    raw = positions[edge_index_G[1]] - positions[edge_index_G[0]]
    return raw + shifts @ cell


def _bond_angles_from_edge_vectors(edge_vec, edge_index_A):
    edge_index_A = _normalize_edge_index(edge_index_A, dtype=np.int64)
    if edge_index_A.shape[1] == 0:
        return np.empty((0,), dtype=np.float32)
    v1 = np.asarray(edge_vec[edge_index_A[0]], dtype=np.float64)
    v2 = np.asarray(edge_vec[edge_index_A[1]], dtype=np.float64)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    if np.any(denom <= 0):
        raise ValueError("Cannot compute a bond angle from a zero-length edge vector.")
    cosang = np.sum(v1 * v2, axis=1) / denom
    return np.arccos(np.clip(cosang, -1.0, 1.0)).astype(np.float32)


def _dihedral_graph_with_central(edge_index_G, shifts):
    """Return dihedral line-graph edges plus their central bond IDs.

    Reverse-bond exclusion is periodic-image aware, so a neighbor represented by
    the same atom index in a different image is not incorrectly discarded.
    """
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    shifts = _normalize_shifts(shifts, edge_index_G.shape[1])
    src, dst = edge_index_G
    angular_edges = []
    central_ids = []

    for central_id, (i, j) in enumerate(edge_index_G.T):
        central_shift = shifts[central_id]
        u_candidates = np.flatnonzero(dst == i)
        v_candidates = np.flatnonzero(dst == j)

        for u in u_candidates:
            is_reverse_central = (
                src[u] == j and np.array_equal(shifts[u], -central_shift)
            )
            if is_reverse_central:
                continue
            for v in v_candidates:
                is_central = (
                    src[v] == i and np.array_equal(shifts[v], central_shift)
                )
                if is_central:
                    continue
                angular_edges.append((int(u), int(v)))
                central_ids.append(int(central_id))

    if not angular_edges:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.asarray(angular_edges, dtype=np.int64).T, np.asarray(central_ids, dtype=np.int64)


def _dihedral_angles_from_edge_vectors(edge_vec, edge_index_dih, central_edge_ids):
    edge_index_dih = _normalize_edge_index(edge_index_dih, dtype=np.int64)
    central_edge_ids = np.asarray(central_edge_ids, dtype=np.int64).reshape(-1)
    if edge_index_dih.shape[1] == 0:
        return np.empty((0,), dtype=np.float32)
    if edge_index_dih.shape[1] != len(central_edge_ids):
        raise ValueError("Dihedral edge list and central-edge IDs have different lengths.")

    values = []
    for (u, v), central_id in zip(edge_index_dih.T, central_edge_ids):
        # Set central atom i at the origin.  For incoming edge a->i,
        # p_a=-edge_vec[u].  For central i->j, p_j=edge_vec[c].  For
        # incoming b->j, p_b=p_j-edge_vec[v].
        p0 = -np.asarray(edge_vec[int(u)], dtype=np.float64)
        p1 = np.zeros(3, dtype=np.float64)
        p2 = np.asarray(edge_vec[int(central_id)], dtype=np.float64)
        p3 = p2 - np.asarray(edge_vec[int(v)], dtype=np.float64)

        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = np.linalg.norm(b1)
        if b1_norm <= 0:
            raise ValueError("Cannot compute a dihedral from a zero-length central bond.")
        b1u = b1 / b1_norm
        vproj = b0 - np.dot(b0, b1u) * b1u
        wproj = b2 - np.dot(b2, b1u) * b1u
        nv = np.linalg.norm(vproj)
        nw = np.linalg.norm(wproj)
        if nv <= 1e-14 or nw <= 1e-14:
            # Collinear terminal bond: use a defined zero torsion rather than NaN.
            values.append(0.0)
            continue
        x = np.dot(vproj, wproj)
        y = np.dot(np.cross(b1u, vproj), wproj)
        angle = np.arctan2(y, x) % (2.0 * np.pi)
        values.append(angle)
    return np.asarray(values, dtype=np.float32)


def _increase_graph_search_params(cutoff, k, *, k_step, cutoff_scale, max_k, max_cutoff):
    """Increase k first when possible, then increase cutoff."""
    changed = False

    if k is not None and k > 0:
        if max_k is None or k < max_k:
            k = min(k + k_step, max_k) if max_k is not None else k + k_step
            changed = True

    if not changed:
        new_cutoff = cutoff * cutoff_scale
        if max_cutoff is not None:
            new_cutoff = min(new_cutoff, max_cutoff)

        if new_cutoff > cutoff:
            cutoff = new_cutoff
            changed = True

    return cutoff, k, changed


def _build_angular_components(atoms, edge_index_G, include_angs, dihedral,
                              require_angles, require_dihedrals, edge_shifts=None):
    if not include_angs:
        return dict(
            edge_index_A=None,
            x_ang=None,
            mask_dih_ang=None,
            x_bnd_ang=np.empty((0,), dtype=np.float32),
            x_dih_ang=np.empty((0,), dtype=np.float32),
        )

    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    edge_shifts = _normalize_shifts(edge_shifts, edge_index_G.shape[1])
    edge_vec = _edge_vectors_from_shifts(atoms, edge_index_G, edge_shifts)

    edge_index_bnd_ang = _normalize_edge_index(line_graph(edge_index_G), dtype=np.int64)
    if require_angles and edge_index_bnd_ang.shape[1] == 0:
        raise RuntimeError("No bond-angle edges were created.")
    x_bnd_ang = _bond_angles_from_edge_vectors(edge_vec, edge_index_bnd_ang)

    if dihedral:
        edge_index_dih_ang, central_edge_ids = _dihedral_graph_with_central(
            edge_index_G, edge_shifts
        )
        if require_dihedrals and edge_index_dih_ang.shape[1] == 0:
            raise RuntimeError("No dihedral-angle edges were created.")
        x_dih_ang = _dihedral_angles_from_edge_vectors(
            edge_vec, edge_index_dih_ang, central_edge_ids
        )
        edge_index_A = _hstack_edge_indices([edge_index_bnd_ang, edge_index_dih_ang])
        x_ang = np.concatenate((x_bnd_ang, x_dih_ang)).astype(np.float32, copy=False)
        mask_dih_ang = np.concatenate((
            np.zeros(len(x_bnd_ang), dtype=bool),
            np.ones(len(x_dih_ang), dtype=bool),
        ))
    else:
        edge_index_A = edge_index_bnd_ang
        x_ang = x_bnd_ang.astype(np.float32, copy=False)
        x_dih_ang = np.empty((0,), dtype=np.float32)
        mask_dih_ang = np.zeros(len(x_ang), dtype=bool)

    if edge_index_A.shape[1] != len(x_ang):
        raise ValueError(
            "edge_index_A and x_ang contain different numbers of edges/values: "
            f"{edge_index_A.shape[1]} versus {len(x_ang)}."
        )
    return dict(
        edge_index_A=edge_index_A,
        x_ang=x_ang,
        mask_dih_ang=mask_dih_ang,
        x_bnd_ang=x_bnd_ang,
        x_dih_ang=x_dih_ang,
    )


def _build_graph_components_once(atoms, cutoff, k, include_angs, dihedral,
                                 require_bonds, require_angles, require_dihedrals,
                                 include_equivariant_fields=True):
    edge_index_G, x_bnd, edge_shifts = atoms2graph(
        atoms, cutoff=cutoff, k=k, return_shifts=True
    )
    edge_index_G, x_bnd, edge_shifts = _deduplicate_bond_graph(
        edge_index_G, x_bnd, edge_shifts
    )

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError(f"No bonds were created with cutoff={cutoff:.6g}, k={k}.")

    angular = _build_angular_components(
        atoms,
        edge_index_G,
        include_angs=include_angs,
        dihedral=dihedral,
        require_angles=bool(require_angles and include_angs),
        require_dihedrals=bool(require_dihedrals and include_angs and dihedral),
        edge_shifts=edge_shifts,
    )

    result = dict(
        edge_index_G=edge_index_G,
        x_bnd=x_bnd,
        cutoff_used=cutoff,
        k_used=k,
        edge_shifts=edge_shifts,
    )
    result.update(angular)

    if include_equivariant_fields:
        result['equivariant_fields'] = build_equivariant_atomic_fields(
            atoms,
            edge_index_G,
            dtype=np.float32,
            include_edge_geometry=True,
            shifts=edge_shifts,
        )
    else:
        result['equivariant_fields'] = {}

    return result


def _build_graph_components_with_retry(atoms, neighbor_params, include_angs, dihedral,
                                       *, auto_retry_graph=True,
                                       max_graph_attempts=8,
                                       k_step=4,
                                       cutoff_scale=1.25,
                                       max_k=None,
                                       max_cutoff=None,
                                       require_bonds=True,
                                       require_angles=True,
                                       require_dihedrals=False,
                                       retry_verbose=True,
                                       context='graph',
                                       include_equivariant_fields=True):
    if len(neighbor_params) < 2:
        raise ValueError("neighbor_params must contain [cutoff, k].")

    cutoff = float(neighbor_params[0])
    k = int(neighbor_params[1])

    if max_graph_attempts < 1:
        raise ValueError("max_graph_attempts must be >= 1.")

    if max_k is None and k > 0 and not np.any(np.asarray(atoms.pbc, dtype=bool)):
        max_k = max(1, len(atoms) - 1)

    last_error = None
    for attempt in range(max_graph_attempts):
        try:
            components = _build_graph_components_once(
                atoms,
                cutoff=cutoff,
                k=k,
                include_angs=include_angs,
                dihedral=dihedral,
                require_bonds=require_bonds,
                require_angles=require_angles,
                require_dihedrals=require_dihedrals,
                include_equivariant_fields=include_equivariant_fields,
            )
            components['graph_build_attempts'] = attempt + 1
            if retry_verbose and attempt > 0:
                print(
                    f"{context}: graph construction succeeded after {attempt + 1} "
                    f"attempts using cutoff={cutoff:.6g}, k={k}."
                )
            return components
        except Exception as exc:
            last_error = exc
            if not auto_retry_graph or attempt == max_graph_attempts - 1:
                break

            old_cutoff = cutoff
            old_k = k
            cutoff, k, changed = _increase_graph_search_params(
                cutoff,
                k,
                k_step=k_step,
                cutoff_scale=cutoff_scale,
                max_k=max_k,
                max_cutoff=max_cutoff,
            )
            if not changed:
                break

            if retry_verbose:
                print(
                    f"{context}: graph construction failed with "
                    f"cutoff={old_cutoff:.6g}, k={old_k}. Retrying with "
                    f"cutoff={cutoff:.6g}, k={k}. Reason: {exc}"
                )

    raise RuntimeError(
        f"{context}: failed to construct a valid graph after "
        f"{max_graph_attempts} attempts. Final cutoff={cutoff:.6g}, k={k}. "
        f"Last error was: {last_error}"
    )


def _build_atom_features_from_basis(atoms, elems_array, atom_labels=''):
    atomic_numbers = atoms.get_atomic_numbers()
    elem_to_idx = {int(el): i for i, el in enumerate(elems_array)}
    x_atm = np.zeros((len(atomic_numbers), len(elems_array)), dtype=np.float32)

    for atom_idx, atom_number in enumerate(atomic_numbers):
        feature_idx = elem_to_idx.get(int(atom_number))
        if feature_idx is not None:
            x_atm[atom_idx, feature_idx] = (
                float(atom_number) if atom_labels == 'atomic_number' else 1.0
            )

    return x_atm


def _element_basis_for_atoms(atoms, use_pt=False):
    if use_pt:
        pdb = Physics_data()
        return np.array([el.number for el in pdb.elements], dtype=np.int64)
    return np.sort(np.unique(atoms.get_atomic_numbers()), axis=None).astype(np.int64)


def _element_basis_for_structures(structures, use_pt=False):
    if use_pt:
        pdb = Physics_data()
        return np.array([el.number for el in pdb.elements], dtype=np.int64)

    all_numbers = []
    for atoms in structures:
        all_numbers.extend(atoms.get_atomic_numbers().tolist())
    return np.sort(np.unique(np.asarray(all_numbers, dtype=np.int64)), axis=None)


# -----------------------------------------------------------------------------
# Global graph builders
# -----------------------------------------------------------------------------

def alignnd(atoms, neighbor_params, dihedral=False, store_atoms=False, use_pt=False,
            include_angs=True, include_equivariant_fields=True, atom_labels='', all_elements=None, *, auto_retry_graph=True,
            max_graph_attempts=8, k_step=4, cutoff_scale=1.25, max_k=None,
            max_cutoff=None, require_bonds=True, require_angles=True,
            require_dihedrals=False, retry_verbose=True):
    """Convert one ASE Atoms object into an ALIGNN-style Atomic_Graph_Data.

    If requested bonds/angles/dihedrals cannot be created, this function can
    automatically increase ``k`` and then ``cutoff`` for this structure only.
    """
    atms = atoms if store_atoms else None
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts['x_dih_ang'] = []

    if use_pt:
        elems_array = _element_basis_for_atoms(atoms, use_pt=True)
    else:
        elems_array = _resolve_element_numbers(all_elements, atoms.get_atomic_numbers())
    x_atm = _build_atom_features_from_basis(atoms, elems_array, atom_labels=atom_labels)

    components = _build_graph_components_with_retry(
        atoms,
        neighbor_params,
        include_angs=include_angs,
        dihedral=dihedral,
        auto_retry_graph=auto_retry_graph,
        max_graph_attempts=max_graph_attempts,
        k_step=k_step,
        cutoff_scale=cutoff_scale,
        max_k=max_k,
        max_cutoff=max_cutoff,
        require_bonds=require_bonds,
        require_angles=require_angles,
        require_dihedrals=require_dihedrals,
        retry_verbose=retry_verbose,
        context='alignnd',
        include_equivariant_fields=include_equivariant_fields,
    )

    edge_index_G_tensor = torch.tensor(components['edge_index_G'], dtype=torch.long)
    x_atm_tensor = torch.tensor(x_atm, dtype=torch.float)
    x_bnd_tensor = torch.tensor(components['x_bnd'], dtype=torch.float)
    graph_storage_kwargs = dict(components.get('equivariant_fields', {}))
    graph_storage_kwargs.setdefault(
        'shifts', torch.tensor(components['edge_shifts'], dtype=torch.long)
    )

    if include_angs:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=edge_index_G_tensor,
            edge_index_A=torch.tensor(components['edge_index_A'], dtype=torch.long),
            x_atm=x_atm_tensor,
            x_bnd=x_bnd_tensor,
            x_ang=torch.tensor(components['x_ang'], dtype=torch.float),
            mask_dih_ang=torch.tensor(components['mask_dih_ang'], dtype=torch.bool),
            **graph_storage_kwargs,
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
        )
    else:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=edge_index_G_tensor,
            edge_index_A=None,
            x_atm=x_atm_tensor,
            x_bnd=x_bnd_tensor,
            x_ang=None,
            mask_dih_ang=None,
            **graph_storage_kwargs,
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=None,
        )

    data = _finalize_equivariant_graph_metadata(data)
    data.graph_cutoff_used = components['cutoff_used']
    data.graph_k_used = components['k_used']
    data.graph_build_attempts = components['graph_build_attempts']
    data.generate_gid()
    return data


def realignnd(structures, neighbor_params, dihedral=False, store_atoms=False,
              use_pt=False, include_angs=True, include_equivariant_fields=True, atom_labels='', all_elements=None, *,
              auto_retry_graph=True, max_graph_attempts=8, k_step=4,
              cutoff_scale=1.25, max_k=None, max_cutoff=None,
              require_bonds=True, require_angles=True, require_dihedrals=False,
              retry_verbose=True):
    """Convert multiple ASE structures into one concatenated Atomic_Graph_Data.

    Retry is applied independently to each structure, so only the failing
    structure receives an expanded cutoff/k.
    """
    structures = list(structures)
    if len(structures) == 0:
        raise ValueError("realignnd requires at least one structure.")

    atms = [] if store_atoms else None
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts['x_dih_ang'] = []

    if use_pt:
        elems_array = _element_basis_for_structures(structures, use_pt=True)
    else:
        present = np.concatenate([atoms.get_atomic_numbers() for atoms in structures])
        elems_array = _resolve_element_numbers(all_elements, present)

    f_edge_index_G = []
    f_edge_index_A = []
    f_x_atm = []
    f_x_bnd = []
    f_x_ang = []
    f_mask_dih_ang = []
    f_edge_shifts = []
    f_eq_z = []
    f_eq_pos = []
    f_eq_edge_index = []
    f_eq_edge_vec = []
    f_eq_edge_dist = []
    f_eq_cell = []
    f_eq_pbc = []
    f_eq_atom_graph_batch = []
    f_eq_edge_graph_batch = []
    f_eq_global_atom_indices = []

    atom_offset = 0
    bond_offset = 0
    atom_total = 0
    bond_total = 0
    angle_total = 0
    dihedral_total = 0
    cutoffs_used = []
    ks_used = []
    attempts_used = []

    for structure_index, atoms in enumerate(structures):
        x_atm = _build_atom_features_from_basis(atoms, elems_array, atom_labels=atom_labels)
        components = _build_graph_components_with_retry(
            atoms,
            neighbor_params,
            include_angs=include_angs,
            dihedral=dihedral,
            auto_retry_graph=auto_retry_graph,
            max_graph_attempts=max_graph_attempts,
            k_step=k_step,
            cutoff_scale=cutoff_scale,
            max_k=max_k,
            max_cutoff=max_cutoff,
            require_bonds=require_bonds,
            require_angles=require_angles,
            require_dihedrals=require_dihedrals,
            retry_verbose=retry_verbose,
            context=f'realignnd structure {structure_index}',
            include_equivariant_fields=include_equivariant_fields,
        )

        edge_index_G = components['edge_index_G'] + atom_offset
        f_edge_index_G.append(edge_index_G)
        f_x_atm.append(x_atm)
        f_x_bnd.append(components['x_bnd'])
        f_edge_shifts.append(torch.tensor(components['edge_shifts'], dtype=torch.long))

        if include_equivariant_fields:
            eq = components.get('equivariant_fields', {})
            if eq:
                n_local_atoms = int(eq['pos'].size(0))
                n_local_edges = int(eq['edge_index'].size(1))
                f_eq_z.append(eq['z'])
                f_eq_pos.append(eq['pos'])
                f_eq_edge_index.append(eq['edge_index'] + atom_offset)
                f_eq_edge_vec.append(eq['edge_vec'])
                f_eq_edge_dist.append(eq['edge_dist'])
                f_eq_cell.append(eq['cell'])
                f_eq_pbc.append(eq['pbc'])
                if 'global_atom_indices' in eq and eq['global_atom_indices'] is not None:
                    f_eq_global_atom_indices.append(eq['global_atom_indices'] + atom_offset)
                else:
                    f_eq_global_atom_indices.append(torch.arange(n_local_atoms, dtype=torch.long) + atom_offset)
                f_eq_atom_graph_batch.append(torch.full((n_local_atoms,), structure_index, dtype=torch.long))
                f_eq_edge_graph_batch.append(torch.full((n_local_edges,), structure_index, dtype=torch.long))

        if include_angs:
            edge_index_A = components['edge_index_A'] + bond_offset
            f_edge_index_A.append(edge_index_A)
            f_x_ang.append(components['x_ang'])
            f_mask_dih_ang.append(components['mask_dih_ang'])

        atom_total += len(x_atm)
        bond_total += len(components['x_bnd'])
        data_amounts['x_atm'].append(atom_total)
        data_amounts['x_bnd'].append(bond_total)

        if include_angs:
            angle_total += len(components['x_ang'])
            data_amounts['x_ang'].append(angle_total)
            if dihedral:
                dihedral_total += int(np.asarray(components['mask_dih_ang']).sum())
                data_amounts['x_dih_ang'].append(dihedral_total)

        if store_atoms:
            atms.append(atoms)

        atom_offset += len(x_atm)
        bond_offset += len(components['x_bnd'])
        cutoffs_used.append(components['cutoff_used'])
        ks_used.append(components['k_used'])
        attempts_used.append(components['graph_build_attempts'])

    final_edge_index_G = np.hstack(f_edge_index_G) if f_edge_index_G else _empty_edge_index()
    final_x_atm = np.concatenate(f_x_atm, axis=0) if f_x_atm else np.empty((0, len(elems_array)), dtype=np.float32)
    final_x_bnd = np.concatenate(f_x_bnd) if f_x_bnd else np.empty((0,), dtype=np.float32)

    final_equivariant_kwargs = {
        'shifts': torch.cat(f_edge_shifts, dim=0) if f_edge_shifts else torch.empty((0, 3), dtype=torch.long)
    }
    if include_equivariant_fields and f_eq_pos:
        final_equivariant_kwargs.update(
            z=torch.cat(f_eq_z, dim=0),
            pos=torch.cat(f_eq_pos, dim=0),
            edge_index=torch.cat(f_eq_edge_index, dim=1),
            edge_vec=torch.cat(f_eq_edge_vec, dim=0),
            edge_dist=torch.cat(f_eq_edge_dist, dim=0),
            cell=torch.stack(f_eq_cell, dim=0),
            pbc=torch.stack(f_eq_pbc, dim=0),
            global_atom_indices=torch.cat(f_eq_global_atom_indices, dim=0),
            atom_graph_batch=torch.cat(f_eq_atom_graph_batch, dim=0),
            edge_graph_batch=torch.cat(f_eq_edge_graph_batch, dim=0),
        )

    if include_angs:
        final_edge_index_A = np.hstack(f_edge_index_A) if f_edge_index_A else _empty_edge_index()
        final_x_ang = np.concatenate(f_x_ang) if f_x_ang else np.empty((0,), dtype=np.float32)
        final_mask_dih_ang = np.concatenate(f_mask_dih_ang) if f_mask_dih_ang else np.empty((0,), dtype=bool)
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=torch.tensor(final_edge_index_G, dtype=torch.long),
            edge_index_A=torch.tensor(final_edge_index_A, dtype=torch.long),
            x_atm=torch.tensor(final_x_atm, dtype=torch.float),
            x_bnd=torch.tensor(final_x_bnd, dtype=torch.float),
            x_ang=torch.tensor(final_x_ang, dtype=torch.float),
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
            mask_dih_ang=torch.tensor(final_mask_dih_ang, dtype=torch.bool),
            **final_equivariant_kwargs,
        )
    else:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=torch.tensor(final_edge_index_G, dtype=torch.long),
            edge_index_A=None,
            x_atm=torch.tensor(final_x_atm, dtype=torch.float),
            x_bnd=torch.tensor(final_x_bnd, dtype=torch.float),
            x_ang=None,
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=None,
            mask_dih_ang=None,
            **final_equivariant_kwargs,
        )

    data = _finalize_equivariant_graph_metadata(data)
    data.graph_cutoffs_used = cutoffs_used
    data.graph_ks_used = ks_used
    data.graph_build_attempts = attempts_used
    data.generate_gid()
    return data


# -----------------------------------------------------------------------------
# Atomic/local graph helpers
# -----------------------------------------------------------------------------

def _make_bidirectional_graph(edge_index_G, x_bnd, shifts=None):
    """Create both directions for each physical periodic-image bond."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)
    shifts = _normalize_shifts(shifts, edge_index_G.shape[1])
    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges: "
            f"{edge_index_G.shape[1]} versus {len(x_bnd)}."
        )
    if edge_index_G.shape[1] == 0:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0, 3), dtype=np.int64)

    physical = {}
    for edge_id, (src, dst) in enumerate(edge_index_G.T):
        shift = shifts[edge_id]
        forward = (int(src), int(dst), int(shift[0]), int(shift[1]), int(shift[2]))
        reverse = (int(dst), int(src), int(-shift[0]), int(-shift[1]), int(-shift[2]))
        key = min(forward, reverse)
        physical.setdefault(key, float(x_bnd[edge_id]))

    records = {}
    for key, bond_value in physical.items():
        src, dst, sx, sy, sz = key
        shift = np.asarray([sx, sy, sz], dtype=np.int64)
        records[(src, dst, sx, sy, sz)] = bond_value
        records[(dst, src, -sx, -sy, -sz)] = bond_value

    ordered = sorted(records)
    edge_index = np.asarray([[r[0] for r in ordered], [r[1] for r in ordered]], dtype=np.int64)
    out_shifts = np.asarray([[r[2], r[3], r[4]] for r in ordered], dtype=np.int64)
    out_bonds = np.asarray([records[r] for r in ordered], dtype=np.float32)
    return edge_index, out_bonds, out_shifts


def build_adjacency_csr(edge_index_G, x_bnd, len_atoms, shifts=None):
    """Build CSR adjacency arrays while retaining periodic image shifts."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)
    shifts = _normalize_shifts(shifts, edge_index_G.shape[1])
    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges: "
            f"{edge_index_G.shape[1]} versus {len(x_bnd)}."
        )
    n_atoms = int(len_atoms)
    if n_atoms < 0:
        raise ValueError("len_atoms must be non-negative.")
    if edge_index_G.shape[1] == 0:
        return (
            np.zeros(n_atoms + 1, dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )
    if edge_index_G.min() < 0 or edge_index_G.max() >= n_atoms:
        raise ValueError(
            "The global edge index contains an atom ID outside the ASE Atoms "
            f"range [0, {n_atoms - 1}]."
        )
    order = np.lexsort((shifts[:,2], shifts[:,1], shifts[:,0], edge_index_G[1], edge_index_G[0]))
    src = edge_index_G[0][order]
    dst = edge_index_G[1][order]
    bond_values = x_bnd[order]
    shift_values = shifts[order]
    counts = np.bincount(src, minlength=n_atoms)
    indptr = np.zeros(n_atoms + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)
    return indptr, np.asarray(dst, dtype=np.int64), np.asarray(bond_values, dtype=np.float32), np.asarray(shift_values, dtype=np.int64)


def _csr_neighbors(atom_index, indptr, indices, bond_data, shift_data):
    start = int(indptr[atom_index])
    end = int(indptr[atom_index + 1])
    return indices[start:end], bond_data[start:end], shift_data[start:end]


def _build_atomic_local_graph(center_atom, indptr, indices, bond_data,
                              shift_data, include_dihedral_shell):
    """Build a local graph without collapsing distinct periodic images."""
    local_edges = {}

    def add_directed_edge(src, dst, bond_value, shift):
        shift = np.asarray(shift, dtype=np.int64).reshape(3)
        key = (int(src), int(dst), int(shift[0]), int(shift[1]), int(shift[2]))
        local_edges.setdefault(key, float(bond_value))

    def add_bidirectional_bond(atom_a, atom_b, bond_value, shift):
        shift = np.asarray(shift, dtype=np.int64).reshape(3)
        add_directed_edge(atom_a, atom_b, bond_value, shift)
        add_directed_edge(atom_b, atom_a, bond_value, -shift)

    first_neighbors, first_bonds, first_shifts = _csr_neighbors(
        center_atom, indptr, indices, bond_data, shift_data
    )
    for neighbor, bond_value, shift in zip(first_neighbors, first_bonds, first_shifts):
        add_bidirectional_bond(center_atom, neighbor, bond_value, shift)

    if include_dihedral_shell:
        for first_neighbor in np.unique(first_neighbors):
            second_neighbors, second_bonds, second_shifts = _csr_neighbors(
                int(first_neighbor), indptr, indices, bond_data, shift_data
            )
            for second_neighbor, bond_value, shift in zip(second_neighbors, second_bonds, second_shifts):
                add_bidirectional_bond(first_neighbor, second_neighbor, bond_value, shift)

    if not local_edges:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0, 3), dtype=np.int64)

    ordered = list(local_edges)
    return (
        np.asarray([[key[0] for key in ordered], [key[1] for key in ordered]], dtype=np.int64),
        np.asarray([local_edges[key] for key in ordered], dtype=np.float32),
        np.asarray([[key[2], key[3], key[4]] for key in ordered], dtype=np.int64),
    )


def _filter_center_bond_angles(edge_index_G, edge_index_A, center_atom):
    """Keep bond angles whose shared central atom is ``center_atom``."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    edge_index_A = _normalize_edge_index(edge_index_A, dtype=np.int64)
    if edge_index_A.shape[1] == 0:
        return edge_index_A

    src_G, dst_G = edge_index_G
    u = edge_index_A[0]
    v = edge_index_A[1]

    valid_indices = (
        (u >= 0) & (u < edge_index_G.shape[1]) &
        (v >= 0) & (v < edge_index_G.shape[1])
    )
    if not np.all(valid_indices):
        raise ValueError("line_graph returned a bond index outside edge_index_G.")

    keep = (
        (dst_G[u] == int(center_atom)) &
        (dst_G[v] == int(center_atom)) &
        (u != v)
    )
    return edge_index_A[:, keep]


def _filter_center_dihedrals(edge_index_G, edge_index_A, center_atom):
    """Keep proper dihedrals with ``center_atom`` on the central bond."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    edge_index_A = _normalize_edge_index(edge_index_A, dtype=np.int64)
    if edge_index_A.shape[1] == 0:
        return edge_index_A

    src_G, dst_G = edge_index_G
    u = edge_index_A[0]
    v = edge_index_A[1]

    valid_indices = (
        (u >= 0) & (u < edge_index_G.shape[1]) &
        (v >= 0) & (v < edge_index_G.shape[1])
    )
    if not np.all(valid_indices):
        raise ValueError("dihedral_graph returned a bond index outside edge_index_G.")

    atom_a = src_G[u]
    atom_b = dst_G[u]
    atom_c = dst_G[v]
    atom_d = src_G[v]

    center_is_on_central_bond = (
        (atom_b == int(center_atom)) | (atom_c == int(center_atom))
    )
    return edge_index_A[:, center_is_on_central_bond]


def _ordered_local_atom_indices(tmp_edge_index_G, center_atom):
    ordered_indices = [int(center_atom)]
    seen = {int(center_atom)}

    if tmp_edge_index_G.shape[1] > 0:
        for src, dst in tmp_edge_index_G.T:
            for atom_index in (int(src), int(dst)):
                if atom_index not in seen:
                    seen.add(atom_index)
                    ordered_indices.append(atom_index)

    return np.asarray(ordered_indices, dtype=np.int64)


def build_x_atm(tmp_edge_index_G, atoms, elems_array, atom_labels,
                center_atom_index=None):
    """Build atomic features while retaining global atom IDs in the graph."""
    if center_atom_index is None:
        if tmp_edge_index_G.shape[1] == 0:
            atom_indices = np.empty((0,), dtype=np.int64)
        else:
            flattened = np.concatenate((tmp_edge_index_G[0], tmp_edge_index_G[1]))
            atom_indices, first_occurrence = np.unique(flattened, return_index=True)
            atom_indices = atom_indices[np.argsort(first_occurrence)]
    else:
        atom_indices = _ordered_local_atom_indices(tmp_edge_index_G, center_atom_index)

    atomic_numbers = atoms.get_atomic_numbers()
    local_atomic_numbers = atomic_numbers[atom_indices]

    elem_to_idx = {int(elem): idx for idx, elem in enumerate(elems_array)}
    x_atm = np.zeros((len(local_atomic_numbers), len(elems_array)), dtype=np.float32)

    for local_i, atomic_number in enumerate(local_atomic_numbers):
        feature_index = elem_to_idx.get(int(atomic_number))
        if feature_index is None:
            continue
        x_atm[local_i, feature_index] = (
            float(atomic_number) if atom_labels == 'atomic_number' else 1.0
        )

    return x_atm


def _build_local_angles(atoms, tmp_edge_index_G, tmp_edge_shifts, center_atom,
                        include_angs, dihedral, require_angles=False,
                        require_dihedrals=False):
    if not include_angs:
        return dict(edge_index_A=None, x_ang=None, mask_dih_ang=None,
                    x_bnd_ang=np.empty((0,), dtype=np.float32),
                    x_dih_ang=np.empty((0,), dtype=np.float32))

    edge_vec = _edge_vectors_from_shifts(atoms, tmp_edge_index_G, tmp_edge_shifts)
    raw_bond_angle_edges = _normalize_edge_index(line_graph(tmp_edge_index_G), dtype=np.int64)
    edge_index_bnd_ang = _filter_center_bond_angles(
        tmp_edge_index_G, raw_bond_angle_edges, center_atom
    )
    if require_angles and edge_index_bnd_ang.shape[1] == 0:
        raise RuntimeError(f"No centered bond angles were created for atom {center_atom}.")
    x_bnd_ang = _bond_angles_from_edge_vectors(edge_vec, edge_index_bnd_ang)

    if dihedral:
        raw_dihedral_edges, raw_central_ids = _dihedral_graph_with_central(
            tmp_edge_index_G, tmp_edge_shifts
        )
        if raw_dihedral_edges.shape[1] > 0:
            keep_mask = np.zeros(raw_dihedral_edges.shape[1], dtype=bool)
            filtered = _filter_center_dihedrals(
                tmp_edge_index_G, raw_dihedral_edges, center_atom
            )
            # Preserve association with central edge IDs using tuple occurrence
            # counts; edge-pair duplicates can legitimately exist for different
            # central periodic images.
            wanted = {}
            for pair in filtered.T:
                key = tuple(map(int, pair))
                wanted[key] = wanted.get(key, 0) + 1
            for idx, pair in enumerate(raw_dihedral_edges.T):
                key = tuple(map(int, pair))
                if wanted.get(key, 0) > 0:
                    keep_mask[idx] = True
                    wanted[key] -= 1
            edge_index_dih_ang = raw_dihedral_edges[:, keep_mask]
            central_ids = raw_central_ids[keep_mask]
        else:
            edge_index_dih_ang = raw_dihedral_edges
            central_ids = raw_central_ids

        if require_dihedrals and edge_index_dih_ang.shape[1] == 0:
            raise RuntimeError(f"No centered dihedrals were created for atom {center_atom}.")
        x_dih_ang = _dihedral_angles_from_edge_vectors(
            edge_vec, edge_index_dih_ang, central_ids
        )
        edge_index_A = _hstack_edge_indices((edge_index_bnd_ang, edge_index_dih_ang))
        x_ang = np.concatenate((x_bnd_ang, x_dih_ang)).astype(np.float32, copy=False)
        mask_dih_ang = np.concatenate((np.zeros(len(x_bnd_ang), dtype=bool), np.ones(len(x_dih_ang), dtype=bool)))
    else:
        edge_index_A = edge_index_bnd_ang
        x_ang = x_bnd_ang.astype(np.float32, copy=False)
        x_dih_ang = np.empty((0,), dtype=np.float32)
        mask_dih_ang = np.zeros(len(x_ang), dtype=bool)

    return dict(edge_index_A=edge_index_A, x_ang=x_ang,
                mask_dih_ang=mask_dih_ang, x_bnd_ang=x_bnd_ang,
                x_dih_ang=x_dih_ang)


def process_atom(i, indptr, indices, data, shift_data, atoms, elems_array, atms,
                 include_angs, dihedral, store_atoms, store_atoms_type,
                 atom_labels, require_bonds=False, require_angles=False,
                 require_dihedrals=False, include_equivariant_fields=True):
    """Build one atomic graph, including a two-hop shell when needed."""
    if store_atoms:
        stored_atoms = atoms if store_atoms_type == 'ase-atoms' else atoms[i]
    else:
        stored_atoms = None

    tmp_edge_index_G, tmp_x_bnd, tmp_edge_shifts = _build_atomic_local_graph(
        center_atom=i,
        indptr=indptr,
        indices=indices,
        bond_data=data,
        shift_data=shift_data,
        include_dihedral_shell=bool(include_angs and dihedral),
    )

    if require_bonds and tmp_edge_index_G.shape[1] == 0:
        raise RuntimeError(f"No local bonds were created for atom {i}.")

    x_atm = build_x_atm(
        tmp_edge_index_G,
        atoms,
        elems_array,
        atom_labels,
        center_atom_index=i,
    )

    equivariant_kwargs = {'shifts': torch.tensor(tmp_edge_shifts, dtype=torch.long)}
    if include_equivariant_fields:
        local_atom_indices = _ordered_local_atom_indices(tmp_edge_index_G, i)
        equivariant_kwargs = build_equivariant_atomic_fields(
            atoms,
            tmp_edge_index_G,
            atom_indices=local_atom_indices,
            dtype=np.float32,
            include_edge_geometry=True,
            shifts=tmp_edge_shifts,
        )

    if include_angs:
        local_angles = _build_local_angles(
            atoms,
            tmp_edge_index_G,
            tmp_edge_shifts,
            center_atom=i,
            include_angs=include_angs,
            dihedral=dihedral,
            require_angles=require_angles,
            require_dihedrals=require_dihedrals,
        )
        edge_index_A = local_angles['edge_index_A']
        x_ang = local_angles['x_ang']
        mask_dih_ang = local_angles['mask_dih_ang']
        x_dih_ang = local_angles['x_dih_ang']

        local_data_amounts = dict(
            x_bnd=[len(tmp_x_bnd) - 1],
            x_atm=[len(x_atm) - 1],
            x_ang=[len(x_ang) - 1],
            x_dih_ang=[len(x_dih_ang) - 1] if dihedral else [],
        )

        data_obj = Atomic_Graph_Data(
            atoms=stored_atoms,
            edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
            edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
            x_atm=torch.tensor(x_atm, dtype=torch.float),
            x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
            x_ang=torch.tensor(x_ang, dtype=torch.float),
            atm_amounts=torch.tensor(local_data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(local_data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=torch.tensor(local_data_amounts['x_ang'], dtype=torch.long),
            mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool),
            **equivariant_kwargs,
        )
    else:
        local_data_amounts = dict(
            x_bnd=[len(tmp_x_bnd) - 1],
            x_atm=[len(x_atm) - 1],
            x_ang=[],
            x_dih_ang=[],
        )

        data_obj = Atomic_Graph_Data(
            atoms=stored_atoms,
            edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
            edge_index_A=None,
            x_atm=torch.tensor(x_atm, dtype=torch.float),
            x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
            x_ang=None,
            atm_amounts=torch.tensor(local_data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(local_data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=None,
            mask_dih_ang=None,
            **equivariant_kwargs,
        )

    data_obj = _finalize_equivariant_graph_metadata(data_obj)
    data_obj.generate_gid()
    return i, data_obj, local_data_amounts


def _resolve_element_numbers(all_elements, atom_numbers):
    """Return the atomic-number feature basis for one dataset.

    When ``all_elements``/``element_list`` is supplied it is authoritative and
    its order defines the one-hot channel order.  This prevents separately
    generated structures from silently changing either feature width or the
    physical meaning of a channel.  Structures containing an element outside
    the declared basis fail early with a clear error.

    When no explicit basis is supplied, the legacy deterministic behavior is
    retained by using the sorted elements present in the structure(s).
    """
    present = np.asarray(np.unique(atom_numbers), dtype=np.int64)

    if all_elements is None or len(all_elements) == 0:
        return np.sort(present)

    from ase.data import atomic_numbers as ase_atomic_numbers

    resolved = []
    seen = set()
    for element in all_elements:
        if isinstance(element, str):
            stripped = element.strip()
            if stripped.isdigit():
                number = int(stripped)
            else:
                if stripped not in ase_atomic_numbers:
                    raise ValueError(f"Unknown element symbol in element_list: {element!r}")
                number = int(ase_atomic_numbers[stripped])
        else:
            number = int(element)
        if number not in seen:
            seen.add(number)
            resolved.append(number)

    basis = np.asarray(resolved, dtype=np.int64)
    missing = sorted(set(map(int, present)) - set(map(int, basis)))
    if missing:
        raise ValueError(
            "The structure contains atomic numbers that are not present in the "
            f"declared element_list: {missing}. element_list must define the full "
            "dataset-wide species basis."
        )
    return basis


def _run_atomic_alignnd_once(atoms, cutoff, k, dihedral, elems_array,
                             store_atoms, include_angs, store_atoms_type,
                             atom_labels, cpu_cores, require_bonds,
                             require_angles, require_dihedrals,
                             include_equivariant_fields=True):
    edge_index_G, x_bnd, edge_shifts = atoms2graph(
        atoms, cutoff=cutoff, k=k, return_shifts=True
    )
    edge_index_G, x_bnd, edge_shifts = _deduplicate_bond_graph(
        edge_index_G, x_bnd, edge_shifts
    )

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError(f"No global bonds were created with cutoff={cutoff:.6g}, k={k}.")

    edge_index_G, x_bnd, edge_shifts = _make_bidirectional_graph(
        edge_index_G, x_bnd, edge_shifts
    )

    indptr, indices_arr, data_arr, shift_arr = build_adjacency_csr(
        edge_index_G, x_bnd, len(atoms), edge_shifts
    )

    # For strict local-angle validation, every atom-centered graph needs enough
    # local neighbors to produce the requested features. This validation happens
    # inside process_atom so it also catches angle-function failures.
    results = Parallel(n_jobs=cpu_cores, backend='loky', verbose=10)(
        delayed(process_atom)(
            i,
            indptr,
            indices_arr,
            data_arr,
            shift_arr,
            atoms,
            elems_array,
            atms=atoms if store_atoms else None,
            include_angs=include_angs,
            dihedral=dihedral,
            store_atoms=store_atoms,
            store_atoms_type=store_atoms_type,
            atom_labels=atom_labels,
            require_bonds=require_bonds,
            require_angles=bool(require_angles and include_angs),
            require_dihedrals=bool(require_dihedrals and include_angs and dihedral),
            include_equivariant_fields=include_equivariant_fields,
        )
        for i in range(len(atoms))
    )

    graph_data = [None] * len(atoms)
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[], x_dih_ang=[])

    for atom_index, data_obj, local_data_amounts in results:
        graph_data[atom_index] = data_obj
        for key, values in local_data_amounts.items():
            if values:
                data_amounts.setdefault(key, []).extend(values)

    for key in data_amounts:
        data_amounts[key] = torch.tensor(data_amounts[key], dtype=torch.long)

    return graph_data, data_amounts


def atomic_alignnd(atoms, neighbor_params, dihedral=False, all_elements=None,
                   store_atoms=False, use_pt=False, include_angs=True,
                   store_atoms_type='ase-atoms', atom_labels='', cpu_cores=-1,
                   include_equivariant_fields=True,
                   *, auto_retry_graph=True, max_graph_attempts=8, k_step=4,
                   cutoff_scale=1.25, max_k=None, max_cutoff=None,
                   require_bonds=True, require_angles=True,
                   require_dihedrals=False, retry_verbose=True):
    """Create one global-ID-preserving local graph per atom.

    Retry is applied to the shared parent atomic graph. If any requested local
    bond/angle/dihedral graph cannot be made, only this structure is retried
    with expanded k/cutoff.
    """
    atom_numbers = atoms.get_atomic_numbers()

    if use_pt:
        pdb = Physics_data()
        elems_array = np.array([el.number for el in pdb.elements], dtype=np.int64)
    else:
        elems_array = _resolve_element_numbers(all_elements, atom_numbers)

    elems_array = np.asarray(elems_array, dtype=np.int64)

    if len(neighbor_params) < 2:
        raise ValueError("neighbor_params must contain [cutoff, k].")

    cutoff = float(neighbor_params[0])
    k = int(neighbor_params[1])
    if max_k is None and k > 0 and not np.any(np.asarray(atoms.pbc, dtype=bool)):
        max_k = max(1, len(atoms) - 1)

    last_error = None
    for attempt in range(max_graph_attempts):
        try:
            graph_data, _ = _run_atomic_alignnd_once(
                atoms,
                cutoff=cutoff,
                k=k,
                dihedral=dihedral,
                elems_array=elems_array,
                store_atoms=store_atoms,
                include_angs=include_angs,
                store_atoms_type=store_atoms_type,
                atom_labels=atom_labels,
                cpu_cores=cpu_cores,
                require_bonds=require_bonds,
                require_angles=require_angles,
                require_dihedrals=require_dihedrals,
                include_equivariant_fields=include_equivariant_fields,
            )
            for graph in graph_data:
                graph.graph_cutoff_used = cutoff
                graph.graph_k_used = k
                graph.graph_build_attempts = attempt + 1

            if retry_verbose and attempt > 0:
                print(
                    f"atomic_alignnd: graph construction succeeded after "
                    f"{attempt + 1} attempts using cutoff={cutoff:.6g}, k={k}."
                )
            return graph_data
        except Exception as exc:
            last_error = exc
            if not auto_retry_graph or attempt == max_graph_attempts - 1:
                break

            old_cutoff = cutoff
            old_k = k
            cutoff, k, changed = _increase_graph_search_params(
                cutoff,
                k,
                k_step=k_step,
                cutoff_scale=cutoff_scale,
                max_k=max_k,
                max_cutoff=max_cutoff,
            )
            if not changed:
                break

            if retry_verbose:
                print(
                    f"atomic_alignnd: graph construction failed with "
                    f"cutoff={old_cutoff:.6g}, k={old_k}. Retrying with "
                    f"cutoff={cutoff:.6g}, k={k}. Reason: {exc}"
                )

    raise RuntimeError(
        "atomic_alignnd: failed to construct valid local graphs after "
        f"{max_graph_attempts} attempts. Final cutoff={cutoff:.6g}, k={k}. "
        f"Last error was: {last_error}"
    )


def atomic_alignnd_from_global_graph(global_graph, dihedral=False, store_atoms=False,
                                     include_angs=True, include_equivariant_fields=True,
                                     store_atoms_type='ase-atoms',
                                     require_bonds=True, require_angles=True,
                                     require_dihedrals=False):
    """Create atom-centered local graphs from an existing global graph.

    This path cannot retry cutoff/k because the graph has already been built.
    It now validates requested local bonds/angles/dihedrals and handles empty
    line/dihedral graphs without shape crashes.
    """
    if 'atoms' not in global_graph:
        raise ValueError("global_graph must contain an 'atoms' field.")

    atoms = global_graph['atoms']
    edge_index_G = _normalize_edge_index(
        _extract_numpy_graph_field(global_graph, 'edge_index_G', dtype=np.int64),
        dtype=np.int64,
    )
    x_bnd = _to_numpy_1d(global_graph['x_bnd'], dtype=np.float32)
    raw_shifts = getattr(global_graph, 'shifts', None)
    if raw_shifts is None and isinstance(global_graph, dict):
        raw_shifts = global_graph.get('shifts')
    if raw_shifts is None:
        # Backward-compatible inference for older saved graphs that predate
        # explicit periodic image shifts.
        eq_tmp = build_equivariant_atomic_fields(
            atoms, edge_index_G, dtype=np.float32, include_edge_geometry=True
        )
        edge_shifts = eq_tmp['shifts'].detach().cpu().numpy()
    else:
        if hasattr(raw_shifts, 'detach'):
            raw_shifts = raw_shifts.detach().cpu().numpy()
        edge_shifts = _normalize_shifts(raw_shifts, edge_index_G.shape[1])
    x_atm_global = global_graph['x_atm']
    if hasattr(x_atm_global, 'detach'):
        x_atm_global_np = x_atm_global.detach().cpu().numpy()
    else:
        x_atm_global_np = np.asarray(x_atm_global)

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError("atomic_alignnd_from_global_graph: input graph has no bonds.")

    edge_index_G, x_bnd, edge_shifts = _make_bidirectional_graph(
        edge_index_G, x_bnd, edge_shifts
    )
    indptr, indices_arr, data_arr, shift_arr = build_adjacency_csr(
        edge_index_G, x_bnd, len(atoms), edge_shifts
    )

    if edge_index_G.shape[1] > 0:
        unique_atoms = np.unique(edge_index_G[0])
    else:
        unique_atoms = np.arange(len(atoms), dtype=np.int64)

    graph_data = []
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts['x_dih_ang'] = []

    for atom in unique_atoms:
        atom = int(atom)
        tmp_edge_index_G, tmp_x_bnd, tmp_edge_shifts = _build_atomic_local_graph(
            center_atom=atom,
            indptr=indptr,
            indices=indices_arr,
            bond_data=data_arr,
            shift_data=shift_arr,
            include_dihedral_shell=bool(include_angs and dihedral),
        )

        if require_bonds and tmp_edge_index_G.shape[1] == 0:
            raise RuntimeError(
                f"atomic_alignnd_from_global_graph: no local bonds for atom {atom}."
            )

        if store_atoms:
            if store_atoms_type == 'ase-atoms':
                stored_atoms = atoms
            else:
                stored_atoms = atoms[atom]
        else:
            stored_atoms = None

        local_atom_indices = _ordered_local_atom_indices(tmp_edge_index_G, atom)
        x_atm = x_atm_global_np[local_atom_indices]

        equivariant_kwargs = {'shifts': torch.tensor(tmp_edge_shifts, dtype=torch.long)}
        if include_equivariant_fields:
            equivariant_kwargs = build_equivariant_atomic_fields(
                atoms,
                tmp_edge_index_G,
                atom_indices=local_atom_indices,
                dtype=np.float32,
                include_edge_geometry=True,
                shifts=tmp_edge_shifts,
            )

        data_amounts['x_atm'].append(len(x_atm) - 1)
        data_amounts['x_bnd'].append(len(tmp_x_bnd) - 1)

        if include_angs:
            local_angles = _build_local_angles(
                atoms,
                tmp_edge_index_G,
                tmp_edge_shifts,
                center_atom=atom,
                include_angs=include_angs,
                dihedral=dihedral,
                require_angles=require_angles,
                require_dihedrals=require_dihedrals,
            )
            edge_index_A = local_angles['edge_index_A']
            x_ang = local_angles['x_ang']
            mask_dih_ang = local_angles['mask_dih_ang']
            x_dih_ang = local_angles['x_dih_ang']

            data_amounts['x_ang'].append(len(x_ang) - 1)
            if dihedral:
                data_amounts['x_dih_ang'].append(len(x_dih_ang) - 1)

            data_obj = Atomic_Graph_Data(
                atoms=stored_atoms,
                edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
                edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
                x_atm=torch.tensor(x_atm, dtype=torch.float),
                x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
                x_ang=torch.tensor(x_ang, dtype=torch.float),
                mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool),
                atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
                **equivariant_kwargs,
            )
        else:
            data_obj = Atomic_Graph_Data(
                atoms=stored_atoms,
                edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
                edge_index_A=None,
                x_atm=torch.tensor(x_atm, dtype=torch.float),
                x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
                x_ang=None,
                mask_dih_ang=None,
                atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                ang_amounts=None,
                **equivariant_kwargs,
            )

        data_obj.generate_gid()
        graph_data.append(data_obj)

    return graph_data
