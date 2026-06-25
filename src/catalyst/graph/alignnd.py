from ..properties.physics_database import Physics_data
from ..properties.structure_properties import *
from ..utilities.data_tools import remove_duplicate_list_pairs
from .graph import Atomic_Graph_Data
from .graph import atoms2graph
from .graph import line_graph
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
            atom_labels=data['node_labels'],
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
            atom_labels=data['node_labels'],
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


def _deduplicate_bond_graph(edge_index_G, x_bnd):
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)

    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges before "
            f"duplicate removal: {edge_index_G.shape[1]} versus {len(x_bnd)}."
        )

    if edge_index_G.shape[1] == 0:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.float32)

    edge_index_G_0, edge_index_G_1, unique_edges = remove_duplicate_list_pairs(
        edge_index_G[0], edge_index_G[1], stack=False
    )
    edge_index_G = _normalize_edge_index(
        np.asarray([edge_index_G_0, edge_index_G_1], dtype=np.int64),
        dtype=np.int64,
    )

    unique_edges = np.asarray(unique_edges, dtype=np.int64)
    x_bnd = x_bnd[unique_edges]

    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges after "
            f"duplicate removal: {edge_index_G.shape[1]} versus {len(x_bnd)}."
        )

    return edge_index_G, x_bnd.astype(np.float32, copy=False)


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
                              require_angles, require_dihedrals):
    if not include_angs:
        return dict(
            edge_index_A=None,
            x_ang=None,
            mask_dih_ang=None,
            x_bnd_ang=np.empty((0,), dtype=np.float32),
            x_dih_ang=np.empty((0,), dtype=np.float32),
        )

    edge_index_bnd_ang = _normalize_edge_index(line_graph(edge_index_G), dtype=np.int64)
    if require_angles and edge_index_bnd_ang.shape[1] == 0:
        raise RuntimeError("No bond-angle edges were created.")

    if edge_index_bnd_ang.shape[1] > 0:
        x_bnd_ang = _to_numpy_1d(
            get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang),
            dtype=np.float32,
        )
    else:
        x_bnd_ang = np.empty((0,), dtype=np.float32)

    if dihedral:
        edge_index_dih_ang = _normalize_edge_index(dihedral_graph(edge_index_G), dtype=np.int64)
        if require_dihedrals and edge_index_dih_ang.shape[1] == 0:
            raise RuntimeError("No dihedral-angle edges were created.")

        if edge_index_dih_ang.shape[1] > 0:
            x_dih_ang = _to_numpy_1d(
                get_dih_angs(atoms, edge_index_G, edge_index_dih_ang),
                dtype=np.float32,
            )
        else:
            x_dih_ang = np.empty((0,), dtype=np.float32)

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
                                 require_bonds, require_angles, require_dihedrals):
    edge_index_G, x_bnd = atoms2graph(atoms, cutoff=cutoff, k=k)
    edge_index_G, x_bnd = _deduplicate_bond_graph(edge_index_G, x_bnd)

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError(f"No bonds were created with cutoff={cutoff:.6g}, k={k}.")

    angular = _build_angular_components(
        atoms,
        edge_index_G,
        include_angs=include_angs,
        dihedral=dihedral,
        require_angles=bool(require_angles and include_angs),
        require_dihedrals=bool(require_dihedrals and include_angs and dihedral),
    )

    result = dict(
        edge_index_G=edge_index_G,
        x_bnd=x_bnd,
        cutoff_used=cutoff,
        k_used=k,
    )
    result.update(angular)
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
                                       context='graph'):
    if len(neighbor_params) < 2:
        raise ValueError("neighbor_params must contain [cutoff, k].")

    cutoff = float(neighbor_params[0])
    k = int(neighbor_params[1])

    if max_graph_attempts < 1:
        raise ValueError("max_graph_attempts must be >= 1.")

    if max_k is None and k > 0:
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
            include_angs=True, atom_labels='', *, auto_retry_graph=True,
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

    elems_array = _element_basis_for_atoms(atoms, use_pt=use_pt)
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
    )

    edge_index_G_tensor = torch.tensor(components['edge_index_G'], dtype=torch.long)
    x_atm_tensor = torch.tensor(x_atm, dtype=torch.float)
    x_bnd_tensor = torch.tensor(components['x_bnd'], dtype=torch.float)

    if include_angs:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=edge_index_G_tensor,
            edge_index_A=torch.tensor(components['edge_index_A'], dtype=torch.long),
            x_atm=x_atm_tensor,
            x_bnd=x_bnd_tensor,
            x_ang=torch.tensor(components['x_ang'], dtype=torch.float),
            mask_dih_ang=torch.tensor(components['mask_dih_ang'], dtype=torch.bool),
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
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=None,
        )

    data.graph_cutoff_used = components['cutoff_used']
    data.graph_k_used = components['k_used']
    data.graph_build_attempts = components['graph_build_attempts']
    data.generate_gid()
    return data


def realignnd(structures, neighbor_params, dihedral=False, store_atoms=False,
              use_pt=False, include_angs=True, atom_labels='', *,
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

    elems_array = _element_basis_for_structures(structures, use_pt=use_pt)

    f_edge_index_G = []
    f_edge_index_A = []
    f_x_atm = []
    f_x_bnd = []
    f_x_ang = []
    f_mask_dih_ang = []

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
        )

        edge_index_G = components['edge_index_G'] + atom_offset
        f_edge_index_G.append(edge_index_G)
        f_x_atm.append(x_atm)
        f_x_bnd.append(components['x_bnd'])

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
        )

    data.graph_cutoffs_used = cutoffs_used
    data.graph_ks_used = ks_used
    data.graph_build_attempts = attempts_used
    data.generate_gid()
    return data


# -----------------------------------------------------------------------------
# Atomic/local graph helpers
# -----------------------------------------------------------------------------

def _make_bidirectional_graph(edge_index_G, x_bnd):
    """Create one directed edge in each direction for every physical bond."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)

    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges: "
            f"{edge_index_G.shape[1]} versus {len(x_bnd)}."
        )

    if edge_index_G.shape[1] == 0:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.float32)

    undirected_bonds = {}
    for src, dst, bond_value in zip(edge_index_G[0], edge_index_G[1], x_bnd):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key not in undirected_bonds:
            undirected_bonds[key] = float(bond_value)

    directed_src = []
    directed_dst = []
    directed_bnd = []
    for (atom_a, atom_b), bond_value in sorted(undirected_bonds.items()):
        directed_src.extend((atom_a, atom_b))
        directed_dst.extend((atom_b, atom_a))
        directed_bnd.extend((bond_value, bond_value))

    return (
        np.asarray([directed_src, directed_dst], dtype=np.int64),
        np.asarray(directed_bnd, dtype=np.float32),
    )


def build_adjacency_csr(edge_index_G, x_bnd, len_atoms):
    """Build CSR adjacency arrays from a directed global bond graph."""
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = _to_numpy_1d(x_bnd, dtype=np.float32)

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
        )

    if edge_index_G.min() < 0 or edge_index_G.max() >= n_atoms:
        raise ValueError(
            "The global edge index contains an atom ID outside the ASE Atoms "
            f"range [0, {n_atoms - 1}]."
        )

    order = np.lexsort((edge_index_G[1], edge_index_G[0]))
    src = edge_index_G[0][order]
    dst = edge_index_G[1][order]
    bond_values = x_bnd[order]

    counts = np.bincount(src, minlength=n_atoms)
    indptr = np.zeros(n_atoms + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)

    return (
        indptr,
        np.asarray(dst, dtype=np.int64),
        np.asarray(bond_values, dtype=np.float32),
    )


def _csr_neighbors(atom_index, indptr, indices, bond_data):
    start = int(indptr[atom_index])
    end = int(indptr[atom_index + 1])
    return indices[start:end], bond_data[start:end]


def _build_atomic_local_graph(center_atom, indptr, indices, bond_data,
                              include_dihedral_shell):
    """Build a center-preserving local bond graph using global atom IDs."""
    local_edges = {}

    def add_directed_edge(src, dst, bond_value):
        key = (int(src), int(dst))
        if key not in local_edges:
            local_edges[key] = float(bond_value)

    def add_bidirectional_bond(atom_a, atom_b, bond_value):
        if int(atom_a) == int(atom_b):
            return
        add_directed_edge(atom_a, atom_b, bond_value)
        add_directed_edge(atom_b, atom_a, bond_value)

    first_neighbors, first_bonds = _csr_neighbors(center_atom, indptr, indices, bond_data)

    for neighbor, bond_value in zip(first_neighbors, first_bonds):
        add_bidirectional_bond(center_atom, neighbor, bond_value)

    if include_dihedral_shell:
        for first_neighbor in first_neighbors:
            second_neighbors, second_bonds = _csr_neighbors(int(first_neighbor), indptr, indices, bond_data)
            for second_neighbor, bond_value in zip(second_neighbors, second_bonds):
                if int(second_neighbor) == int(center_atom):
                    continue
                add_bidirectional_bond(first_neighbor, second_neighbor, bond_value)

    if not local_edges:
        return _empty_edge_index(dtype=np.int64), np.empty((0,), dtype=np.float32)

    src = []
    dst = []
    local_bonds = []
    for (edge_src, edge_dst), bond_value in local_edges.items():
        src.append(edge_src)
        dst.append(edge_dst)
        local_bonds.append(bond_value)

    return (
        np.asarray([src, dst], dtype=np.int64),
        np.asarray(local_bonds, dtype=np.float32),
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
        (src_G[u] != src_G[v])
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
    four_distinct_atoms = (
        (atom_a != atom_b) & (atom_a != atom_c) & (atom_a != atom_d) &
        (atom_b != atom_c) & (atom_b != atom_d) &
        (atom_c != atom_d)
    )

    return edge_index_A[:, center_is_on_central_bond & four_distinct_atoms]


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


def _build_local_angles(atoms, tmp_edge_index_G, center_atom, include_angs, dihedral,
                        require_angles=False, require_dihedrals=False):
    if not include_angs:
        return dict(
            edge_index_A=None,
            x_ang=None,
            mask_dih_ang=None,
            x_bnd_ang=np.empty((0,), dtype=np.float32),
            x_dih_ang=np.empty((0,), dtype=np.float32),
        )

    raw_bond_angle_edges = _normalize_edge_index(line_graph(tmp_edge_index_G), dtype=np.int64)
    edge_index_bnd_ang = _filter_center_bond_angles(
        tmp_edge_index_G, raw_bond_angle_edges, center_atom
    )

    if require_angles and edge_index_bnd_ang.shape[1] == 0:
        raise RuntimeError(f"No centered bond angles were created for atom {center_atom}.")

    if edge_index_bnd_ang.shape[1] == 0:
        x_bnd_ang = np.empty((0,), dtype=np.float32)
    else:
        x_bnd_ang = _to_numpy_1d(
            get_bnd_angs(atoms, tmp_edge_index_G, edge_index_bnd_ang),
            dtype=np.float32,
        )

    if dihedral:
        raw_dihedral_edges = _normalize_edge_index(dihedral_graph(tmp_edge_index_G), dtype=np.int64)
        edge_index_dih_ang = _filter_center_dihedrals(
            tmp_edge_index_G, raw_dihedral_edges, center_atom
        )

        if require_dihedrals and edge_index_dih_ang.shape[1] == 0:
            raise RuntimeError(f"No centered dihedrals were created for atom {center_atom}.")

        if edge_index_dih_ang.shape[1] == 0:
            x_dih_ang = np.empty((0,), dtype=np.float32)
        else:
            x_dih_ang = _to_numpy_1d(
                get_dih_angs(atoms, tmp_edge_index_G, edge_index_dih_ang),
                dtype=np.float32,
            )

        edge_index_A = _hstack_edge_indices((edge_index_bnd_ang, edge_index_dih_ang))
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

    return dict(
        edge_index_A=edge_index_A,
        x_ang=x_ang,
        mask_dih_ang=mask_dih_ang,
        x_bnd_ang=x_bnd_ang,
        x_dih_ang=x_dih_ang,
    )


def process_atom(i, indptr, indices, data, atoms, elems_array, atms,
                 include_angs, dihedral, store_atoms, store_atoms_type,
                 atom_labels, require_bonds=False, require_angles=False,
                 require_dihedrals=False):
    """Build one atomic graph, including a two-hop shell when needed."""
    if store_atoms:
        stored_atoms = atoms if store_atoms_type == 'ase-atoms' else atoms[i]
    else:
        stored_atoms = None

    tmp_edge_index_G, tmp_x_bnd = _build_atomic_local_graph(
        center_atom=i,
        indptr=indptr,
        indices=indices,
        bond_data=data,
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

    if include_angs:
        local_angles = _build_local_angles(
            atoms,
            tmp_edge_index_G,
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
        )

    data_obj.generate_gid()
    return i, data_obj, local_data_amounts


def _resolve_element_numbers(all_elements, atom_numbers):
    """Return a sorted atomic-number feature basis."""
    present = np.asarray(np.unique(atom_numbers), dtype=np.int64)

    if all_elements is None or len(all_elements) == 0:
        return present

    from ase.data import atomic_numbers as ase_atomic_numbers

    resolved = []
    for element in all_elements:
        if isinstance(element, str):
            stripped = element.strip()
            if stripped.isdigit():
                resolved.append(int(stripped))
            else:
                if stripped not in ase_atomic_numbers:
                    raise ValueError(f"Unknown element symbol in all_elements: {element!r}")
                resolved.append(int(ase_atomic_numbers[stripped]))
        else:
            resolved.append(int(element))

    return np.unique(np.concatenate((present, np.asarray(resolved, dtype=np.int64))))


def _run_atomic_alignnd_once(atoms, cutoff, k, dihedral, elems_array,
                             store_atoms, include_angs, store_atoms_type,
                             atom_labels, cpu_cores, require_bonds,
                             require_angles, require_dihedrals):
    edge_index_G, x_bnd = atoms2graph(atoms, cutoff=cutoff, k=k)
    edge_index_G, x_bnd = _deduplicate_bond_graph(edge_index_G, x_bnd)

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError(f"No global bonds were created with cutoff={cutoff:.6g}, k={k}.")

    edge_index_G, x_bnd = _make_bidirectional_graph(edge_index_G, x_bnd)

    indptr, indices_arr, data_arr = build_adjacency_csr(edge_index_G, x_bnd, len(atoms))

    # For strict local-angle validation, every atom-centered graph needs enough
    # local neighbors to produce the requested features. This validation happens
    # inside process_atom so it also catches angle-function failures.
    results = Parallel(n_jobs=cpu_cores, backend='loky', verbose=10)(
        delayed(process_atom)(
            i,
            indptr,
            indices_arr,
            data_arr,
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

    elems_array = np.sort(np.asarray(elems_array, dtype=np.int64))

    if len(neighbor_params) < 2:
        raise ValueError("neighbor_params must contain [cutoff, k].")

    cutoff = float(neighbor_params[0])
    k = int(neighbor_params[1])
    if max_k is None and k > 0:
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
                                     include_angs=True, store_atoms_type='ase-atoms',
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
    x_atm_global = global_graph['x_atm']
    if hasattr(x_atm_global, 'detach'):
        x_atm_global_np = x_atm_global.detach().cpu().numpy()
    else:
        x_atm_global_np = np.asarray(x_atm_global)

    if require_bonds and edge_index_G.shape[1] == 0:
        raise RuntimeError("atomic_alignnd_from_global_graph: input graph has no bonds.")

    edge_index_G, x_bnd = _make_bidirectional_graph(edge_index_G, x_bnd)
    indptr, indices_arr, data_arr = build_adjacency_csr(edge_index_G, x_bnd, len(atoms))

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
        tmp_edge_index_G, tmp_x_bnd = _build_atomic_local_graph(
            center_atom=atom,
            indptr=indptr,
            indices=indices_arr,
            bond_data=data_arr,
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

        data_amounts['x_atm'].append(len(x_atm) - 1)
        data_amounts['x_bnd'].append(len(tmp_x_bnd) - 1)

        if include_angs:
            local_angles = _build_local_angles(
                atoms,
                tmp_edge_index_G,
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
            )

        data_obj.generate_gid()
        graph_data.append(data_obj)

    return graph_data
