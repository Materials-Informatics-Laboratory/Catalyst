from ..properties.physics_database import Physics_data
from ..properties.structure_properties import *
from ..utilities.data_tools import remove_duplicate_list_pairs
from .graph import Atomic_Graph_Data
from .graph import atoms2graph
from .graph import line_graph
from joblib import Parallel, delayed
from collections import defaultdict
import torch
import math

def check_params(target_dict):
    source_dict = {
        'type':None,
        'raw_data': None,
        'node_labels': None,
        'element_list': None,
        'neighbor_params': [5.0, -1],
        'is_dihedral': False,
        'store_raw_data': False,
        'use_pt': False,
        'include_angs': False,
        'cpu_cores':-1,
        'store_atoms_type': 'ase-atoms'
    }
    updated_dict = target_dict.copy()

    for key, source_value in source_dict.items():
        if key not in updated_dict:
            updated_dict[key] = source_value

    return updated_dict

def alignn_gen(data):
    data = check_params(data)
    if data['type'] == 'alignnd':
        graphs = alignnd(atoms=data['raw_data'],neighbor_params=data['neighbor_params'],dihedral=data['is_dihedral'],
                         store_atoms=data['store_raw_data'],use_pt=data['use_pt'],include_angs=data['include_angs'],
                         atom_labels=data['node_labels'])
    if data['type'] == 'realignnd':
        graphs = realignnd(structures=data['raw_data'],neighbor_params=data['neighbor_params'],dihedral=data['is_dihedral'],
                         store_atoms=data['store_raw_data'],use_pt=data['use_pt'],include_angs=data['include_angs'],
                         atom_labels=data['node_labels'])
    if data['type'] == 'atomic_alignnd':
        graphs = atomic_alignnd(atoms=data['raw_data'],neighbor_params=data['neighbor_params'],dihedral=data['is_dihedral'],
                         store_atoms=data['store_raw_data'],use_pt=data['use_pt'],include_angs=data['include_angs'],
                         atom_labels=data['node_labels'],all_elements=data['element_list'],store_atoms_type=data['store_atoms_type'],
                         cpu_cores=data['cpu_cores'])
    if data['type'] == 'atomic_alignnd_from_global_graph':
        graphs = atomic_alignnd_from_global_graph(global_graph=data['raw_data'],dihedral=data['is_dihedral'],
                         store_atoms=data['store_raw_data'],include_angs=data['include_angs'],store_atoms_type=data['store_atoms_type'])

    return graphs

def alignnd(atoms, neighbor_params, dihedral=False, store_atoms=False, use_pt=False,
            include_angs=True, atom_labels=''):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """
    atms = atoms if store_atoms else None

    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts['x_dih_ang'] = []

    pdb = Physics_data() if use_pt else None

    if use_pt:
        all_elem = np.array([el.number for el in pdb.elements])
    else:
        all_elem = np.sort(np.unique(atoms.get_atomic_numbers()), axis=None)

    atomic_numbers = atoms.get_atomic_numbers()

    x_atm = np.zeros((len(atomic_numbers), len(all_elem)), dtype=float)

    elem_to_idx = {el: i for i, el in enumerate(all_elem)}

    for atom_idx, atom_number in enumerate(atomic_numbers):
        i = elem_to_idx.get(atom_number)
        if i is not None:
            x_atm[atom_idx, i] = atom_number if atom_labels == 'atomic_number' else 1.0

    edge_index_G, x_bnd = atoms2graph(
        atoms,
        cutoff=neighbor_params[0],
        k=neighbor_params[1]
    )

    edge_index_G_0, edge_index_G_1, unique_edges = remove_duplicate_list_pairs(
        edge_index_G[0],
        edge_index_G[1],
        stack=False
    )

    edge_index_G = np.array([edge_index_G_0, edge_index_G_1])
    x_bnd = x_bnd[unique_edges]

    edge_index_G_tensor = torch.tensor(edge_index_G, dtype=torch.long)
    x_atm_tensor = torch.tensor(x_atm, dtype=torch.float)
    x_bnd_tensor = torch.tensor(x_bnd, dtype=torch.float)

    if include_angs:
        edge_index_bnd_ang = line_graph(edge_index_G)
        x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang)

        if dihedral:
            edge_index_dih_ang = dihedral_graph(edge_index_G)
            edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])

            x_dih_ang = get_dih_angs(atoms, edge_index_G, edge_index_dih_ang)
            x_ang = np.concatenate([x_bnd_ang, x_dih_ang])

            mask_dih_ang = [False] * len(x_bnd_ang) + [True] * len(x_dih_ang)
        else:
            edge_index_A = edge_index_bnd_ang
            x_ang = x_bnd_ang
            mask_dih_ang = [False] * len(x_ang)

        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=edge_index_G_tensor,
            edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
            x_atm=x_atm_tensor,
            x_bnd=x_bnd_tensor,
            x_ang=torch.tensor(x_ang, dtype=torch.float),
            mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool),
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

    data.generate_gid()

    return data

def realignnd(structures,neighbor_params,dihedral=False,store_atoms=False,use_pt=False,include_angs=True, atom_labels=''):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """
    f_edge_index_G = None
    f_edge_index_A = None
    f_x_atm = None
    f_x_ang = None
    f_x_bnd = None
    f_x_dih_ang = None
    f_mask_dih_ang = None
    atms = None

    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts.append(x_dih_ang=[])

    scale = 0
    all_elements = []
    for atoms in structures:
        elements = np.sort(np.unique(atoms.get_atomic_numbers()), axis=None)
        if len(all_elements) == 0:
            all_elements = elements
        else:
            for ee in elements:
                check = 0
                for ae in all_elements:
                    if ee == ae:
                        check = 1
                        break
                if check == 0:
                    all_elements.append(ee)
    for atoms in structures:
        pdb = None
        if use_pt:
            pdb = Physics_data()
        ohe = []
        for atom in atoms:
            all_elem = all_elements
            if use_pt:
                elems = []
                for el in pdb.elements:
                    elems.append(el.number)
                all_elem = elems
            tx = [0.0] * len(all_elem)
            for i in range(len(all_elem)):
                if atom.number == all_elem[i]:
                    tx[i] = 1.0
                    if atom_labels == 'atomic_number':
                        tx[i] *= atom.number
                    break
            ohe.append(tx)
        x_atm = np.array(ohe)

        edge_index_G, x_bnd = atoms2graph(atoms, cutoff=neighbor_params[0], k=neighbor_params[1])
        edge_index_G_0, edge_index_G_1, unique_edges = remove_duplicate_list_pairs(edge_index_G[0], edge_index_G[1],
                                                                                   stack=False)
        edge_index_G = [edge_index_G_0, edge_index_G_1]
        edge_index_G = np.array(edge_index_G)
        x_bnd = x_bnd[unique_edges]
        t_edge_index_G = []
        for i in range(len(edge_index_G)):
            t_edge_index_G.append([])
            for j in range(len(edge_index_G[i])):
                t_edge_index_G[-1].append(edge_index_G[i][j] + scale)
        t_edge_index_G = np.array(t_edge_index_G)

        if include_angs:
            edge_index_bnd_ang = line_graph(t_edge_index_G)
            x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang)

            if dihedral:
                edge_index_dih_ang = dihedral_graph(t_edge_index_G)
                edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
                x_dih_ang = get_dih_angs(atoms, edge_index_G, edge_index_dih_ang)
                x_ang = np.concatenate([x_bnd_ang,x_dih_ang])
                mask_dih_ang = [False] * len(x_bnd_ang) + [True]*len(x_dih_ang)
            else:
                edge_index_A = np.hstack([edge_index_bnd_ang])
                x_ang = np.concatenate([x_bnd_ang])
                mask_dih_ang = [False]

        if scale == 0:
            data_amounts["x_atm"].append(len(x_atm))
            data_amounts["x_bnd"].append(len(x_bnd))
            if include_angs:
                data_amounts["x_ang"].append(len(x_ang))
                if dihedral:
                    data_amounts["x_dih_ang"].append(len(x_dih_ang))
        else:
            data_amounts["x_atm"].append(data_amounts["x_atm"][-1] + len(x_atm))
            data_amounts["x_bnd"].append(data_amounts["x_bnd"][-1] + len(x_bnd))
            if include_angs:
                data_amounts["x_ang"].append(data_amounts["x_ang"][-1] + len(x_ang))
                if dihedral:
                    data_amounts["x_dih_ang"].append(data_amounts["x_dih_ang"][-1] + len(x_dih_ang))

        if scale > 0:
            if include_angs:
                if dihedral:
                    f_x_dih_ang = np.append(f_x_dih_ang, x_dih_ang)
                else:
                    ang_scale = data_amounts["x_bnd"][-2]
                    edge_index_A += ang_scale

                    f_edge_index_A = np.hstack((f_edge_index_A, edge_index_A))
                    f_x_ang = np.append(f_x_ang, x_ang)
                    f_mask_dih_ang = np.append(f_mask_dih_ang, mask_dih_ang)

            f_edge_index_G = np.hstack((f_edge_index_G, t_edge_index_G))
            f_x_atm = np.concatenate((f_x_atm, x_atm), axis=0)
            f_x_bnd = np.append(f_x_bnd, x_bnd)

            if store_atoms:
                atms.append(atoms)
        else:
            if include_angs:

                f_edge_index_A = edge_index_A
                f_x_ang = x_ang
                f_mask_dih_ang = mask_dih_ang
                if dihedral:
                    f_x_dih_ang = x_dih_ang
            f_edge_index_G = edge_index_G
            f_x_atm = x_atm
            f_x_bnd = x_bnd

            if store_atoms:
                atms = [atoms]
        scale += len(x_atm)


    if include_angs:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=torch.tensor(f_edge_index_G, dtype=torch.long),
            edge_index_A=torch.tensor(f_edge_index_A, dtype=torch.long),
            x_atm=torch.tensor(f_x_atm, dtype=torch.float),
            x_bnd=torch.tensor(f_x_bnd, dtype=torch.float),
            x_ang=torch.tensor(f_x_ang, dtype=torch.float),
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts = torch.tensor(data_amounts['x_ang'], dtype=torch.long),
            mask_dih_ang = torch.tensor(f_mask_dih_ang, dtype=torch.bool)
        )
    else:
        data = Atomic_Graph_Data(
            atoms=atms,
            edge_index_G=torch.tensor(f_edge_index_G, dtype=torch.long),
            edge_index_A=None,
            x_atm=torch.tensor(f_x_atm, dtype=torch.float),
            x_bnd=torch.tensor(f_x_bnd, dtype=torch.float),
            x_ang=None,
            atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
            bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
            ang_amounts=None,
            mask_dih_ang=None
        )

    data.generate_gid()

    return data


def _normalize_edge_index(edge_index, dtype=np.int64):
    """Return an edge-index array with the canonical shape ``(2, N)``.

    ``line_graph`` and ``dihedral_graph`` currently return ``shape == (0,)``
    when no graph edges are found.  Normalizing the empty case to ``(2, 0)``
    keeps concatenation and tensor conversion well defined.
    """
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


def _make_bidirectional_graph(edge_index_G, x_bnd):
    """Create one directed edge in each direction for every physical bond.

    The atomic graphs need incoming edges on both sides of a central bond for
    ``dihedral_graph`` to identify a four-atom path.  Bond attributes are
    treated as symmetric, which matches the distance-like ``x_bnd`` values
    used by this module.
    """
    edge_index_G = _normalize_edge_index(edge_index_G, dtype=np.int64)
    x_bnd = np.asarray(x_bnd, dtype=np.float32).reshape(-1)

    if edge_index_G.shape[1] != len(x_bnd):
        raise ValueError(
            "edge_index_G and x_bnd contain different numbers of edges: "
            f"{edge_index_G.shape[1]} versus {len(x_bnd)}."
        )

    if edge_index_G.shape[1] == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    # Collapse any existing forward/reverse duplicates into one physical bond.
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
    x_bnd = np.asarray(x_bnd, dtype=np.float32).reshape(-1)

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

    # CSR requires edges grouped by source atom.  Sorting also makes each local
    # graph deterministic across serial and parallel execution.
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
    """Return the neighbors and bond values for one atom from CSR arrays."""
    start = int(indptr[atom_index])
    end = int(indptr[atom_index + 1])
    return indices[start:end], bond_data[start:end]


def _build_atomic_local_graph(center_atom, indptr, indices, bond_data,
                              include_dihedral_shell):
    """Build a center-preserving local bond graph using global atom IDs.

    With ``include_dihedral_shell=False``, the result is the original one-hop
    star centered on ``center_atom``.

    With ``include_dihedral_shell=True``, bonds from each first-shell neighbor
    to its other neighbors are also included.  This supplies paths of the form
    ``A-B-center-D`` or ``A-center-C-D`` that are required for a proper
    four-atom dihedral, while retaining the original global atom indices.
    """
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

    first_neighbors, first_bonds = _csr_neighbors(
        center_atom, indptr, indices, bond_data
    )

    # Add the original one-hop atomic star first.  This keeps the center atom
    # first in the edge ordering and preserves the prior atomic-graph behavior.
    for neighbor, bond_value in zip(first_neighbors, first_bonds):
        add_bidirectional_bond(center_atom, neighbor, bond_value)

    if include_dihedral_shell:
        # Add only the branch bonds extending from first-shell neighbors.  This
        # is sufficient for center-associated dihedrals without copying the
        # entire induced two-hop global graph into every atomic graph.
        for first_neighbor in first_neighbors:
            second_neighbors, second_bonds = _csr_neighbors(
                int(first_neighbor), indptr, indices, bond_data
            )
            for second_neighbor, bond_value in zip(second_neighbors, second_bonds):
                if int(second_neighbor) == int(center_atom):
                    continue
                add_bidirectional_bond(first_neighbor, second_neighbor, bond_value)

    if not local_edges:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

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

    # line_graph pairs incoming bonds with the same destination.  Restricting
    # that destination to the requested atom preserves the original meaning of
    # an atom-centered bond-angle graph after the dihedral shell is added.
    keep = (
        (dst_G[u] == int(center_atom)) &
        (dst_G[v] == int(center_atom)) &
        (src_G[u] != src_G[v])
    )
    return edge_index_A[:, keep]


def _filter_center_dihedrals(edge_index_G, edge_index_A, center_atom):
    """Keep proper dihedrals with ``center_atom`` on the central bond.

    For one dihedral edge ``(u, v)`` generated by ``dihedral_graph``, the four
    atoms are ``src[u]-dst[u]-dst[v]-src[v]``.  The two middle atoms form the
    central bond.  Requiring the requested atom to be one of those middle atoms
    gives every atomic graph a clear center-associated dihedral definition.
    """
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

    # A proper dihedral requires four distinct atoms.  This also removes
    # triangle-generated paths such as center-B-C-center.
    four_distinct_atoms = (
        (atom_a != atom_b) & (atom_a != atom_c) & (atom_a != atom_d) &
        (atom_b != atom_c) & (atom_b != atom_d) &
        (atom_c != atom_d)
    )

    return edge_index_A[:, center_is_on_central_bond & four_distinct_atoms]


def _ordered_local_atom_indices(tmp_edge_index_G, center_atom):
    """Return global atom IDs in deterministic local-feature order."""
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
        atom_indices = _ordered_local_atom_indices(
            tmp_edge_index_G, center_atom_index
        )

    atomic_numbers = atoms.get_atomic_numbers()
    local_atomic_numbers = atomic_numbers[atom_indices]

    elem_to_idx = {int(elem): idx for idx, elem in enumerate(elems_array)}
    x_atm = np.zeros(
        (len(local_atomic_numbers), len(elems_array)), dtype=np.float32
    )

    for local_i, atomic_number in enumerate(local_atomic_numbers):
        feature_index = elem_to_idx.get(int(atomic_number))
        if feature_index is None:
            continue
        x_atm[local_i, feature_index] = (
            float(atomic_number) if atom_labels == 'atomic_number' else 1.0
        )

    return x_atm


def process_atom(i, indptr, indices, data, atoms, elems_array, atms,
                 include_angs, dihedral, store_atoms, store_atoms_type,
                 atom_labels):
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

    x_atm = build_x_atm(
        tmp_edge_index_G,
        atoms,
        elems_array,
        atom_labels,
        center_atom_index=i,
    )

    if include_angs:
        raw_bond_angle_edges = _normalize_edge_index(
            line_graph(tmp_edge_index_G), dtype=np.int64
        )
        edge_index_bnd_ang = _filter_center_bond_angles(
            tmp_edge_index_G, raw_bond_angle_edges, i
        )

        if edge_index_bnd_ang.shape[1] == 0:
            x_bnd_ang = np.empty((0,), dtype=np.float32)
        else:
            x_bnd_ang = np.asarray(
                get_bnd_angs(atoms, tmp_edge_index_G, edge_index_bnd_ang),
                dtype=np.float32,
            ).reshape(-1)

        if dihedral:
            raw_dihedral_edges = _normalize_edge_index(
                dihedral_graph(tmp_edge_index_G), dtype=np.int64
            )
            edge_index_dih_ang = _filter_center_dihedrals(
                tmp_edge_index_G, raw_dihedral_edges, i
            )

            if edge_index_dih_ang.shape[1] == 0:
                x_dih_ang = np.empty((0,), dtype=np.float32)
            else:
                x_dih_ang = np.asarray(
                    get_dih_angs(atoms, tmp_edge_index_G, edge_index_dih_ang),
                    dtype=np.float32,
                ).reshape(-1)

            edge_index_A = np.hstack(
                (edge_index_bnd_ang, edge_index_dih_ang)
            )
            x_ang = np.concatenate((x_bnd_ang, x_dih_ang))
            mask_dih_ang = np.concatenate((
                np.zeros(len(x_bnd_ang), dtype=bool),
                np.ones(len(x_dih_ang), dtype=bool),
            ))
        else:
            edge_index_A = edge_index_bnd_ang
            x_ang = x_bnd_ang
            x_dih_ang = np.empty((0,), dtype=np.float32)
            mask_dih_ang = np.zeros(len(x_ang), dtype=bool)

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
            atm_amounts=torch.tensor(
                local_data_amounts['x_atm'], dtype=torch.long
            ),
            bnd_amounts=torch.tensor(
                local_data_amounts['x_bnd'], dtype=torch.long
            ),
            ang_amounts=torch.tensor(
                local_data_amounts['x_ang'], dtype=torch.long
            ),
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
            atm_amounts=torch.tensor(
                local_data_amounts['x_atm'], dtype=torch.long
            ),
            bnd_amounts=torch.tensor(
                local_data_amounts['x_bnd'], dtype=torch.long
            ),
            ang_amounts=None,
            mask_dih_ang=None,
        )

    # The returned list is indexed by the global atom ID, and edge_index_G
    # continues to use those same global IDs.  This preserves the provenance
    # behavior of the current atomic_alignnd implementation.
    data_obj.generate_gid()
    return i, data_obj, local_data_amounts



def _resolve_element_numbers(all_elements, atom_numbers):
    """Return a sorted atomic-number feature basis.

    ``all_elements`` may contain atomic numbers or element symbols.  The
    elements actually present in ``atoms`` are always included so no local
    atomic feature row becomes silently all-zero because of an incomplete
    supplied element list.
    """
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

def atomic_alignnd(atoms, neighbor_params, dihedral=False, all_elements=[],
                   store_atoms=False, use_pt=False, include_angs=True,
                   store_atoms_type='ase-atoms', atom_labels='', cpu_cores=-1):
    """Create one global-ID-preserving local graph per atom.

    Bond-angle-only graphs retain the original one-hop atomic star.  When
    ``dihedral=True``, each local graph is expanded with the bonds connecting
    first-shell neighbors to their other neighbors.  Bond angles are then
    filtered back to those centered on the requested atom, while dihedrals are
    retained only when that atom lies on the dihedral's central bond.
    """
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[], x_dih_ang=[])

    atom_numbers = atoms.get_atomic_numbers()

    if use_pt:
        pdb = Physics_data()
        elems_array = np.array([el.number for el in pdb.elements], dtype=np.int64)
    else:
        elems_array = _resolve_element_numbers(all_elements, atom_numbers)

    elems_array = np.sort(np.asarray(elems_array, dtype=np.int64))

    edge_index_G, x_bnd = atoms2graph(
        atoms,
        cutoff=neighbor_params[0],
        k=neighbor_params[1],
    )
    edge_index_G_0, edge_index_G_1, unique_edges = remove_duplicate_list_pairs(
        edge_index_G[0], edge_index_G[1], stack=False
    )
    edge_index_G = np.asarray(
        [edge_index_G_0, edge_index_G_1], dtype=np.int64
    )
    x_bnd = np.asarray(x_bnd)[unique_edges]

    # Ensure both directions are available.  dihedral_graph needs incoming
    # branch bonds on each side of a central bond.
    edge_index_G, x_bnd = _make_bidirectional_graph(edge_index_G, x_bnd)

    indptr, indices_arr, data_arr = build_adjacency_csr(
        edge_index_G, x_bnd, len(atoms)
    )

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
        )
        for i in range(len(atoms))
    )

    graph_data = [None] * len(atoms)
    for atom_index, data_obj, local_data_amounts in results:
        graph_data[atom_index] = data_obj
        for key, values in local_data_amounts.items():
            if values:
                data_amounts.setdefault(key, []).extend(values)

    # Retained for compatibility with the previous implementation, even though
    # the per-graph amount tensors are stored directly on each data object.
    for key in data_amounts:
        data_amounts[key] = torch.tensor(data_amounts[key], dtype=torch.long)

    return graph_data
'''

def atomic_alignnd(atoms,neighbor_params,dihedral=False,all_elements=[],store_atoms=False,use_pt=False,
                   include_angs=True,store_atoms_type='ase-atoms',atom_labels='',cpu_cores=-1):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """

    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    atms = None
    if store_atoms:
        atms = atoms

    pdb = None
    if use_pt:
        pdb = Physics_data()
    elements = np.sort(np.unique(atoms.get_atomic_numbers()), axis=None)
    ohe = []
    if len(elements) < len(all_elements):
        elements = all_elements
    for atom in atoms:
        all_elem = elements
        if use_pt:
            elems = []
            for el in pdb.elements:
                elems.append(el.number)
            all_elem = elems
        tx = [0.0] * len(all_elem)
        for i in range(len(all_elem)):
            if atom.number == all_elem[i]:
                tx[i] = 1.0
                if atom_labels == 'atomic_number':
                    tx[i] *= atom.number
                break
        ohe.append(tx)

    edge_index_G, x_bnd = atoms2graph(atoms, cutoff=neighbor_params[0], k=neighbor_params[1])
    #edge_index_G, unique_edges = remove_duplicate_list_pairs(edge_index_G[0], edge_index_G[1])
    edge_index_G_0, edge_index_G_1, unique_edges = remove_duplicate_list_pairs(edge_index_G[0], edge_index_G[1],
                                                                               stack=False)
    edge_index_G = [edge_index_G_0, edge_index_G_1]
    edge_index_G = np.array(edge_index_G)
    x_bnd = x_bnd[unique_edges]
    data = []

    for i,atom in enumerate(atoms):
        if store_atoms:
            if store_atoms_type == 'ase-atoms':
                atm = atoms
            else:
                atm = atom
        idx = np.where(edge_index_G[0] == i)
        tmp_edge_index_G = [edge_index_G[0][idx],edge_index_G[1][idx]]
        tmp_x_bnd = x_bnd[idx]
        for j, val in enumerate(tmp_edge_index_G[1]):
            tmp_edge_index_G[0] = np.append(tmp_edge_index_G[0],val)
            tmp_edge_index_G[1] = np.append(tmp_edge_index_G[1],i)
            tmp_x_bnd = np.append(tmp_x_bnd,x_bnd[j])
        tmp_edge_index_G = np.array(tmp_edge_index_G)

        data_amounts["x_bnd"].append(len(tmp_x_bnd) - 1)
        if include_angs:
            edge_index_bnd_ang = line_graph(tmp_edge_index_G)
            x_bnd_ang = get_bnd_angs(atoms, tmp_edge_index_G, edge_index_bnd_ang)

            unique_elements, indices = np.unique(tmp_edge_index_G[0], return_index=True)
            unique_elements_in_order = unique_elements[np.argsort(indices)]
            for m in range(len(tmp_edge_index_G[0])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == tmp_edge_index_G[0][m]:
                        tmp_edge_index_G[0][m] = n
                        break
            for m in range(len(tmp_edge_index_G[1])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == tmp_edge_index_G[1][m]:
                        tmp_edge_index_G[1][m] = n
                        break
            x_atm = np.array(ohe)[unique_elements_in_order]
            edge_index_bnd_ang = line_graph(tmp_edge_index_G)

            if dihedral:
                edge_index_dih_ang = dihedral_graph(tmp_edge_index_G)
                edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
                x_dih_ang = get_dih_angs(atoms, tmp_edge_index_G, edge_index_dih_ang)
                x_ang = np.concatenate([x_bnd_ang,x_dih_ang])
                mask_dih_ang = [False] * len(x_bnd_ang) + [True]*len(x_dih_ang)
            else:
                edge_index_A = np.hstack([edge_index_bnd_ang])
                x_ang = np.concatenate([x_bnd_ang])
                mask_dih_ang = [False]

            data_amounts["x_atm"].append(len(x_atm) - 1)
            data_amounts["x_ang"].append(len(x_ang) - 1)
            if dihedral:
                data_amounts["x_dih_ang"].append(len(x_dih_ang) - 1)

            data.append(Atomic_Graph_Data(
                    atoms=atms,
                    edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
                    edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
                    x_atm=torch.tensor(x_atm, dtype=torch.float),
                    x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
                    x_ang=torch.tensor(x_ang, dtype=torch.float),
                    atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                    bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                    ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
                    mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool)
                ))
        else:
            unique_elements, indices = np.unique(tmp_edge_index_G[0], return_index=True)
            unique_elements_in_order = unique_elements[np.argsort(indices)]
            for m in range(len(tmp_edge_index_G[0])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == tmp_edge_index_G[0][m]:
                        tmp_edge_index_G[0][m] = n
                        break
            for m in range(len(tmp_edge_index_G[1])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == tmp_edge_index_G[1][m]:
                        tmp_edge_index_G[1][m] = n
                        break
            x_atm = np.array(ohe)[unique_elements_in_order]
            data_amounts["x_atm"].append(len(x_atm) - 1)
            data.append(Atomic_Graph_Data(
                    atoms=atms,
                    edge_index_G=torch.tensor(tmp_edge_index_G, dtype=torch.long),
                    edge_index_A=None,
                    x_atm=torch.tensor(x_atm, dtype=torch.float),
                    x_bnd=torch.tensor(tmp_x_bnd, dtype=torch.float),
                    x_ang=None,
                    atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                    bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                    ang_amounts=None,
                    mask_dih_ang=None
            ))

    for graph in data:
        graph.generate_gid()

    return data
'''
def atomic_alignnd_from_global_graph(global_graph,dihedral=False, store_atoms=False,include_angs=True,store_atoms_type='ase-atoms'):
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    atm = None

    atoms = []
    for g in global_graph['edge_index_G'][0]:
        atoms.append(g.item())
    atoms = np.array(atoms)
    neighbors = []
    for g in global_graph['edge_index_G'][1]:
        neighbors.append(g.item())
    neighbors = np.array(neighbors)
    x_atm = global_graph['x_atm']

    unique_atoms = np.unique(atoms)
    data = []
    for atom in unique_atoms:
        ids = np.where(atoms == atom)[0]
        edge_index_G = [atoms[ids],neighbors[ids]]
        x_bnd = []
        for b in global_graph['x_bnd'][ids]:
            x_bnd.append(b.item())
        x_bnd = np.array(x_bnd)

        for i,val in enumerate(edge_index_G[1]):
            edge_index_G[0] = np.append(edge_index_G[0], val)
            edge_index_G[1] = np.append(edge_index_G[1], edge_index_G[0][0])
            x_bnd = np.append(x_bnd, x_bnd[i])

        edge_index_G = np.array(edge_index_G)

        if store_atoms:
            if store_atoms_type == 'ase-atoms':
                atm = global_graph['atoms']
            else:
                atm = global_graph['atoms'][atom]

        if include_angs:
            edge_index_bnd_ang = line_graph(edge_index_G)
            x_bnd_ang = get_bnd_angs(global_graph['atoms'], edge_index_G, edge_index_bnd_ang)

            unique_elements, indices = np.unique(edge_index_G[0], return_index=True)
            unique_elements_in_order = unique_elements[np.argsort(indices)]
            for m in range(len(edge_index_G[0])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == edge_index_G[0][m]:
                        edge_index_G[0][m] = n
                        break
            for m in range(len(edge_index_G[1])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == edge_index_G[1][m]:
                        edge_index_G[1][m] = n
                        break
            x_atm = global_graph['x_atm'][unique_elements_in_order]
            edge_index_bnd_ang = line_graph(edge_index_G)

            if dihedral:
                edge_index_dih_ang = dihedral_graph(edge_index_G)
                edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
                x_dih_ang = get_dih_angs(global_graph['atoms'], edge_index_G, edge_index_dih_ang)
                x_ang = np.concatenate([x_bnd_ang, x_dih_ang])
                mask_dih_ang = [False] * len(x_bnd_ang) + [True] * len(x_dih_ang)
            else:
                edge_index_A = np.hstack([edge_index_bnd_ang])
                x_ang = np.concatenate([x_bnd_ang])
                mask_dih_ang = [False]

            data.append(Atomic_Graph_Data(
                    atoms=atm,
                    edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
                    edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
                    x_atm=torch.tensor(x_atm, dtype=torch.float),
                    x_bnd=torch.tensor(x_bnd, dtype=torch.float),
                    x_ang=torch.tensor(x_ang, dtype=torch.float),
                    mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool),
                    atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                    bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                    ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
                )
            )
        else:
            unique_elements, indices = np.unique(edge_index_G[0], return_index=True)
            unique_elements_in_order = unique_elements[np.argsort(indices)]
            for m in range(len(edge_index_G[0])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == edge_index_G[0][m]:
                        edge_index_G[0][m] = n
                        break
            for m in range(len(edge_index_G[1])):
                for n in range(len(unique_elements_in_order)):
                    if unique_elements_in_order[n] == edge_index_G[1][m]:
                        edge_index_G[1][m] = n
                        break
            x_atm = global_graph['x_atm'][unique_elements_in_order]
            data_amounts["x_atm"].append(len(x_atm) - 1)
            data.append(Atomic_Graph_Data(
                    atoms=atm,
                    edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
                    edge_index_A=None,
                    x_atm=torch.tensor(x_atm, dtype=torch.float),
                    x_bnd=torch.tensor(x_bnd, dtype=torch.float),
                    x_ang=None,
                    mask_dih_ang=None,
                    atm_amounts=torch.tensor(data_amounts['x_atm'], dtype=torch.long),
                    bnd_amounts=torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
                    ang_amounts=torch.tensor(data_amounts['x_ang'], dtype=torch.long),
                )
            )
    for graph in data:
        graph.generate_gid()

    return data
