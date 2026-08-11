from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree
from functools import partial
import pandas as pd
import numpy as np
import itertools
import secrets
import sys

import torch
from torch_geometric.data import Data

import numba
from numba import njit, prange

mask2index = lambda mask: np.flatnonzero(mask)

import warnings
import numpy as np
from ase.geometry import get_distances


def _ase_neighbor_candidates(atoms, search_cutoff, dtype=np.float32):
    """Return directed ASE neighbor candidates including periodic image shifts.

    ``shifts`` follows ASE's convention: the image position of destination ``j``
    is ``pos[j] + shifts @ cell``.  Keeping this integer shift is essential for
    small periodic cells, where the same atom index can legitimately appear
    several times in one coordination shell through different periodic images.
    """
    if search_cutoff is None or not np.isfinite(search_cutoff) or search_cutoff <= 0:
        raise ValueError("search_cutoff must be a positive finite value.")

    i, j, shifts, distances = neighbor_list(
        "ijSd",
        atoms,
        float(search_cutoff),
        self_interaction=False,
    )
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    distances = np.asarray(distances, dtype=dtype).reshape(-1)

    # Guard against a zero-distance self edge while retaining periodic self-image
    # neighbors such as those that occur in one-atom primitive cells.
    keep = ~((i == j) & np.all(shifts == 0, axis=1))
    keep &= distances > np.finfo(np.dtype(dtype)).eps
    return i[keep], j[keep], shifts[keep], distances[keep]


def _initial_knn_search_cutoff(atoms):
    """Choose a finite initial radius when kNN is requested without a cutoff."""
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if len(positions) <= 1:
        span = 1.0
    else:
        span = float(np.linalg.norm(np.ptp(positions, axis=0)))

    periodic_lengths = [
        float(np.linalg.norm(atoms.cell.array[axis]))
        for axis, periodic in enumerate(np.asarray(atoms.pbc, dtype=bool))
        if periodic and np.linalg.norm(atoms.cell.array[axis]) > 0
    ]
    if periodic_lengths:
        span = max(span, max(periodic_lengths))

    return max(span, 1.0)


def build_knn_edges_from_atoms(
    atoms,
    k,
    cutoff=None,
    cutoff_multiplier=2.0,
    dtype=np.float32,
    require_full_k=False,
    symmetrize=False,
    return_shifts=False,
):
    """Build a periodic-image-aware k-nearest-neighbor graph from ASE Atoms.

    Unlike a minimum-image distance matrix over only the atoms in the reference
    cell, this implementation selects from ASE neighbor-list entries identified
    by ``(source, destination, cell_shift)``.  Consequently a small periodic cell
    can still represent a complete coordination shell even when several physical
    neighbors correspond to the same destination atom index in different images.

    Returns ``(edge_index, distances)`` by default.  With ``return_shifts=True``
    it returns ``(edge_index, distances, shifts)`` where ``shifts`` has shape
    ``(n_edges, 3)`` and integer ASE cell offsets.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if cutoff_multiplier <= 0:
        raise ValueError("cutoff_multiplier must be > 0.")

    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("Cannot build a kNN graph for an empty Atoms object.")

    atoms = atoms.copy()
    atoms.wrap()

    if cutoff is not None:
        search_cutoff = float(cutoff_multiplier) * float(cutoff)
        max_search_attempts = 1
    else:
        search_cutoff = _initial_knn_search_cutoff(atoms)
        # When no upper bound is requested, increase the radius until every atom
        # has k periodic-image candidates or a conservative attempt limit is hit.
        max_search_attempts = 10

    candidates = None
    counts = None
    for _ in range(max_search_attempts):
        i, j, shifts, distances = _ase_neighbor_candidates(
            atoms, search_cutoff, dtype=dtype
        )
        counts = np.bincount(i, minlength=n_atoms) if len(i) else np.zeros(n_atoms, dtype=int)
        candidates = (i, j, shifts, distances)
        if np.all(counts >= k) or cutoff is not None:
            break
        search_cutoff *= 1.5

    i, j, shifts, distances = candidates
    all_i, all_j, all_s, all_d = [], [], [], []

    for atom_i in range(n_atoms):
        idx = np.flatnonzero(i == atom_i)
        if idx.size:
            # Stable lexicographic tie-breaking makes equal-distance shells
            # deterministic across platforms.
            order = np.lexsort((
                shifts[idx, 2], shifts[idx, 1], shifts[idx, 0], j[idx], distances[idx]
            ))
            idx = idx[order[:k]]

        if len(idx) < k:
            message = (
                f"Atom {atom_i} only has {len(idx)} valid periodic-image neighbors "
                f"within search cutoff {search_cutoff:.6g}, but k={k} was requested."
            )
            if require_full_k:
                raise ValueError(message)
            warnings.warn(message + " Returning fewer than k edges for this atom.", RuntimeWarning)

        all_i.extend(i[idx].tolist())
        all_j.extend(j[idx].tolist())
        all_s.extend(shifts[idx].tolist())
        all_d.extend(distances[idx].tolist())

    edge_index = np.asarray([all_i, all_j], dtype=np.int64)
    edge_shifts = np.asarray(all_s, dtype=np.int64).reshape(-1, 3)
    edge_distances = np.asarray(all_d, dtype=dtype)

    if symmetrize and edge_index.shape[1] > 0:
        rev_index = edge_index[::-1]
        rev_shifts = -edge_shifts
        edge_index = np.concatenate((edge_index, rev_index), axis=1)
        edge_shifts = np.concatenate((edge_shifts, rev_shifts), axis=0)
        edge_distances = np.concatenate((edge_distances, edge_distances), axis=0)

        # Remove only exact periodic-image duplicates, never distinct images.
        seen = set()
        keep = []
        for edge_id, (src, dst) in enumerate(edge_index.T):
            shift = edge_shifts[edge_id]
            key = (int(src), int(dst), int(shift[0]), int(shift[1]), int(shift[2]))
            if key not in seen:
                seen.add(key)
                keep.append(edge_id)
        keep = np.asarray(keep, dtype=np.int64)
        edge_index = edge_index[:, keep]
        edge_shifts = edge_shifts[keep]
        edge_distances = edge_distances[keep]

    if return_shifts:
        return edge_index, edge_distances, edge_shifts
    return edge_index, edge_distances


# =============================================================================
# Equivariant graph-field helpers
# =============================================================================


def _normalize_edge_index_array(edge_index, dtype=np.int64):
    """Return a NumPy edge index with canonical shape (2, n_edges)."""
    edge_index = np.asarray(edge_index, dtype=dtype)

    if edge_index.size == 0:
        return np.empty((2, 0), dtype=dtype)

    if edge_index.ndim == 1:
        if edge_index.size != 2:
            raise ValueError(
                f"A one-dimensional edge index must have exactly two values; "
                f"received shape {edge_index.shape}."
            )
        return edge_index.reshape(2, 1)

    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D; received shape {edge_index.shape}.")

    if edge_index.shape[0] == 2:
        return edge_index

    if edge_index.shape[1] == 2:
        return edge_index.T

    raise ValueError(
        f"edge_index must have shape (2, n_edges) or (n_edges, 2); "
        f"received shape {edge_index.shape}."
    )


def _to_torch_optional(value, dtype=None):
    """Convert optional NumPy/list values to torch tensors without touching tensors."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def _infer_integer_shifts(edge_vec, raw_vec, cell, pbc):
    """Infer integer periodic image shifts from MIC and raw vectors.

    Convention:
        edge_vec = pos[j] + shifts @ cell - pos[i]

    The inference is best-effort. If the cell is singular or nonperiodic, zeros
    are returned.
    """
    edge_vec = np.asarray(edge_vec, dtype=np.float64).reshape(-1, 3)
    raw_vec = np.asarray(raw_vec, dtype=np.float64).reshape(-1, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    pbc = np.asarray(pbc, dtype=bool).reshape(3)

    shifts = np.zeros((edge_vec.shape[0], 3), dtype=np.int64)

    if edge_vec.shape[0] == 0 or not np.any(pbc):
        return shifts

    try:
        if np.linalg.matrix_rank(cell) < 3:
            return shifts
        shift_cart = edge_vec - raw_vec
        frac_shift = np.linalg.solve(cell.T, shift_cart.T).T
        shifts = np.rint(frac_shift).astype(np.int64)
        shifts[:, ~pbc] = 0
        return shifts
    except Exception:
        return np.zeros((edge_vec.shape[0], 3), dtype=np.int64)


def _edge_geometry_from_atoms(atoms, edge_index, dtype=np.float32, shifts=None):
    """Compute edge vectors/distances and integer periodic-image shifts.

    If explicit ASE image shifts are supplied they are treated as authoritative.
    This preserves distinct periodic images that share the same ``(src, dst)``
    atom indices.  Without shifts, the function falls back to minimum-image
    geometry for backward compatibility.
    """
    edge_index = _normalize_edge_index_array(edge_index, dtype=np.int64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64).reshape(3, 3)
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(3)
    n_edges = edge_index.shape[1]

    if shifts is not None:
        shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
        if shifts.shape[0] != n_edges:
            raise ValueError(
                "shifts and edge_index must contain the same number of edges: "
                f"{shifts.shape[0]} versus {n_edges}."
            )
        shifts = shifts.copy()
        shifts[:, ~pbc] = 0
        raw_vec = positions[edge_index[1]] - positions[edge_index[0]]
        edge_vec = raw_vec + shifts @ cell
        edge_dist = np.linalg.norm(edge_vec, axis=1)
        return (
            np.asarray(edge_vec, dtype=dtype),
            np.asarray(edge_dist, dtype=dtype),
            shifts,
        )

    edge_vec = np.empty((n_edges, 3), dtype=dtype)
    edge_dist = np.empty((n_edges,), dtype=dtype)
    raw_vec = np.empty((n_edges, 3), dtype=np.float64)

    for edge_id, (src, dst) in enumerate(edge_index.T):
        src = int(src)
        dst = int(dst)
        raw_vec[edge_id] = positions[dst] - positions[src]
        vectors, distances = get_distances(
            positions[src], positions[dst], cell=cell, pbc=pbc
        )
        edge_vec[edge_id] = np.asarray(vectors, dtype=dtype).reshape(-1, 3)[0]
        edge_dist[edge_id] = np.asarray(distances, dtype=dtype).reshape(-1)[0]

    inferred_shifts = _infer_integer_shifts(edge_vec, raw_vec, cell, pbc)
    return edge_vec, edge_dist, inferred_shifts


def build_equivariant_atomic_fields(
    atoms,
    edge_index,
    atom_indices=None,
    *,
    dtype=np.float32,
    include_edge_geometry=True,
    shifts=None,
):
    """Build standardized equivariant fields for an ASE-atom graph.

    Returned fields follow the Catalyst equivariant contract:

        z, pos, edge_index, edge_vec, edge_dist, cell, pbc, shifts

    Parameters
    ----------
    atoms
        ASE Atoms object.
    edge_index
        Edges referencing the parent ASE Atoms object. Shape can be (2, E) or
        (E, 2).
    atom_indices
        Optional global atom IDs included in a local graph. If supplied,
        ``pos`` and ``z`` are local/subset arrays and the returned ``edge_index``
        is remapped into this local index space. ``global_atom_indices`` is also
        stored for traceability.
    dtype
        Floating dtype for positions, cell, vectors, and distances.
    include_edge_geometry
        If True, also stores edge_vec and edge_dist. For conservative
        energy-gradient force models, equivariant processors can ignore these
        precomputed tensors and recompute geometry from pos/cell/shifts inside
        forward so autograd tracks positions.
    """
    edge_index_global = _normalize_edge_index_array(edge_index, dtype=np.int64)

    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    positions = np.asarray(atoms.get_positions(), dtype=dtype)
    cell = np.asarray(atoms.cell.array, dtype=dtype).reshape(3, 3)
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(3)

    if atom_indices is None:
        selected = np.arange(len(atoms), dtype=np.int64)
        edge_index_local = edge_index_global
    else:
        selected = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
        global_to_local = {int(atom_id): local_id for local_id, atom_id in enumerate(selected)}

        src_local = []
        dst_local = []
        for src, dst in edge_index_global.T:
            src = int(src)
            dst = int(dst)
            if src not in global_to_local or dst not in global_to_local:
                raise ValueError(
                    "Cannot build local equivariant fields because an edge "
                    f"({src}, {dst}) references an atom outside atom_indices."
                )
            src_local.append(global_to_local[src])
            dst_local.append(global_to_local[dst])

        edge_index_local = np.asarray([src_local, dst_local], dtype=np.int64)

    fields = {
        "z": torch.as_tensor(atomic_numbers[selected], dtype=torch.long),
        "pos": torch.as_tensor(positions[selected], dtype=torch.float),
        "edge_index": torch.as_tensor(edge_index_local, dtype=torch.long),
        "cell": torch.as_tensor(cell, dtype=torch.float),
        "pbc": torch.as_tensor(pbc, dtype=torch.bool),
        "global_atom_indices": torch.as_tensor(selected, dtype=torch.long),
    }

    if include_edge_geometry:
        edge_vec, edge_dist, shifts = _edge_geometry_from_atoms(
            atoms,
            edge_index_global,
            dtype=dtype,
            shifts=shifts,
        )
        fields["edge_vec"] = torch.as_tensor(edge_vec, dtype=torch.float)
        fields["edge_dist"] = torch.as_tensor(edge_dist, dtype=torch.float)
        fields["shifts"] = torch.as_tensor(shifts, dtype=torch.long)
    else:
        fields["edge_vec"] = None
        fields["edge_dist"] = None
        fields["shifts"] = torch.zeros((edge_index_global.shape[1], 3), dtype=torch.long)

    return fields


def attach_equivariant_fields(data, **fields):
    """Attach standardized equivariant fields to a PyG Data object in-place."""
    for key, value in fields.items():
        if value is not None:
            setattr(data, key, value)
    return data

class Generic_Graph_Data(Data):
    """Custom PyG data for representing a primary graph plus optional line graph.

    In addition to the legacy generic fields, this class can also carry the
    standardized equivariant graph fields used by future equivariant processors:

        z, pos, edge_index, edge_vec, edge_dist, cell, pbc, shifts

    These fields are optional so existing generic workflows remain compatible.
    """

    def __init__(self,
                 edge_index_G,
                 node_G,
                 reference=None,
                 edge_index_A=None,
                 node_A=None,
                 edge_A=None,
                 mask_edge_A=None,
                 node_G_amounts=None,
                 node_A_amounts=None,
                 edge_A_amounts=None,
                 z=None,
                 pos=None,
                 edge_index=None,
                 edge_vec=None,
                 edge_dist=None,
                 cell=None,
                 pbc=None,
                 shifts=None,
                 global_atom_indices=None,
                 atom_graph_batch=None,
                 edge_graph_batch=None
                 ):
        super().__init__()
        self.reference = reference
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.node_G = node_G
        self.node_A = node_A
        self.edge_A = edge_A
        self.mask_edge_A = mask_edge_A
        self.node_G_amounts = node_G_amounts
        self.node_A_amounts = node_A_amounts
        self.edge_A_amounts = edge_A_amounts

        # Order-style aliases used by the modular GNN framework.
        self.x_1 = node_G
        self.x_2 = node_A
        self.x_3 = edge_A
        self.edge_index_2 = edge_index_G
        self.edge_index_3 = edge_index_A

        # PyG/equivariant-style aliases.  If an explicit equivariant edge_index
        # is not supplied, default to the primary graph connectivity.
        self.edge_index = edge_index if edge_index is not None else edge_index_G
        self.z = z
        self.pos = pos
        self.edge_vec = edge_vec
        self.edge_dist = edge_dist
        self.cell = cell
        self.pbc = pbc
        self.shifts = shifts
        self.global_atom_indices = global_atom_indices
        self.atom_graph_batch = atom_graph_batch
        self.edge_graph_batch = edge_graph_batch

        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_G":
            return self.node_G.size(0)

        if key == "edge_index_A":
            return self.node_A.size(0)

        if key == "edge_index":
            if self.z is not None:
                return self.z.size(0)
            if self.pos is not None:
                return self.pos.size(0)
            return self.node_G.size(0)

        if key == "global_atom_indices":
            return 0

        if key in {"shifts", "edge_vec", "edge_dist", "cell", "pbc"}:
            return 0

        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {'cell', 'pbc'}:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)

class Atomic_Graph_Data(Data):
    """Custom PyG data for atom/bond graphs plus optional angle/dihedral graph.

    The legacy ALIGNN fields are kept unchanged:

        x_atm, x_bnd, x_ang, edge_index_G, edge_index_A

    This class can also carry standardized equivariant fields:

        z, pos, edge_index, edge_vec, edge_dist, cell, pbc, shifts

    ``edge_index`` is the equivariant/primary neighbor graph.  For global atomic
    graphs it is usually identical to ``edge_index_G``.  For local atom-centered
    graphs, it can be a local remapping while ``edge_index_G`` preserves the
    existing legacy/global-index behavior.
    """

    def __init__(self,
                 atoms,
                 edge_index_G,
                 edge_index_A,
                 x_atm,
                 x_bnd,
                 x_ang,
                 atm_amounts,
                 bnd_amounts,
                 ang_amounts,
                 mask_dih_ang=None,
                 z=None,
                 pos=None,
                 edge_index=None,
                 edge_vec=None,
                 edge_dist=None,
                 cell=None,
                 pbc=None,
                 shifts=None,
                 global_atom_indices=None,
                 atom_graph_batch=None,
                 edge_graph_batch=None
                 ):
        super().__init__()
        self.atoms = atoms
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        self.atm_amounts = atm_amounts
        self.bnd_amounts = bnd_amounts
        self.ang_amounts = ang_amounts

        # Order-style aliases used by the modular GNN framework.
        self.x_1 = x_atm
        self.x_2 = x_bnd
        self.x_3 = x_ang
        self.edge_index_2 = edge_index_G
        self.edge_index_3 = edge_index_A

        # Standard equivariant graph fields.
        self.edge_index = edge_index if edge_index is not None else edge_index_G
        self.z = z
        self.pos = pos
        self.edge_vec = edge_vec
        self.edge_dist = edge_dist
        self.cell = cell
        self.pbc = pbc
        self.shifts = shifts
        self.global_atom_indices = global_atom_indices
        self.atom_graph_batch = atom_graph_batch
        self.edge_graph_batch = edge_graph_batch

        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_G":
            return self.x_atm.size(0)

        if key == "edge_index_A":
            return self.x_bnd.size(0)

        if key == "edge_index":
            if hasattr(self, "z") and self.z is not None:
                return self.z.size(0)
            if hasattr(self, "pos") and self.pos is not None:
                return self.pos.size(0)
            return self.x_atm.size(0)

        if key == "global_atom_indices":
            return 0

        if key in {"shifts", "edge_vec", "edge_dist", "cell", "pbc"}:
            return 0

        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {'cell', 'pbc'}:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)

def index2mask(idx_arr, n):
    mask = np.zeros(n, dtype=int)
    mask[idx_arr] = 1
    return mask.astype(np.bool)

def np_groupby(arr, groups):
    """Numpy implementation of `groupby` operation (a common method in pandas).
    """
    arr, groups = np.array(arr), np.array(groups)
    sort_idx = groups.argsort()
    arr = arr[sort_idx]
    groups = groups[sort_idx]
    return np.split(arr, np.unique(groups, return_index=True)[1])[1:]

def np_scatter(src, index, func):
    """Generalization of the `torch_scatter.scatter` operation for any reduce function.
    See https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html for how `scatter` works.

    Args:
        src (array): The source array.
        index (array of int): The indices of elements to scatter.
        func (function): Reduce function (e.g., mean, sum) that operates on elements with the same indices.

    :rtype: generator
    """
    return (func(g) for g in np_groupby(src, index))

def summary(model):
    """Returns a dataframe describing the numbers of trainable parameters in a torch model.
    """
    params = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total_num = sum(n for _, n in params)
    params.append(('Total', total_num))
    return pd.DataFrame(params, columns=['Layer', 'Params'])

import numpy as np
import numba
from numba import njit, prange

@njit
def minimum_image(dx, box_length):
    if dx > 0.5 * box_length:
        dx -= box_length
    elif dx < -0.5 * box_length:
        dx += box_length
    return dx

@njit
def build_cell_list(positions, box, cell_size):
    n_atoms = positions.shape[0]
    nx = int(box[0] // cell_size)
    ny = int(box[1] // cell_size)
    nz = int(box[2] // cell_size)
    if nx == 0: nx = 1
    if ny == 0: ny = 1
    if nz == 0: nz = 1

    n_cells = nx * ny * nz
    atom_cell_indices = np.empty(n_atoms, dtype=np.int32)

    for i in range(n_atoms):
        cx = int(positions[i,0] / cell_size) % nx
        cy = int(positions[i,1] / cell_size) % ny
        cz = int(positions[i,2] / cell_size) % nz
        atom_cell_indices[i] = cx + cy*nx + cz*nx*ny

    counts = np.zeros(n_cells, dtype=np.int32)
    for c in atom_cell_indices:
        counts[c] += 1

    cell_starts = np.zeros(n_cells, dtype=np.int32)
    for i in range(1, n_cells):
        cell_starts[i] = cell_starts[i-1] + counts[i-1]

    cell_ends = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_cells):
        cell_ends[i] = cell_starts[i] + counts[i]

    cell_list_atoms = np.empty(n_atoms, dtype=np.int32)
    counters = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_atoms):
        c = atom_cell_indices[i]
        pos = cell_starts[c] + counters[c]
        cell_list_atoms[pos] = i
        counters[c] += 1

    return nx, ny, nz, cell_starts, cell_ends, cell_list_atoms, atom_cell_indices

@njit
def get_neighbor_cells(cx, cy, cz, nx, ny, nz):
    neighbors = []
    for dx in (-1,0,1):
        nxn = (cx + dx) % nx
        for dy in (-1,0,1):
            nyn = (cy + dy) % ny
            for dz in (-1,0,1):
                nzn = (cz + dz) % nz
                neighbors.append(nxn + nyn*nx + nzn*nx*ny)
    return neighbors

@njit(parallel=True)
def find_neighbors_cell_list(positions, box, cell_size, cutoff):
    n_atoms = positions.shape[0]
    cutoff_sq = cutoff * cutoff
    nx, ny, nz, cell_starts, cell_ends, cell_list_atoms, atom_cell_indices = build_cell_list(positions, box, cell_size)

    max_edges_per_thread = 10000  # Adjust based on memory, e.g. ~max expected neighbors per thread
    n_threads = numba.get_num_threads()

    # create buffers per thread:
    edge_i_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.int32)
    edge_j_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.int32)
    edge_d_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.float32)
    edge_count_threads = np.zeros(n_threads, dtype=np.int32)

    for atom_idx in prange(n_atoms):
        thread_id = numba.get_thread_id()
        pos_i = positions[atom_idx]

        cx = int(pos_i[0] / cell_size) % nx
        cy = int(pos_i[1] / cell_size) % ny
        cz = int(pos_i[2] / cell_size) % nz

        neighbor_cells = get_neighbor_cells(cx, cy, cz, nx, ny, nz)

        for cell in neighbor_cells:
            start = cell_starts[cell]
            end = cell_ends[cell]
            for idx in range(start, end):
                j = cell_list_atoms[idx]
                if j <= atom_idx:
                    continue

                dx = positions[j,0] - pos_i[0]
                dy = positions[j,1] - pos_i[1]
                dz = positions[j,2] - pos_i[2]

                if dx > 0.5 * box[0]: dx -= box[0]
                elif dx < -0.5 * box[0]: dx += box[0]
                if dy > 0.5 * box[1]: dy -= box[1]
                elif dy < -0.5 * box[1]: dy += box[1]
                if dz > 0.5 * box[2]: dz -= box[2]
                elif dz < -0.5 * box[2]: dz += box[2]

                r2 = dx*dx + dy*dy + dz*dz
                if r2 <= cutoff_sq:
                    count = edge_count_threads[thread_id]
                    if count == max_edges_per_thread:
                        # Could handle buffer extension or skip overflow edges safely
                        continue
                    edge_i_threads[thread_id, count] = atom_idx
                    edge_j_threads[thread_id, count] = j
                    edge_d_threads[thread_id, count] = np.sqrt(r2)
                    edge_count_threads[thread_id] = count + 1

    # After parallel: gather all thread edges together
    total_edges = np.sum(edge_count_threads)
    edge_i = np.empty(total_edges, dtype=np.int32)
    edge_j = np.empty(total_edges, dtype=np.int32)
    edge_d = np.empty(total_edges, dtype=np.float32)

    pos = 0
    for t in range(n_threads):
        count = edge_count_threads[t]
        edge_i[pos:pos+count] = edge_i_threads[t, :count]
        edge_j[pos:pos+count] = edge_j_threads[t, :count]
        edge_d[pos:pos+count] = edge_d_threads[t, :count]
        pos += count

    return edge_i, edge_j, edge_d

'''
def atoms2graph(atoms, cutoff, k=5):
    """
    Fast neighbor finder for atoms based on cell linked list with PBC.
    Fully replaces ASE neighbor_list and KDTree approaches.

    Returns:
        edge_index (2 x N_edges ndarray)
        edge_attr distances as ndarray (N_edges,)
    """
    atoms.wrap()  # wrap positions into the box
    pos = atoms.get_positions()
    box = np.linalg.norm(atoms.cell[:3], axis=1)
    cell_size = cutoff

    edge_i, edge_j, edge_d = find_neighbors_cell_list(pos, box, cell_size, cutoff)

    return np.stack((edge_i, edge_j)), edge_d.astype(np.float32)

'''
def atoms2graph(atoms, cutoff, k=5, return_shifts=False):
    """Convert ASE Atoms to a directed periodic-image-aware neighbor graph.

    Radius graphs (``k < 0``) use ASE's ``(i, j, S, d)`` neighbor-list output so
    distinct periodic images are retained.  kNN graphs use the same image-aware
    candidate representation and select the nearest ``k`` entries per source.

    With ``return_shifts=True`` the third return value is an integer ``(E, 3)``
    array of ASE cell shifts satisfying
    ``edge_vec = pos[j] + shifts @ cell - pos[i]``.
    """
    if k < 0:
        i, j, shifts, distances = _ase_neighbor_candidates(
            atoms, float(cutoff), dtype=np.float32
        )
        edge_index = np.stack((i, j)) if len(i) else np.empty((2, 0), dtype=np.int64)
        if return_shifts:
            return edge_index, distances.astype(np.float32, copy=False), shifts
        return edge_index, distances.astype(np.float32, copy=False)

    result = build_knn_edges_from_atoms(
        atoms=atoms,
        k=k,
        cutoff=cutoff,
        cutoff_multiplier=2.0,
        require_full_k=False,
        symmetrize=False,
        return_shifts=return_shifts,
    )
    return result


permute_2 = partial(itertools.permutations, r=2)
def line_graph(edge_index_G):
    """Return the (angular) line graph of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src_G, dst_G = edge_index_G
    edge_index_A = [
        (u, v)
        for edge_pairs in np_scatter(np.arange(len(dst_G)), dst_G, permute_2)
        for u, v in edge_pairs
    ]
    return np.array(edge_index_A).T

def dihedral_graph(edge_index_G):
    """Return the "dihedral angle line graph" of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src, dst = edge_index_G
    edge_index_A = [
        (u, v)
        for i, j in edge_index_G.T
        for u in np.flatnonzero((dst == i) & (src != j))
        for v in np.flatnonzero((dst == j) & (src != i))
    ]
    return np.array(edge_index_A).T

'''
DEPRECIATED CLASS: To remove prior to v1.0
'''
class Graph_Data(Data):
    """Custom PyG data for representing a pair of two graphs: one for regular atomic
    structure (atom and bonds) and the other for bond/dihedral angles.

    The following arguments assume an atomic graph of `N_atm` atoms with `N_bnd` bonds,
    and an angular graph of `N_ang` angles (including dihedral angles, if there's any).

    Args:
        edge_index_G (LongTensor): Edge index of the atomic graph "G".
        x_atm (Tensor): Atom features.
        x_bnd (Tensor): Bond features.
        edge_index_A (LongTensor): Edge index of the angular graph "A".
        x_ang (Tensor): Angle features.
        mask_dih_ang (Boolean Tensor, optional): If the angular graph contains dihedral
            angles, this mask indicates which angles are dihedral angles.
    """

    def __init__(self,
                 atoms,
                 edge_index_G,
                 edge_index_A,
                 x_atm,
                 x_bnd,
                 x_ang,
                 atm_amounts,
                 bnd_amounts,
                 ang_amounts,
                 mask_dih_ang=None
                 ):
        super().__init__()
        self.atoms = atoms
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        self.atm_amounts = atm_amounts
        self.bnd_amounts = bnd_amounts
        self.ang_amounts = ang_amounts

        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_A':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)
