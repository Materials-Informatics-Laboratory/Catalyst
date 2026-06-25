from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree
from functools import partial
import pandas as pd
import numpy as np
import itertools
import secrets
import sys

from torch_geometric.data import Data

import numba
from numba import njit, prange

mask2index = lambda mask: np.flatnonzero(mask)

import warnings
import numpy as np
from ase.geometry import get_distances


def build_knn_edges_from_atoms(
    atoms,
    k,
    cutoff=None,
    cutoff_multiplier=2.0,
    dtype=np.float32,
    require_full_k=False,
    symmetrize=False,
):
    """
    Build a robust k-nearest-neighbor graph from an ASE Atoms object.

    This uses ASE's minimum-image convention, so it works for arbitrary
    triclinic cells, orthorhombic cells, partial PBC, full PBC, and
    nonperiodic structures.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    k : int
        Number of nearest neighbors requested per atom.
    cutoff : float or None
        If provided, only neighbors within cutoff_multiplier * cutoff are kept.
        If None, no distance upper bound is applied.
    cutoff_multiplier : float
        Multiplier used for the distance upper bound.
    dtype : dtype
        Output distance dtype.
    require_full_k : bool
        If True, raise an error if any atom has fewer than k valid neighbors.
        If False, return fewer edges for those atoms.
    symmetrize : bool
        If True, add reverse edges j -> i as well.

    Returns
    -------
    edge_index : np.ndarray, shape (2, n_edges)
        Directed edge indices.
    edge_distances : np.ndarray, shape (n_edges,)
        Edge distances.
    """

    if k < 1:
        raise ValueError("k must be >= 1.")

    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("Cannot build a kNN graph for an empty Atoms object.")

    if n_atoms == 1:
        if require_full_k:
            raise ValueError("Cannot find neighbors for a single-atom structure.")
        return np.empty((2, 0), dtype=int), np.empty((0,), dtype=dtype)

    atoms = atoms.copy()
    atoms.wrap()

    positions = atoms.get_positions()
    cell = atoms.cell
    pbc = atoms.pbc

    max_neighbors = min(k, n_atoms - 1)

    distance_upper_bound = None
    if cutoff is not None:
        distance_upper_bound = cutoff_multiplier * cutoff

    all_i = []
    all_j = []
    all_d = []

    for atom_i in range(n_atoms):
        _, distances = get_distances(
            positions[atom_i],
            positions,
            cell=cell,
            pbc=pbc,
        )

        distances = np.asarray(distances).reshape(-1)

        # Remove self-neighbor.
        distances[atom_i] = np.inf

        # Apply optional upper-bound cutoff.
        if distance_upper_bound is not None:
            distances[distances > distance_upper_bound] = np.inf

        order = np.argsort(distances)
        chosen = order[:max_neighbors]
        chosen_dists = distances[chosen]

        valid = np.isfinite(chosen_dists)
        chosen = chosen[valid]
        chosen_dists = chosen_dists[valid]

        if require_full_k and len(chosen) < k:
            raise ValueError(
                f"Atom {atom_i} only has {len(chosen)} valid neighbors, "
                f"but k={k} was requested."
            )

        if len(chosen) < k:
            warnings.warn(
                f"Atom {atom_i} only has {len(chosen)} valid neighbors "
                f"within the requested search range. Returning fewer than k edges "
                f"for this atom.",
                RuntimeWarning,
            )

        all_i.extend([atom_i] * len(chosen))
        all_j.extend(chosen.tolist())
        all_d.extend(chosen_dists.tolist())

    edge_index = np.array([all_i, all_j], dtype=int)
    edge_distances = np.array(all_d, dtype=dtype)

    if symmetrize and edge_index.shape[1] > 0:
        rev_edge_index = edge_index[::-1]
        edge_index = np.concatenate([edge_index, rev_edge_index], axis=1)
        edge_distances = np.concatenate([edge_distances, edge_distances], axis=0)

    return edge_index, edge_distances

class Generic_Graph_Data(Data):
    """Custom PyG data for representing a pair of two graphs: one for the original graph and the other for its line grpah variant.
    This data is used to represent an arbitrary graph and does not hard-code variables based on their atomistic connection.
    """

    def __init__(self,
                 edge_index_G,
                 node_G,
                 reference=None,
                 edge_index_A=None,
                 node_A=None,
                 edge_A=None,
                 mask_edge_A=None,
                 node_G_amounts = None,
                 node_A_amounts = None,
                 edge_A_amounts = None
                 ):
        super().__init__()
        self.reference = reference
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.node_G = node_G
        self.node_A = node_A
        self.edge_A = edge_A
        self.node_G_amounts = node_G_amounts
        self.node_A_amounts = node_A_amounts
        self.edge_A_amounts = edge_A_amounts
        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.node_G.size(0)
        if key == 'edge_index_A':
            return self.node_A.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)

class Atomic_Graph_Data(Data):
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
def atoms2graph(atoms, cutoff,k=5):
    """Convert an ASE `Atoms` object into a graph based on a radius cutoff.
    Returns the graph (in COO format) and its edge attributes (format
    determined by `edge_dist`).

    Args:
        atoms (ase.Atoms): Collection of atoms to be converted to a graph.
        cutoff (float): Cutoff radius for nearest neighbor search.
        edge_dist (bool, optional): Set to `True` to output edge distances.
            Otherwise, output edge vectors.

    Returns:
       tuple: Tuple of (edge_index, edge_attr) that describes the atomic graph.

    :rtype: (ndarray, ndarray)
    """
    if k < 0:
        i, j, d = neighbor_list('ijd', atoms, cutoff)
        return np.stack((i, j)), np.array(d).astype(np.float32)
    else:
        edge_index, edge_distances = build_knn_edges_from_atoms(
            atoms=atoms,
            k=k,
            cutoff=cutoff,
            cutoff_multiplier=2.0,
            require_full_k=False,
            symmetrize=False,
        )

        return edge_index, edge_distances

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
