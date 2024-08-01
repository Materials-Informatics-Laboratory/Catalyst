from ..utilities.structure_properties import *
from ..graphite.utils.alignn import *
from .graph import Graph_Data
from .graph import atoms2graph
import torch

def alignnd(atoms,cutoff,dihedral=False, store_atoms=False, atom_labels=''):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """
    atms = None
    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts.append(x_dih_ang=[])

    elements = np.sort(np.unique(atoms.get_atomic_numbers()), axis=None)
    ohe = []
    for atom in atoms:
        tx = [0.0] * len(elements)
        for i in range(len(elements)):
            if atom.number == elements[i]:
                tx[i] = 1.0
                if atom_labels == 'atomic_number':
                    tx[i] *= atom.number
                break
        ohe.append(tx)
    x_atm = np.array(ohe)

    edge_index_G, x_bnd = atoms2graph(atoms,cutoff=cutoff)
    edge_index_bnd_ang = line_graph(edge_index_G)
    x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang)

    if dihedral:
        edge_index_dih_ang = dihedral_graph(edge_index_G)
        edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
        x_dih_ang = get_dih_angs(atoms, edge_index_G, edge_index_dih_ang)
        x_ang = np.concatenate([x_bnd_ang,x_dih_ang])
        mask_dih_ang = [False] * len(x_bnd_ang) + [True]*len(x_dih_ang)

    else:
        edge_index_A = np.hstack([edge_index_bnd_ang])
        x_ang = np.concatenate([x_bnd_ang])
        mask_dih_ang = [False] * len(x_bnd_ang)

    if store_atoms:
        atms = atoms

    data = Graph_Data(
        atoms = atms,
        edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
        edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
        x_atm=torch.tensor(x_atm, dtype=torch.float),
        x_bnd=torch.tensor(x_bnd, dtype=torch.float),
        x_ang=torch.tensor(x_ang, dtype=torch.float),
        mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool),
        atm_amounts = torch.tensor(data_amounts['x_atm'], dtype=torch.long),
        bnd_amounts = torch.tensor(data_amounts['x_bnd'], dtype=torch.long),
        ang_amounts = torch.tensor(data_amounts['x_ang'], dtype=torch.long),
    )

    return data

def realignnd(structures,cutoff,dihedral=False,store_atoms=False, atom_labels=''):
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
        ohe = []
        for atom in atoms:
            tx = [0.0] * len(all_elements)
            for i in range(len(all_elements)):
                if atom.number == all_elements[i]:
                    tx[i] = 1.0
                    if atom_labels == 'atomic_number':
                        tx[i] *= atom.number
                    break
            ohe.append(tx)
        x_atm = np.array(ohe)

        edge_index_G, x_bnd = atoms2graph(atoms,cutoff=cutoff)
        t_edge_index_G = []
        for i in range(len(edge_index_G)):
            t_edge_index_G.append([])
            for j in range(len(edge_index_G[i])):
                t_edge_index_G[-1].append(edge_index_G[i][j] + scale)
        t_edge_index_G = np.array(t_edge_index_G)
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
            mask_dih_ang = [False] * len(x_bnd_ang)

        if len(data_amounts["x_atm"]) == 0:
            data_amounts["x_atm"].append(len(x_atm) - 1)
            data_amounts["x_bnd"].append(len(x_bnd) - 1)
            data_amounts["x_ang"].append(len(x_ang) - 1)
            if dihedral:
                data_amounts["x_dih_ang"].append(len(x_dih_ang) - 1)
        else:
            data_amounts["x_atm"].append(data_amounts["x_atm"][-1] + len(x_atm) - 1)
            data_amounts["x_bnd"].append(data_amounts["x_bnd"][-1] + len(x_bnd) - 1)
            data_amounts["x_ang"].append(data_amounts["x_ang"][-1] + len(x_ang) - 1)
            if dihedral:
                data_amounts["x_dih_ang"].append(data_amounts["x_dih_ang"][-1] + len(x_dih_ang) - 1)

        if scale > 0:
            f_edge_index_G = np.hstack((f_edge_index_G, t_edge_index_G))
            f_edge_index_A = np.hstack((f_edge_index_A, edge_index_A))
            f_x_atm = np.concatenate((f_x_atm, x_atm), axis=0)
            f_x_bnd = np.append(f_x_bnd, x_bnd)
            f_x_ang = np.append(f_x_ang, x_ang)
            f_mask_dih_ang = np.append(f_mask_dih_ang, mask_dih_ang)
            if dihedral:
                f_x_dih_ang = np.append(f_x_dih_ang, x_dih_ang)
            if store_atoms:
                atms.append(atoms)
        else:
            f_edge_index_G = edge_index_G
            f_edge_index_A = edge_index_A
            f_x_atm = x_atm
            f_x_ang = x_ang
            f_x_bnd = x_bnd
            f_mask_dih_ang = mask_dih_ang
            if dihedral:
                f_x_dih_ang = x_dih_ang
            if store_atoms:
                atms = [atoms]
        scale += len(atoms)

    data = Graph_Data(
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

    return data

def atomic_alignnd(atoms,cutoff,dihedral=False,all_elements=[],store_atoms=False,atom_labels=''):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """

    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    atms = None
    if store_atoms:
        atms = atoms

    elements = np.sort(np.unique(atoms.get_atomic_numbers()), axis=None)
    ohe = []
    if len(elements) < len(all_elements):
        elements = all_elements
    for atom in atoms:
        tx = [0.0] * len(elements)
        for i in range(len(elements)):
            if atom.number == elements[i]:
                tx[i] = 1.0
                if atom_labels == 'atomic_number':
                    tx[i] *= atom.number
                break
        ohe.append(tx)
    x_atm = np.array(ohe)

    edge_index_G, x_bnd = atoms2graph(atoms,cutoff=cutoff)
    data = []
    for i,atom in enumerate(atoms):
        try:
            idx = np.where(edge_index_G[0] == i)
            tmp_edge_index_G = [edge_index_G[0][idx],edge_index_G[1][idx]]
            tmp_x_bnd = x_bnd[idx]
            #print('start: ' + str(len(tmp_edge_index_G[0])) + ' bl ' + str(len(tmp_x_bnd)))
            for j, val in enumerate(tmp_edge_index_G[1]):
                tmp_edge_index_G[0] = np.append(tmp_edge_index_G[0],val)
                tmp_edge_index_G[1] = np.append(tmp_edge_index_G[1],i)
                #idx_bnd = np.where((edge_index_G[0] == i) & (edge_index_G[1] == val))
                tmp_x_bnd = np.append(tmp_x_bnd,x_bnd[j])
            #print('end: ' + str(len(tmp_edge_index_G[0])) + ' bl ' + str(len(tmp_x_bnd)))
            tmp_edge_index_G = np.array(tmp_edge_index_G)
            edge_index_bnd_ang = line_graph(tmp_edge_index_G)
            x_bnd_ang = get_bnd_angs(atoms, tmp_edge_index_G, edge_index_bnd_ang)

            if dihedral:
                edge_index_dih_ang = dihedral_graph(tmp_edge_index_G)
                edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
                x_dih_ang = get_dih_angs(atoms, tmp_edge_index_G, edge_index_dih_ang)
                x_ang = np.concatenate([x_bnd_ang,x_dih_ang])
                mask_dih_ang = [False] * len(x_bnd_ang) + [True]*len(x_dih_ang)

            else:
                edge_index_A = np.hstack([edge_index_bnd_ang])
                x_ang = np.concatenate([x_bnd_ang])
                mask_dih_ang = [False] * len(x_bnd_ang)

            data_amounts["x_atm"].append(len(x_atm) - 1)
            data_amounts["x_bnd"].append(len(tmp_x_bnd) - 1)
            data_amounts["x_ang"].append(len(x_ang) - 1)
            if dihedral:
                data_amounts["x_dih_ang"].append(len(x_dih_ang) - 1)

            data.append(Graph_Data(
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
        except:
            print('Failed graph gen...')
            data.append(None)
    return data

def atomic_alignnd_from_global_graph(global_graph,cutoff,dihedral=False, store_atoms=False,return_ids=False):
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
        edge_index_bnd_ang = line_graph(edge_index_G)
        x_bnd_ang = get_bnd_angs(global_graph['atoms'], edge_index_G, edge_index_bnd_ang)

        if dihedral:
            edge_index_dih_ang = dihedral_graph(edge_index_G)
            edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
            x_dih_ang = get_dih_angs(global_graph['atoms'], edge_index_G, edge_index_dih_ang)
            x_ang = np.concatenate([x_bnd_ang, x_dih_ang])
            mask_dih_ang = [False] * len(x_bnd_ang) + [True] * len(x_dih_ang)
        else:
            edge_index_A = np.hstack([edge_index_bnd_ang])
            x_ang = np.concatenate([x_bnd_ang])
            mask_dih_ang = [False] * len(x_bnd_ang)

        if store_atoms:
            atm = global_graph['atoms']

        data.append(Graph_Data(
            atoms = atm,
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
    if return_ids:
        return data, unique_atoms
    else:
        return data








