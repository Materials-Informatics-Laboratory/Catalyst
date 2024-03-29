from ..utilities.structure_properties import *
from ..graphite.utils.alignn import *
from .graph.graph import Graph_Data
from .graph.graph import atoms2graph
import torch

def alignnd(atoms,cutoff,dihedral=False):
    """Converts ASE `atoms` into a PyG graph data holding the atomic graph (G) and the angular graph (A).
    The angular graph holds bond angle information, but can also calculate dihedral information upon request.
    """

    elements = np.unique(atoms.get_chemical_symbols())
    ohe = []
    for atom in atoms:
        tx = [0.0] * len(elements)
        for i in range(len(elements)):
            if atom.symbol == elements[i]:
                tx[i] = 1.0
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

    data = Graph_Data(
        edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
        edge_index_A=torch.tensor(edge_index_A, dtype=torch.long),
        x_atm=torch.tensor(x_atm, dtype=torch.float),
        x_bnd=torch.tensor(x_bnd, dtype=torch.float),
        x_ang=torch.tensor(x_ang, dtype=torch.float),
        mask_dih_ang=torch.tensor(mask_dih_ang, dtype=torch.bool)
    )

    return data

def realignnd(structures,cutoff,dihedral=False):
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

    data_amounts = dict(x_atm=[], x_bnd=[], x_ang=[])
    if dihedral:
        data_amounts.append(x_dih_ang=[])

    scale = 0
    all_elements = []
    for atoms in structures:
        elements = np.unique(atoms.get_chemical_symbols())
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
                if atom.symbol == all_elements[i]:
                    tx[i] = 1.0
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
        else:
            f_edge_index_G = edge_index_G
            f_edge_index_A = edge_index_A
            f_x_atm = x_atm
            f_x_ang = x_ang
            f_x_bnd = x_bnd
            f_mask_dih_ang = mask_dih_ang
            if dihedral:
                f_x_dih_ang = x_dih_ang
        scale += len(atoms)

    data = Graph_Data(
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


