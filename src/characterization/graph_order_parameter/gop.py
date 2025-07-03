from ...graph.graph import Generic_Graph_Data,Atomic_Graph_Data
from sklearn.neighbors import KDTree
from ase.atoms import Atoms
from ase.geometry import get_distances
import networkx as nx
import numpy as np
import math

from itertools import product

class GOP():
    def __init__(self,params=dict(
                        cutoffs=[], # [3.0,4.0,5.0,...]
                        interactions=[], # [['Al','Al'],...]
                        k=1
                    )
                 ):
        super().__init__()

        self.params = params

    def calc_gop(self,G):
        subgraphs = list(nx.connected_components(G))
        op = 0.0
        for sg in subgraphs:
            sg_op = 0.0
            degrees = []
            for node in sg:
                degrees.append(0.0)
                edges = list(G.edges(node))
                if len(edges) > 0:
                    for edge in edges:
                        degrees[-1] += G[edge[0]][edge[1]]['weight']
            unique_degrees, counts = np.unique(degrees, return_counts=True)
            for ii, deg in enumerate(unique_degrees):
                p_ii = counts[ii] / sum(counts)
                sg_op += p_ii * math.log(p_ii) + deg * p_ii
            op += math.pow(sg_op, self.params['k'])
        return op

    def predict(self,data,flatten=False):
        '''
        data = ASE atoms object
        '''
        predictions = []
        for snapshot in data:
            predictions.append([])
            graph_type = [0,0]
            if isinstance(snapshot, Atomic_Graph_Data) or isinstance(snapshot, Generic_Graph_Data):
                if hasattr(snapshot, 'x_atm'):
                    symbols = snapshot['x_atm']
                    graph_type[0] = 1
                elif hasattr(snapshot, 'node_G'):
                    symbols = snapshot['node_G']
                    graph_type[1] = 1


                for interaction in self.params['interactions']:
                    i = np.where(np.array(interaction[0]) != 0)[0][0]
                    j = np.where(np.array(interaction[1]) != 0)[0][0]


                    check = [0, 0]
                    for symbol in symbols:
                        k = np.where(np.array(symbol) != 0)[0][0]
                        if k == i:
                            check[0] = 1
                        if k == j:
                            check[1] = 1
                    if sum(check) < 2:
                        print('Requested interaction not possible with given graph object...killing run...')
                        exit(0)
                for rc in self.params['cutoffs']:
                    predictions[-1].append([])
                    for interaction in self.params['interactions']:
                        predictions[-1][-1].append([])

                        i = np.where(np.array(interaction[0]) != 0)[0][0]
                        j = np.where(np.array(interaction[1]) != 0)[0][0]
                        kks = []
                        for symbol in symbols:
                            kks.append(np.where(np.array(symbol) != 0)[0][0])
                        edge_list = []
                        if i == j:
                            edge_index = np.array(snapshot['edge_index_G'])
                            edge_lookup = {(u, v): idx for idx, (u, v) in enumerate(zip(edge_index[0], edge_index[1]))}
                            idx = np.where(np.array(kks) == i)[0]
                            for i_, j_ in product(idx, idx):
                                if i_ == j_:
                                    continue

                                edge_id = edge_lookup.get((i_, j_))
                                if edge_id is None:
                                    continue

                                if graph_type[0]:
                                    edge = snapshot['x_bnd'][edge_id]
                                elif graph_type[1]:
                                    edge = snapshot['node_A'][edge_id]

                                if edge < rc:
                                    edge_weight = 1.0 / edge
                                    edge_list.append((i_, j_, {'weight': edge_weight.item()}))
                            '''
                            edge_list = []
                            idx = np.where(np.array(kks) == i)[0]
                            edge_index = np.array(snapshot['edge_index_G'])
                            for i, j in product(idx,idx):
                                if i != j:
                                    mask = (edge_index[0] == i) & (edge_index[1] == j)
                                    if any(mask):
                                        edge_id = np.where(mask)[0]
                                        if graph_type[0]:
                                            edge = snapshot['x_bnd'][edge_id]
                                        elif graph_type[1]:
                                            edge = snapshot['node_A'][edge_id]
                                        if edge < rc:
                                            edge = 1.0 / edge
                                            edge_list.append((i, j, {'weight': edge.item()}))
                            '''
                            '''              
                            for ii, idi in enumerate(idx):
                                #G.add_node(ii)
                                print(ii)
                                for jj, idj in enumerate(idx):
                                    if ii != jj:
                                        #edge_id = np.where(np.logical_and(np.array(snapshot['edge_index_G'][0]) == idi,
                                        #                                  np.array(snapshot['edge_index_G'][1]) == idj))
                                        mask = (edge_index[0] == idi) & (edge_index[1] == idj)
                                        if any(mask):
                                            edge_id = np.where(mask)[0]
                                            if graph_type[0]:
                                                edge = snapshot['x_bnd'][edge_id]
                                            elif graph_type[1]:
                                                edge = snapshot['node_A'][edge_id]
                                            if edge < rc:
                                                edge = 1.0 / edge
                                                #edge_list.append((ii, jj, {'weight':edge.item()}))
                            '''
                        else:
                            idx_i = np.where(np.array(kks) == i)[0]
                            idx_j = np.where(np.array(kks) == j)[0]
                            edge_index = np.array(snapshot['edge_index_G'])
                            edge_lookup = {(u, v): eid for eid, (u, v) in enumerate(zip(edge_index[0], edge_index[1]))}

                            for i_, j_ in product(idx_i, idx_j):
                                if i_ == j_:
                                    continue  # skip same enumeration index pairs if needed

                                edge_id = edge_lookup.get((i_, j_))
                                if edge_id is None:
                                    continue

                                if hasattr(snapshot, 'x_atm'):
                                    edge = snapshot['x_bnd'][edge_id]
                                elif hasattr(snapshot, 'node_G'):
                                    edge = snapshot['node_A'][edge_id]

                                if edge < rc:
                                    edge_weight = 1.0 / edge
                                    edge_list.append((i_, j_, {'weight': edge_weight.item()}))
                        G = nx.Graph(edge_list)
                        predictions[-1][-1][-1].append(self.calc_gop(G))
            else:
                symbols = np.unique(snapshot.get_chemical_symbols())
                for interaction in self.params['interactions']:
                    check = 0
                    for symbol in symbols:
                        if symbol == interaction[0]:
                            check += 1
                        if symbol == interaction[1]:
                            check += 1
                    if check < 2:
                        print('Requested interaction not possible with given Atoms object...killing run...')
                        exit(0)
                for rc in self.params['cutoffs']:
                    predictions[-1].append([])
                    for interaction in self.params['interactions']:
                        predictions[-1][-1].append([])
                        G = nx.Graph()
                        selected_atoms = []
                        if interaction[0] == interaction[1]:
                            idx = np.where(np.array(snapshot.get_chemical_symbols()) == interaction[0])
                            selected_atoms.append([snapshot[i] for i in idx])
                            for i, atom_i in enumerate(selected_atoms[0][0]):
                                G.add_node(i)
                                for j, atom_j in enumerate(selected_atoms[0][0]):
                                    r = get_distances(atom_i.position, atom_j.position,cell=snapshot.get_cell(),pbc=True)[1][0][0]
                                    if r < rc and r > .01:
                                        G.add_edge(i, j, weight=1.0 / r)
                        else:
                            idx = []
                            for symbol in interaction:
                                idx.append(np.where(np.array(snapshot.get_chemical_symbols()) == symbol))
                            selected_atoms.append([snapshot[i] for i in idx[0]])
                            selected_atoms.append([snapshot[i] for i in idx[1]])
                            for i, atom_i in enumerate(selected_atoms[0][0]):
                                G.add_node(i)
                                for j, atom_j in enumerate(selected_atoms[1][0]):
                                    if i == 0:
                                        G.add_node(len(selected_atoms[0][0]) + j)
                                    r = get_distances(atom_i.position, atom_j.position,cell=snapshot.get_cell(),pbc=True)[1][0][0]
                                    if r < rc and r > .01:
                                        G.add_edge(i, len(selected_atoms[0][0]) + j, weight=1.0 / r)
                        predictions[-1][-1][-1].append(self.calc_gop(G))
        if flatten:
            flattened_predictions = []
            for pred in predictions:
                flattened_predictions.append([])
                for p1 in pred:
                    for p2 in p1:
                        if len(flattened_predictions[-1]) == 0:
                            flattened_predictions[-1] = p2
                        else:
                            flattened_predictions[-1] = flattened_predictions[-1] + p2
            return predictions, np.array(flattened_predictions)
        else:
            return predictions








