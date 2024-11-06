from ....src.graph.graph import Generic_Graph_Data,Atomic_Graph_Data
from sklearn.neighbors import KDTree
from ase.atoms import Atoms
import networkx as nx
import numpy as np
import math

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
            if isinstance(snapshot, Atomic_Graph_Data) or isinstance(snapshot, Generic_Graph_Data):
                if hasattr(snapshot, 'x_atm'):
                    symbols = snapshot['x_atm']
                elif hasattr(snapshot, 'node_G'):
                    symbols = snapshot['node_G']

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
                        G = nx.Graph()
                        i = np.where(np.array(interaction[0]) != 0)[0][0]
                        j = np.where(np.array(interaction[1]) != 0)[0][0]
                        kks = []
                        for symbol in symbols:
                            kks.append(np.where(np.array(symbol) != 0)[0][0])
                        if i == j:
                            idx = np.where(np.array(kks) == i)[0]
                            for ii, idi in enumerate(idx):
                                G.add_node(ii)
                                for jj, idj in enumerate(idx):
                                    if ii != jj:
                                        edge_id = np.where(np.logical_and(np.array(snapshot['edge_index_G'][0]) == idi,
                                                                          np.array(snapshot['edge_index_G'][1]) == idj))
                                        if len(edge_id[0]) > 0:
                                            if hasattr(snapshot, 'x_atm'):
                                                edge = snapshot['x_bnd'][edge_id]
                                            elif hasattr(snapshot, 'node_G'):
                                                edge = snapshot['node_A'][edge_id]
                                            if rc > -1.0 and edge < rc:
                                                edge = 1.0 / edge
                                            G.add_edge(ii, jj, weight=edge.item())
                        else:
                            idx = []
                            idx.append(np.where(np.array(kks) == i)[0])
                            idx.append(np.where(np.array(kks) == j)[0])

                            for ii, idi in enumerate(idx[0]):
                                G.add_node(ii)
                                for jj, idj in enumerate(idx[1]):
                                    if ii != jj:
                                        edge_id = np.where(np.logical_and(np.array(snapshot['edge_index_G'][0]) == idi,
                                                                          np.array(snapshot['edge_index_G'][1]) == idj))
                                        if len(edge_id[0]) > 0:
                                            if len(idx[0]) + jj not in list(G.nodes):
                                                G.add_node(len(idx[0]) + jj)
                                            if hasattr(snapshot, 'x_atm'):
                                                edge = snapshot['x_bnd'][edge_id]
                                            elif hasattr(snapshot, 'node_G'):
                                                edge = snapshot['node_A'][edge_id]
                                            if rc > -1.0 and edge < rc:
                                                edge = 1.0 / edge
                                            G.add_edge(ii, len(idx[0]) + jj, weight=edge.item())
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
                                    r = math.dist(atom_i.position, atom_j.position)
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
                                    r = math.dist(atom_i.position, atom_j.position)
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
            return predictions, flattened_predictions
        else:
            return predictions








