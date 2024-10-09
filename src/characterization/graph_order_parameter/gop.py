from sklearn.neighbors import KDTree
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

    def predict(self,data):
        '''
        data = ASE atoms object
        '''
        predictions = []

        for snapshot in data:
            predictions.append([])
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
                        idx = np.where(np.array(snapshot.get_chemical_symbols()) == symbol)
                        selected_atoms.append([snapshot[i] for i in idx])
                        for i, atom_i in enumerate(selected_atoms[0][0]):
                            G.add_node(i)
                            for j, atom_j in enumerate(selected_atoms[0][0]):
                                r = math.dist(atom_i.position,atom_j.position)
                                if r < rc and r > .01:
                                    G.add_edge(i,j,weight=1.0/r)
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
                                r = math.dist(atom_i.position,atom_j.position)
                                if r < rc and r > .01:
                                    G.add_edge(i,len(selected_atoms[0][0]) + j,weight=1.0/r)
                    subgraphs = list(nx.connected_components(G))
                    predictions[-1][-1][-1].append(0.0)
                    for sg in subgraphs:
                        op = 0.0
                        degrees = []
                        for node in sg:
                            degrees.append(0.0)
                            edges = list(G.edges(node))
                            if len(edges) > 0:
                                for edge in edges:
                                    degrees[-1] += G[edge[0]][edge[1]]['weight']
                        unique_degrees, counts = np.unique(degrees,return_counts=True)
                        for ii, deg in enumerate(unique_degrees):
                            p_ii = counts[ii]/sum(counts)
                            op += p_ii*math.log(p_ii) + deg*p_ii
                        predictions[-1][-1][-1][-1] += math.pow(op,self.params['k'])
        return predictions








