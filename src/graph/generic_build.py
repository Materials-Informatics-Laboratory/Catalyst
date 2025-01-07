from src.properties.structure_properties import get_3body_angle
from .graph import Generic_Graph_Data
from .graph import line_graph

import numpy as np
import torch

def generic_pairwise(data,data_params,gen_line_graph=True):
    ind = data_params['ind']
    dist = data_params['dist']
    g_nodes = data_params['g_nodes']

    g_edge_index = [[], []]
    a_nodes = []
    for i, x in enumerate(ind):
        for j, xx in enumerate(ind[i]):
            if j > 0:
                g_edge_index[0].append(x[0])
                g_edge_index[1].append(xx)
                a_nodes.append(dist[i][j])
    g_edge_index = np.array(g_edge_index)
    if gen_line_graph:
        a_edge_index = np.hstack([line_graph(g_edge_index)])
        a_edges = np.concatenate([get_3body_angle(data, g_edge_index, a_edge_index)])
        graph = Generic_Graph_Data(
            node_G=torch.tensor(g_nodes, dtype=torch.float),
            edge_index_G=torch.tensor(g_edge_index, dtype=torch.long),
            node_A=torch.tensor(a_nodes, dtype=torch.float),
            edge_index_A=torch.tensor(a_edge_index, dtype=torch.long),
            edge_A=torch.tensor(a_edges, dtype=torch.float),
            node_G_amounts=torch.tensor(len(g_nodes) - 1, dtype=torch.long),
            node_A_amounts=torch.tensor(len(a_nodes) - 1, dtype=torch.long),
            edge_A_amounts=torch.tensor(len(a_edges) - 1, dtype=torch.long)
        )
    else:
        graph = Generic_Graph_Data(
            node_G=torch.tensor(g_nodes, dtype=torch.float),
            edge_index_G=torch.tensor(g_edge_index, dtype=torch.long),
            node_A=torch.tensor(a_nodes, dtype=torch.float),
            edge_index_A=None,
            edge_A=None,
            node_G_amounts=torch.tensor(len(g_nodes) - 1, dtype=torch.long),
            node_A_amounts=torch.tensor(len(a_nodes) - 1, dtype=torch.long),
            edge_A_amounts=None
        )
    graph.generate_gid()

    return graph