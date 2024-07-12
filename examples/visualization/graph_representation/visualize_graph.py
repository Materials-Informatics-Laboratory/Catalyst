from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

def visualize_graph(data,atomic=False):
    # Drawing options
    G_options = {
        'edgecolors': 'black',
        'width': 0.4,
        'font_size': 16,
        'node_size': 100,
    }

    edge_index_bnd = data.edge_index_G.numpy()
    G = nx.Graph(list(edge_index_bnd.T))
    G_pos = nx.spring_layout(G)
    color_map = []
    colors = ['aqua','mediumslateblue','peru','limegreen','darkorange']
    if atomic:
        fig, ax = plt.subplots(1,2)
        idx = np.unique(data.edge_index_G[0])
        for id in idx:
            x = np.where(data.x_atm[id] == 1.0)[0][0]
            color_map.append(colors[x])

        edge_index_bnd = data.edge_index_A.numpy()
        A = nx.Graph(list(edge_index_bnd.T))
        A_pos = nx.spring_layout(A)
        nx.draw_networkx(A, A_pos, **G_options, with_labels=False, edge_color='dimgrey',
                         arrows=False,ax=ax[1])
        nx.draw_networkx(G, G_pos, **G_options, with_labels=False, node_color=color_map, edge_color='dimgrey',
                         arrows=False,ax=ax[0])
    else:
        for atm in data.x_atm:
            x = np.where(atm == 1.0)[0][0]
            color_map.append(colors[x])

        fig, ax = plt.subplots(1, 2)
        edge_index_bnd = data.edge_index_A.numpy()
        A = nx.Graph(list(edge_index_bnd.T))
        A_pos = nx.spring_layout(A)
        nx.draw_networkx(A, A_pos, **G_options, with_labels=False, edge_color='dimgrey',
                             arrows=False, ax=ax[1])

        nx.draw_networkx(G, G_pos, **G_options, with_labels=False, node_color=color_map, edge_color='dimgrey',
                         arrows=False, ax=ax[0])
    plt.draw()
    plt.show()

path = str(Path(__file__).parent)
graphs = glob.glob(os.path.join(path,'data','*.pt'))
for graph in graphs:
    visualize_graph(torch.load(os.path.join(path,graph)),atomic=False)




