from pathlib import Path
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

def visualize_graph(data):
    edge_index_bnd = data.edge_index_G.numpy()
    G = nx.Graph(list(edge_index_bnd.T))
    G_pos = nx.spring_layout(G)
    color_map = []
    colors = ['darkorange','aqua','limegreen','peru','mediumslateblue']
    for atm in data.x_atm:
        x = np.where(atm == 1.0)[0][0]
        color_map.append(colors[x])

    # Drawing options
    G_options = {
        'edgecolors': 'black',
        'width': 0.4,
        'font_size': 16,
        'node_size': 100,
    }
    nx.draw_networkx(G, G_pos, **G_options,with_labels=False,node_color=color_map,edge_color='dimgrey',arrows=False)
    plt.draw()
    plt.show()

path = str(Path(__file__).parent)
graph_file = 'graph.pt'
visualize_graph(torch.load(os.path.join(path,graph_file)))




