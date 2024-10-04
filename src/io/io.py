import numpy as np
import torch
import glob
import os

def read_training_data(params,samples_file,pretrain=False):
    training_graphs = []
    validation_graphs = []

    graph_files = glob.glob(os.path.join(params['graph_data_dir'],'*'))
    samples = np.load(samples_file, allow_pickle=True)
    training_samples = samples.item().get('training')

    graph_names = [graph.split('\\')[-1].split('.')[0] for graph in graph_files]
    cross_list = set(training_samples).intersection(graph_names)
    idx = [graph_names.index(x) for x in cross_list]
    selected_graphs = [graph_files[i] for i in idx]
    training_graphs = [torch.load(x) for x in selected_graphs]

    if pretrain == False:
        validation_samples = samples.item().get('validation')
        cross_list = set(validation_samples).intersection(graph_names)
        idx = [graph_names.index(x) for x in cross_list]
        selected_graphs = [graph_files[i] for i in idx]
        validation_graphs = [torch.load(x) for x in selected_graphs]

    return dict(training=training_graphs, validation=validation_graphs)

def port_graphs_without_gids(params):
    print('Porting graphs')
    graph_files = glob.glob(os.path.join(params['graph_data_dir'], '*'))

    check = 1
    for graph in graph_files:
        data = torch.load(graph)
        if not hasattr(data, 'gid'):
            if check:
                from ..graph.graph import Graph_Data
                check = 0

            if not hasattr(data, 'atoms'):
                atoms = None
            else:
                atoms = data.atoms
            if not hasattr(data, 'y'):
                y = None
            else:
                y = data.y
            if not hasattr(data, 'x_ang'):
                x_ang = None
            else:
                x_ang = data.x_ang
            if not hasattr(data, 'mask_dih_ang'):
                mask_dih_ang = None
            else:
                mask_dih_ang = data.mask_dih_ang
            if not hasattr(data, 'ang_amounts'):
                ang_amounts = None
            else:
                ang_amounts = data.ang_amounts

            new_data = Graph_Data(
                atoms=atoms,
                edge_index_G=data.edge_index_G,
                edge_index_A=data.edge_index_A,
                x_atm=data.x_atm,
                x_bnd=data.x_bnd,
                x_ang=x_ang,
                mask_dih_ang=mask_dih_ang,
                atm_amounts=data.atm_amounts,
                bnd_amounts=data.bnd_amounts,
                ang_amounts=ang_amounts,
            )
            new_data.y = y
            data = new_data
            data.generate_gid()
        elif data.gid is None:
            data.generate_gid()
        os.remove(graph)
        torch.save(data, os.path.join(params['graph_data_dir'], data.gid + '.pt'))
