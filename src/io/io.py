import numpy as np
import torch
import glob
import os

def read_training_data(params,samples_file,pretrain=False):
    training_graphs = []
    validation_graphs = []

    graph_files = glob.glob(os.path.join(params['graph_data_dir'],'*'))
    training_samples = np.load(samples_file,allow_pickle=True).item().get('training')
    for sample in training_samples:
        for graph in graph_files:
            fn = graph.split('\\')[-1].split('.')[0]
            if fn == sample:
                training_graphs.append(torch.load(graph))
    if pretrain == False:
        validation_samples = np.load(samples_file, allow_pickle=True).item().get('validation')
        for sample in validation_samples:
            for graph in graph_files:
                fn = graph.split('\\')[-1].split('.')[0]
                if fn == sample:
                    validation_graphs.append(torch.load(graph))

    return dict(training=training_graphs, validation=validation_graphs)








