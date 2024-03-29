from pathlib import Path
import shutil
import glob
import os

from torch_geometric.loader import DataLoader
import torch as torch

from catalyst.src.ml.testing import test_intepretable, test_non_intepretable
from catalyst.src.ml.ml import ML

import numpy as np

path = str(Path(__file__).parent)
models_dir = os.path.join(path,'models')
data_dir = os.path.join(path,'data')
results_dir = os.path.join(path,'results')
if os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.mkdir(results_dir)
models = glob.glob(os.path.join(models_dir,'model_*'))

ml_parameters = dict(gnn_dim = 10,
                        num_convs = 5,
                        num_inputs = 5,
                        num_epochs = 1000,
                        BATCH_SIZE = 1,
                        n_models = 1,
                        graph_cutoff = 4.0,
                        LEARN_RATE = 2e-4,
                        is_dihedral = False,
                        remove_old_model = False,
                        interpretable = True,
                        pre_training = False,
                        write_indv_pred = True,
                        device = 'cuda',
                        model_dir=None,
                        results_dir = None,
                        elements=['Co','Mo','Cu','Ni','Fe'])
ml = ML()
ml.set_params(ml_parameters,results_dir)
ml.set_model()

data = []
for graph in glob.glob(os.path.join(data_dir,'*.pt')):
    data.append(torch.load(graph))
follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data[0], 'x_ang') else ['x_atm']
loader = DataLoader(data, batch_size=ml.parameters['BATCH_SIZE'], shuffle=False, follow_batch=follow_batch)
print(f'Number of test graphs: {len(loader.dataset)}')

stats_file = open(os.path.join(results_dir, 'loss.data'), 'w')
for i, model_name in enumerate(models):
    print('Testing model ' + model_name.split('\\')[-1])
    final_dir = os.path.join(results_dir,str(i))
    os.mkdir(final_dir)
    ml.parameters['results_dir'] = final_dir
    ml.model = ml.model.to(ml.parameters['device'])
    ml.model.load_state_dict(torch.load(model_name))
    stats_file.write('Model_number     Test_loss\n')
    if ml.parameters['interpretable']:
        loss = test_intepretable(loader=loader,model=ml.model,parameters=ml.parameters)
    else:
        loss = test_non_intepretable(loader=loader, model=ml.model, parameters=ml.parameters)
    stats_file.write(str(i) + '     ' + str(loss) + '\n')
stats_file.close()





