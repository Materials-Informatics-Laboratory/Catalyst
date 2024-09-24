from pathlib import Path
import shutil
import glob
import os

from torch_geometric.loader import DataLoader
import torch as torch
from torch import nn

from src.ml.nn.models.alignn import Encoder, Processor, PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.testing import predict_intepretable, predict_non_intepretable
from catalyst.src.ml.ml import ML

import numpy as np

path = str(Path(__file__).parent)
ml_parameters = dict(world_size = torch.cuda.device_count(),
                    gnn_dim=100,
                    num_convs=5,
                    num_inputs=5,
                    graph_cutoff = 4.0,
                    LEARN_RATE = 2e-4,
                    is_dihedral = False,
                    remove_old_model = False,
                    interpretable = True,
                    write_indv_pred = True,
                    run_ddp = False,
                    main_path=path,
                    device = 'cuda',
                    results_dir = None,
                     elements=['Al'],
                     model_dict=dict(
                         model=ALIGNN(
                             encoder=Encoder(num_species=1, cutoff=4.0, dim=10, act_func=nn.SiLU()),
                             processor=Processor(num_convs=5, dim=10, conv_type='mesh'),
                             decoder=PositiveScalarsDecoder(dim=10),
                         )
                     )
                )

models_dir = os.path.join(path,'models')
data_dir = os.path.join(path,'graphs')
results_dir = os.path.join(path,'results')
if os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.mkdir(results_dir)
models = glob.glob(os.path.join(models_dir,'model_*'))

ml = ML()
ml.set_params(ml_parameters,save_params=False)

data = []
for graph in glob.glob(os.path.join(data_dir,'*.pt')):
    data.append(torch.load(graph))
follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data[0], 'x_ang') else ['x_atm','x_bnd']
loader = DataLoader(data, batch_size=1, shuffle=False, follow_batch=follow_batch)
print(f'Number of test graphs: {len(loader.dataset)}')

stats_file = open(os.path.join(results_dir, 'loss.data'), 'w')
for i, model_name in enumerate(models):
    print('Testing model ' + model_name.split('\\')[-1])
    final_dir = os.path.join(results_dir,str(i))
    os.mkdir(final_dir)
    ml.parameters['results_dir'] = final_dir
    device = torch.device(ml.parameters['device'])
    ml.set_model()
    #ml.model.to(ml.parameters['device'])
    #device = torch.device(ml.parameters['device'])
    #model.load_state_dict(torch.load(os.path.join(ml.parameters['pretrain_dir'], 'model_pre'), map_location=device))
    ml.model.load_state_dict(torch.load(model_name,map_location=device))
    ml.model = ml.model.to(ml.parameters['device'])
    stats_file.write('Model_number     Test_loss\n')
    if ml.parameters['interpretable']:
        loss = predict_intepretable(loader=loader,model=ml.model,parameters=ml.parameters)
    else:
        loss = predict_non_intepretable(loader=loader, model=ml.model, parameters=ml.parameters)
    stats_file.write(str(i) + '     ' + str(loss) + '\n')
stats_file.close()





