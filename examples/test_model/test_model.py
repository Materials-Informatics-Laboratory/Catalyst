from pathlib import Path
import shutil
import glob
import os

from torch_geometric.loader import DataLoader
import torch as torch
from torch import nn

from src.ml.nn.models.alignn import Encoder, Processor, PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.testing import test_intepretable, test_non_intepretable
from catalyst.src.ml.ml import ML

import numpy as np

path = str(Path(__file__).parent)
ml_parameters = dict(
                               device_dict = dict(
                                   world_size=torch.cuda.device_count(),
                                   device='cuda',
                                   ddp_backend='gloo',
                                   run_ddp=False,
                                   pin_memory=False,
                               ),
                               io_dict = dict(
                                   main_path=path,
                                   restart_model_name='',
                                   graph_data_dir=os.path.join(path, '..\sodas_sampling\graph_data'),
                                   model_dir='',
                                   model_save_dir='',
                                   results_dir='',
                                   pretrain_dir=os.path.join(path, 'pre_training'),
                                   samples_dir=os.path.join(path, 'data'),
                                   remove_old_model=False,
                                   write_indv_pred=True,
                               ),
                               loader_dict=dict(
                                    shuffle_loader=True,
                                    batch_size=[100, 10, 10],
                                    num_workers=0,
                                    shuffle_steps=10
                                ),
                               model_dict=dict(
                                    n_models=2,
                                    num_epochs=[2,2],
                                    train_tolerance=0.0001,  # should probably make this a list for [pretrain,train]
                                    max_deltas=10,
                                    accumulate_loss=['sum', 'exact', 'exact'],
                                    loss_func=torch.nn.MSELoss(),
                                    interpretable=False,
                                    pre_training=True,
                                    restart_training=False,
                                    model=ALIGNN(
                                        encoder=Encoder(num_species=1, cutoff=4.0, dim=100, act_func=nn.SiLU()),
                                        processor=Processor(num_convs=5, dim=100, conv_type='mesh'),
                                        decoder=PositiveScalarsDecoder(dim=100),
                                    ),
                                    optimizer_params=dict(
                                        lr_scale=[1.0, 0.05],
                                        dynamic_lr=False,
                                        dist_type='exp',
                                        optimizer='AdamW',
                                        params_group={
                                            'lr': 0.01
                                        }
                                    )
                                )
                            )

models_dir = os.path.join(path,'models')
data_dir = os.path.join(path,'data')
results_dir = os.path.join(path,'results')
if os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.mkdir(results_dir)
models = glob.glob(os.path.join(models_dir,'model_*'))

ml = ML()
ml.set_params(ml_parameters)

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
    ml.parameters['io_dict']['results_dir'] = final_dir
    device = torch.device(ml.parameters['device_dict']['device'])
    ml.set_model()
    ml.model.load_state_dict(torch.load(model_name,map_location=device))
    ml.model = ml.model.to(ml.parameters['device_dict']['device'])
    stats_file.write('Model_number     Test_loss\n')
    if ml.parameters['model_dict']['interpretable']:
        loss = test_intepretable(loader=loader,model=ml.model,parameters=ml.parameters)
    else:
        loss = test_non_intepretable(loader=loader, model=ml.model, parameters=ml.parameters)
    stats_file.write(str(i) + '     ' + str(loss) + '\n')
stats_file.close()





