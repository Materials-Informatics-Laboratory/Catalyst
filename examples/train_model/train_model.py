from pathlib import Path
import numpy as np
import shutil
import glob
import sys
import os
import gc

from numba import cuda
import torch

from umap import umap_

from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch as torch
from torch import nn

from catalyst.src.sodas.nn.models.alignn import Encoder, Processor, PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.training import run_training, run_pre_training
import catalyst.src.utilities.sampling as sampling
from catalyst.src.sodas.model.sodas import SODAS
from catalyst.src.sodas.nn import MLP
from catalyst.src.ml.ml import ML

torch.set_float32_matmul_precision

# ddp_destory is needed when calling mp.spawn() multiple times within a SLURM or ORACLE scheduling environment
def ddp_destroy():
    # collect memory via garbage collection
    gc.collect()
    # loop through active devices (assumes you are using all devices available), clear their memory, and then reset the device and close cuda
    for gpu_id in range(torch.cuda.device_count()):
        cuda.select_device(gpu_id)
        torch.cuda.empty_cache()
        device = cuda.get_current_device()
        device.reset()
        cuda.close()

def main(ml_parameters):
    # initialize ML class
    ml = ML()
    ml.set_params(ml_parameters)
    
    # create model directories
    if ml_parameters['restart_training']:
        ml.parameters['model_dir'] = os.path.join(ml_parameters['main_path'], 'models_restart')
    else:
        ml.parameters['model_dir'] = os.path.join(ml_parameters['main_path'], 'models')
    if os.path.isdir(ml.parameters['model_dir']):
        shutil.rmtree(ml.parameters['model_dir'])
    os.mkdir(ml.parameters['model_dir'])

    # run pretraining  
    if ml.parameters['run_pretrain']:
        print('Performing pretraining...')

        if ml.parameters['run_ddp']:
            mp.spawn(run_pre_training, args=(ml,), nprocs=ml_parameters['world_size'], join=True)
            ddp_destroy()
        else:
            run_pre_training(rank=0, ml=ml)

    if ml.parameters['run_ddp']:
        mp.spawn(run_training, args=(ml,), nprocs=ml.parameters['world_size'],
                join=True)
        ddp_destroy()
    else:
        run_training(rank=0,ml=ml)
        
if __name__ == "__main__":
    import time
    start_time = time.time()

    # setup parameters
    path = str(Path(__file__).parent)
    ml_parameters = dict(
                         world_size = torch.cuda.device_count(),
                         sampling_seed=12345,
                         remove_old_model=False,
                         interpretable=False,
                         pre_training=False,
                         run_pretrain=False,
                         write_indv_pred=False,
                         restart_training=False,
                         run_ddp = False,
                         pin_memory=False,
                         main_path=path,
                         ddp_backend='gloo',
                         device='cuda',
                         restart_model_name=None,
                         graph_data_dir=os.path.join(path, 'data'),
                         model_dir=None,
                         model_save_dir=None,
                         pretrain_dir=os.path.join(path, 'pre_training'),
                         results_dir=None,
                         elements=['Al'],
                         model_dict = dict(
                             n_models=1,
                             num_epochs=[10, 100],
                             train_tolerance=1e-5,
                             batch_size=[10,1000],
                             model = ALIGNN(
                                    encoder=Encoder(num_species=1,cutoff=4.0,dim=10,act_func=nn.SiLU()),
                                    processor=Processor(num_convs=5, dim=10,conv_type='mesh'),
                                    decoder=PositiveScalarsDecoder(dim=10),
                            ),
                             optimizer_params=dict(
                                 lr_scale=[1.0, 0.05],
                                 dynamic_lr=False,
                                 dist_type='exp',
                                 optimizer = 'AdamW',
                                 params_group = {
                                     'lr':0.001
                                 }
                             )
                         )
                    )

    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main(ml_parameters)

    print("--- %s seconds ---" % (time.time() - start_time))




