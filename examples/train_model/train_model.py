from pathlib import Path
import numpy as np
import shutil
import glob
import sys
import os

from numba import cuda
import torch
import gc

from umap import umap_

from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch as torch
from torch import nn

from catalyst.src.sodas.nn.models.alignn import Encoder, Processor, Decoder, ALIGNN
from catalyst.src.ml.training import run_training, run_pre_training
import catalyst.src.utilities.sampling as sampling
from catalyst.src.sodas.model.sodas import SODAS
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

        # load pretraining data
        if os.path.isdir(ml.parameters['pretrain_dir']):
            shutil.rmtree(ml.parameters['pretrain_dir'])
        os.mkdir(ml.parameters['pretrain_dir'])
        ml.set_model()
        partitioned_data = np.load(os.path.join(ml.parameters['graph_data_dir'],'pre_training','train_valid_split.npy'),allow_pickle=True)

        samples = dict(training=partitioned_data.item().get('training'),validation=None)
        # perform pretraining
        if ml.parameters['run_ddp']:
            mp.spawn(run_pre_training, args=(samples, ml.parameters, ml.model), nprocs=ml_parameters['world_size'], join=True)
            ddp_destroy()
        else:
            run_pre_training(rank=0, data=samples, parameters=ml.parameters, model=ml.model)

    # loop over number of requested models
    #for iteration in range(ml.parameters['n_models']):
        '''
        ml.parameters['model_save_dir'] = os.path.join(ml.parameters['model_dir'], str(iteration))
        os.mkdir(ml.parameters['model_save_dir'])
        ml.set_model()
        if ml_parameters['restart_training']:
            print('Restarting model training using model ',ml_parameters['restart_model_name'])
        print('Training model ',iteration)

        partitioned_data = np.load(os.path.join(ml.parameters['graph_data_dir'],'model_samples',str(iteration),'train_valid_split.npy'),allow_pickle=True)
 
        samples = dict(training=partitioned_data.item().get('training'),validation=partitioned_data.item().get('validation'))
        '''
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
    ml_parameters = dict(gnn_dim=10,
                         num_convs=5,
                         num_inputs=1,
                         num_epochs=100,
                         BATCH_SIZE=4000,
                         n_models=1,
                         world_size = torch.cuda.device_count(),
                         sampling_seed=12345,
                         graph_cutoff=4.0,
                         LEARN_RATE=2e-4,
                         train_tolerance=1e-5,
                         is_dihedral=False,
                         remove_old_model=False,
                         interpretable=False,
                         pre_training=False,
                         run_pretrain=False,
                         write_indv_pred=False,
                         restart_training=False,
                         run_ddp = True,
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
                         samples_dir=None,
                         elements=['Al'],
                         sampling_dict=dict(test_sampling_type='y_bin',
                                            pretraining_sampling_type='kmeans',
                                            sampling_type='gaussian_mixture',
                                            split=[0.1, 0.25, 0.75],
                                            clusters=3
                                            ),
                        )

    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main(ml_parameters)

    print("--- %s seconds ---" % (time.time() - start_time))




