from pathlib import Path
import shutil
import os
import gc

from numba import cuda

import torch.multiprocessing as mp
import torch as torch
from torch import nn

from src.ml.nn.models.alignn import Encoder, Processor, PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.training import run_training, run_pre_training
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
            # mp.spawn(run_pre_training, args=(ml,), nprocs=ml_parameters['world_size'], join=True)
            ml.set_model()
            ml.model.share_memory()
            processes = []
            for rank in range(ml.parameters['world_size']):
                p = mp.Process(target=run_pre_training, args=(rank, ml,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            ddp_destroy()
        else:
            run_pre_training(rank=0, ml=ml)

    if ml.parameters['run_ddp']:
        print('Performing training...')
        ml.set_model()
        ml.model.share_memory()
        processes = []
        for rank in range(ml.parameters['world_size']):
            p = mp.Process(target=run_training, args=(rank, ml,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # ddp_destroy()
        # mp.spawn(run_training, args=(ml,), nprocs=ml.parameters['world_size'], join=True)
        ddp_destroy()
    else:
        run_training(rank=0, ml=ml)

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
                         pre_training=True,
                         run_pretrain=True,
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
                         loader_dict = dict(
                             shuffle_loader=False,
                             batch_size=[10,1000,1000],
                             num_workers=0
                         ),
                         model_dict = dict(
                             n_models=2,
                             num_epochs=[100,100],
                             train_tolerance=100000000.0,
                             accumulate_loss=['sum','exact','exact'],
                             loss_func=torch.nn.MSELoss(),
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
                                     'lr':0.01
                                 }
                             )
                         )
                    )

    main(ml_parameters)

    print("--- %s seconds ---" % (time.time() - start_time))




