from pathlib import Path
import shutil
import time
import os
import gc

import torch.multiprocessing as mp
import torch as torch
from torch import nn

from src.ml.nn.models.alignn import Encoder, Processor, PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.training import run_training, run_pre_training
from catalyst.src.ml.utils.distributed import cuda_destroy
from catalyst.src.ml.ml import ML

torch.set_float32_matmul_precision

def main(ml_parameters):
    ml = ML()
    ml.set_params(ml_parameters)

    # create model directories
    if ml_parameters['model_dict']['restart_training']:
        ml.parameters['io_dict']['model_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'models_restart')
    else:
        ml.parameters['io_dict']['model_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'models')
    if os.path.isdir(ml.parameters['io_dict']['model_dir']):
        shutil.rmtree(ml.parameters['io_dict']['model_dir'])
    os.mkdir(ml.parameters['io_dict']['model_dir'])

    # run pretraining
    if ml.parameters['model_dict']['pre_training']:
        print('Performing pretraining...')

        if ml.parameters['device_dict']['run_ddp']:
            processes = []
            for rank in range(ml.parameters['device_dict']['world_size']):
                p = mp.Process(target=run_pre_training, args=(rank, ml,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cuda_destroy()
        else:
            run_pre_training(rank=0, ml=ml)

    for iteration in range(ml.parameters['model_dict']['n_models']):
        print('Performing training on model ', iteration)
        if ml.parameters['device_dict']['run_ddp']:
            processes = []
            for rank in range(ml.parameters['device_dict']['world_size']):
                p = mp.Process(target=run_training, args=(rank, iteration,ml,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cuda_destroy()
        else:
            run_training(rank=0, iteration=iteration,ml=ml)

if __name__ == "__main__":
    start_time = time.time()

    # setup parameters
    path = str(Path(__file__).parent)
    ml_parameters = dict(
                               device_dict = dict(
                                   world_size=torch.cuda.device_count(),
                                   device='cuda',
                                   ddp_backend='gloo',
                                   run_ddp=True,
                                   pin_memory=False,
                               ),
                               io_dict = dict(
                                   main_path=path,
                                   data_dir=os.path.join(path, '..\sodas_sampling\graph_data'),
                                   loaded_model_name='',
                                   samples_dir=[os.path.join(path, 'data','pre_training'),os.path.join(path, 'data','model_samples')],
                                   remove_old_model=False,
                                   write_indv_pred=False,
                               ),
                               loader_dict=dict(
                                    shuffle_loader=True,
                                    batch_size=[2,2,2],
                                    num_workers=0,
                                    shuffle_steps=10
                                ),
                               model_dict=dict(
                                    n_models=2,
                                    num_epochs=[1,1],
                                    train_tolerance=0.0001,  # should probably make this a list for [pretrain,train]
                                    max_deltas=4,
                                    accumulate_loss=['sum', 'sum', 'sum'],
                                    loss_func=torch.nn.MSELoss(),
                                    interpretable=False,
                                    pre_training=True,
                                    restart_training=False,
                                    model=ALIGNN(
                                        encoder=Encoder(num_species=119, cutoff=4.0, dim=100, act_func=nn.SiLU()),
                                        processor=Processor(num_convs=5, dim=100, conv_type='mesh'),
                                        decoder=PositiveScalarsDecoder(dim=100),
                                    ),
                                    optimizer_params=dict(
                                        dynamic_lr=False,
                                        optimizer='AdamW',
                                        params_group={
                                            'lr': 0.001
                                        }
                                    )
                                )
                            )

    main(ml_parameters)

    print("--- %s seconds ---" % (time.time() - start_time))




