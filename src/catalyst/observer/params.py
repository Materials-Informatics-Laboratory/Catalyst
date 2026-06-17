from ..ml.utils.distributed import set_spawn_method
from ..data.utils import save_dictionary
from ..ml.utils.memory import clear_torch_memory, change_model_device
from torch import nn
import numpy as np
import torch
import os
import gc

import platform
import psutil
import GPUtil

class Catalyst():
    def __init__(self):
        super().__init__()

        self.parameters = dict(
                               device_dict = dict(
                                   world_size=1,
                                   device='',
                                   ddp_backend='',
                                   run_ddp=False,
                                   pin_memory=False,
                                   find_unused_parameters=False,
                                   system_info = None
                               ),
                               io_dict = dict(
                                   main_path='',
                                   loaded_model_name='',
                                   data_dir='',
                                   model_dir='',
                                   results_dir='',
                                   samples_dir='',
                                   projection_dir='',
                                   remove_old_model=False,
                                   write_indv_pred=False,
                                   graph_read_format=0
                               ),
                               sampling_dict = dict(sampling_types=['random','random'],
                                                    split=[0.5,0.5],
                                                    sampling_seed=112358,
                                                    params_groups = [{
                                                        'clusters':1,
                                                    },{
                                                        'clusters':1,
                                                    }]
                                ),
                                loader_dict=dict(
                                    shuffle_loader=False,
                                    batch_size=[1,1],
                                    shuffle_steps=10,
                                    num_workers=0
                                ),
                               model_dict = dict(
                                   n_models=1,
                                   num_epochs=1,
                                   train_delta=0.001,
                                   train_tolerance=1.0,
                                   worsen_tolerance=0.05,
                                   max_deltas=4,
                                   accumulate_loss='exact',
                                   loss_params={
                                       'function':None
                                   },
                                   model = None,
                                   strict_loss_policy=True,
                                   model_params_group=dict(
                                        encoder=dict(

                                        ),
                                        processor=dict(

                                        ),
                                        decoder=dict(

                                        )
                                   ),
                                   active_learning=False,
                                   active_learning_params_group=dict(
                                       sampling_params_group={
                                           'algorithm': 'property',
                                           'exploration_weight': 0.5,
                                           'samples_per_iteration': 2,
                                           'exploitation_strategy': 'greedy'
                                       },
                                       training_params_group=dict(
                                           train_with_previous=False,
                                           percent_use_previous=0.0,
                                           epochs_per_iteration=1,
                                           iterations=1,
                                           loss_regularization='EWC',
                                           regularization_params_group={
                                               'lambda': 1E9
                                           }
                                       ),
                                       training_data_dir=''
                                   ),
                                   interpretable=False,
                                   restart_training=False,
                                   optimizer_params=dict(
                                       lr_scale=[1.0, 0.1],
                                       dynamic_lr=False,
                                       optimizer='',
                                       params_group={
                                           'lr': 0.001,
                                           'lr_decay_factor': 0.5
                                       }
                                   )
                               )
                            )

        self.accumulate_loss_options = ['exact','sum']
        self.device_options = ['cuda','cpu']
        self.optimizer_options = ['AdamW','Adadelta','Adagrad','Adam','SparseAdam','Adamax','ASGD',
                                  'LBFGS','NAdam','RAdam','RMSprop','Rprop','SGD']
        self.version = '1.1'

        '''
        graph clustering params
        {
            'leaf_size':2,
            'neighbors':10,
            'metric':'minkowski'
        }
        '''

    def get_system_info(self):
        system_info = platform.uname()
        memory_info = psutil.virtual_memory()
        gpus = GPUtil.getGPUs()

        if not gpus:
            info = dict(
                system=system_info.system,
                node=system_info.node,
                release=system_info.release,
                version=system_info.version,
                machine=system_info.machine,
                processor=system_info.processor,
                cpu_count=psutil.cpu_count(logical=False),
                logical_count=psutil.cpu_count(logical=True),
                total_memory=memory_info.total,
                ngpus=len(gpus),
            )
        else:
            info = dict(
                system=system_info.system,
                node=system_info.node,
                release=system_info.release,
                version=system_info.version,
                machine=system_info.machine,
                processor=system_info.processor,
                cpu_count=psutil.cpu_count(logical=False),
                logical_count=psutil.cpu_count(logical=True),
                total_memory=memory_info.total,
                ngpus=len(gpus),
                gpu_type=gpus[0].name,
                gpu_driver=gpus[0].driver,
                gpu_memory=gpus[0].memoryTotal
            )
        self.parameters['device_dict']['system_info'] = info


    def set_model(self,model):
        if self.parameters['model_dict']['model'] is None:
            self.parameters['model_dict']['model'] = model
        else:
            del self.parameters['model_dict']['model']
            self.parameters['model_dict']['model'] = None
            clear_torch_memory()
            self.parameters['model_dict']['model'] = model
        change_model_device(self.parameters['model_dict']['model'],self.parameters['device_dict']['device'])

    def set_params(self,new_params,save_params=True):
        set_spawn_method(new_params)
        new_params['device_dict']['system_info'] = self.get_system_info()
        self.parameters = new_params

        if save_params:
            save_dictionary(fname=os.path.join(self.parameters['io_dict']['main_path'], 'parameters.data'),data=self.parameters)




