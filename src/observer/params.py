from src.ml.utils.distributed import set_spawn_method
from src.io.io import save_dictionary, get_system_info
from torch import nn
import numpy as np
import torch
import os
import gc

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
                               sampling_dict = dict(sampling_types=['random','random','random'],
                                                    split=[0.5,0.5,0.5],
                                                    sampling_seed=112358,
                                                    params_groups = [{
                                                        'clusters':1,
                                                    },{
                                                        'clusters':1,
                                                    },{
                                                        'clusters':1,
                                                    }]
                                ),
                                loader_dict=dict(
                                    shuffle_loader=False,
                                    batch_size=[1,1,1],
                                    shuffle_steps=10,
                                    num_workers=0
                                ),
                               characterization_dict = dict(
                                    model = None
                                ),
                               model_dict = dict(
                                   n_models=1,
                                   num_epochs=[1, 1],
                                   train_delta = [1.0,1.0],
                                   train_tolerance =[1.0,1.0],
                                   max_deltas=4,
                                   accumulate_loss=['sum', 'sum', 'sum'],
                                   loss_func=torch.nn.MSELoss(),
                                   model = None,
                                   model_params_group=dict(
                                        encoder=dict(

                                        ),
                                        processor=dict(

                                        ),
                                        decoder=dict(

                                        )
                                   ),
                                   interpretable=False,
                                   pre_training=False,
                                   restart_training=False,
                                   optimizer_params=dict(
                                       lr_scale=[1.0, 0.1],
                                       dynamic_lr=False,
                                       dist_type='',
                                       optimizer='',
                                       params_group={
                                           'lr': 0.001
                                       }
                                   )
                               )
                            )

        self.model = None
        self.accumulate_loss_options = ['exact','sum']
        self.device_options = ['cuda','cpu']
        self.optimizer_options = ['AdamW','Adadelta','Adagrad','Adam','SparseAdam','Adamax','ASGD',
                                  'LBFGS','NAdam','RAdam','RMSprop','Rprop','SGD']

        '''
        graph clustering params
        {
            'leaf_size':2,
            'neighbors':10,
            'metric':'minkowski'
        }
        '''

    def set_model(self):
        del self.model
        gc.collect()
        if self.parameters['device_dict']['device'] == 'cuda':
            torch.cuda.empty_cache()
        self.model = None
        self.model = self.parameters['model_dict']['model']

    def set_params(self,new_params,save_params=True):
        if not 'characterization_dict' in new_params:
            print('WARNING: No characterization dictionary set...')
        if 'characterization_dict' in new_params:
            if not 'model' in new_params['characterization_dict']:
                print('Warning: no model in characterization dictionary...')
        else:
            if not 'model_dict' in new_params:
                print('No model dictionary set...killing run...')
                exit(0)
            if not 'optimizer_params' in new_params['model_dict']:
                print('No optimizer dictionary set inside of model dictionary...killing run...')
                exit(0)
            checks = [0, 0, 0]
            if 'accumulate_loss' in new_params['model_dict']:
                for i,curr_option in enumerate(new_params['model_dict']['accumulate_loss']):
                    for option in self.accumulate_loss_options:
                        if option == curr_option:
                            checks[i] = 1
                            break
            if sum(checks) < 3:
                for i, curr_option in enumerate(new_params['model_dict']['accumulate_loss']):
                    if checks[i] == 0:
                        print('WARNING: Loss accumulation at element ',i,' has bad option...setting to exact')
                    new_params['model_dict']['accumulate_loss'][i] = 'exact'
            if 'optimizer' in new_params['model_dict']['optimizer_params']:
                for option in self.optimizer_options:
                    if option == new_params['model_dict']['optimizer_params']['optimizer']:
                        check = True
                        break
            if check == False:
                print('Optimizer not set...please choose from these available options: ', self.optimizer_options)
                print('Killing job...')
                exit(0)
            if not 'loss_func' in new_params['model_dict']:
                print('No loss function set...killing run...')
                exit(0)
            if not 'model' in new_params['model_dict']:
                print('No model set...killing run...')
                exit(0)
            if not 'num_epochs' in new_params['model_dict']:
                print('No epochs set...killing run...')
                exit(0)
            else:
                if len(new_params['model_dict']['num_epochs']) != 2:
                    print('Num_epochs tag does not have 2 values...killing job...')
                    exit(0)
            if not 'pre_training' in new_params['model_dict']:
                print('WARNING: Pretraining not set...setting to False...')
                new_params['model_dict']['pre_training'] = False
            if not 'train_tolerance' in new_params['model_dict']:
                print('WARNGING: training tolerance not set...setting to 1e-3')
                new_params['model_dict']['train_tolerance'] = [1e-3,1e-3]
            else:
                if not isinstance(new_params['model_dict']['train_delta'], list):
                    print('WARNING: train_delta is not a list...fixing this for you...')
                    if new_params['model_dict']['pre_training']:
                        print(
                            'Pretraining set to True, interpreting your train_delta as pretraining delta...setting value for both training and pretraining')
                        new_params['model_dict']['train_delta'] = [new_params['model_dict']['train_delta'],
                                                                    new_params['model_dict']['train_delta']]
                    else:
                        print(
                            'Pretraining set to False, interpreting your train_tolerance as training delta...setting value for training only')
                        new_params['model_dict']['train_delta'] = [-1.0, new_params['model_dict']['train_delta']]
                else:
                    if len(new_params['model_dict']['train_delta']) != 2:
                        if new_params['model_dict']['pre_training']:
                            print(
                                'Pretraining set to True, interpreting your train_tolerance as pretraining tolerance...setting value for both training and pretraining')
                            new_params['model_dict']['train_delta'] = [new_params['model_dict']['train_delta'],
                                                                        new_params['model_dict']['train_delta']]
                        else:
                            print(
                                'Pretraining set to False, interpreting your train_tolerance as training tolerance...setting value for training only')
                            new_params['model_dict']['train_delta'] = [-1.0, new_params['model_dict']['train_delta']]
            if not 'train_delta' in new_params['model_dict']:
                print('WARNGING: training delta not set...setting to 1e-3')
                new_params['model_dict']['train_delta'] = [1e-3,1e-3]
            else:
                if not isinstance(new_params['model_dict']['train_tolerance'], list):
                    print('WARNING: train_tolerance is not a list...fixing this for you...')
                    if new_params['model_dict']['pre_training']:
                        print('Pretraining set to True, interpreting your train_tolerance as pretraining tolerance...setting value for both training and pretraining')
                        new_params['model_dict']['train_tolerance'] = [new_params['model_dict']['train_tolerance'],new_params['model_dict']['train_tolerance']]
                    else:
                        print('Pretraining set to False, interpreting your train_tolerance as training tolerance...setting value for training only')
                        new_params['model_dict']['train_tolerance'] = [-1.0,new_params['model_dict']['train_tolerance']]
                else:
                    if len(new_params['model_dict']['train_tolerance']) != 2:
                        if new_params['model_dict']['pre_training']:
                            print(
                                'Pretraining set to True, interpreting your train_tolerance as pretraining tolerance...setting value for both training and pretraining')
                            new_params['model_dict']['train_tolerance'] = [new_params['model_dict']['train_tolerance'],
                                                                        new_params['model_dict']['train_tolerance']]
                        else:
                            print(
                                'Pretraining set to False, interpreting your train_tolerance as training tolerance...setting value for training only')
                            new_params['model_dict']['train_tolerance'] = [-1.0, new_params['model_dict']['train_tolerance']]
            if not 'max_deltas' in new_params['model_dict']:
                print('WARNGING: max_deltas not set...setting to 3')
                new_params['model_dict']['max_deltas'] = 3
            if not 'shuffle_steps' in new_params['loader_dict']:
                print('WARNGING: shuffle_Steps not set...setting to 10')
                new_params['loader_dict']['shuffle_steps'] = 10
            if not 'run_ddp' in new_params['device_dict']:
                print('run_ddp not set...setting to false')
                new_params['device_dict']['run_ddp'] = False
            if new_params['device_dict']['run_ddp'] == True:
                if not 'ddp_backend' in new_params['device_dict']:
                    print('ddp_backend not set while using DDP...killing job...')
                    exit(0)
                if not 'world_size' in new_params['device_dict']:
                    print(
                        'world_size not set while using DDP...attempting to grab current world_size based on GPU count...')
                    new_params['device_dict']['world_size'] = torch.cuda.device_count()
            if new_params['device_dict']['run_ddp'] == False:
                new_params['device_dict']['world_size'] = 1
            if not 'dynamic_lr' in new_params['model_dict']['optimizer_params']:
                print('WARNING: Dynamic learning rates not set...detting to false')
                new_params['model_dict']['optimizer_params']['dynamic_lr'] = False
            if new_params['model_dict']['optimizer_params']['dynamic_lr'] == True:
                if not 'dist_type' in new_params['model_dict']['optimizer_params']:
                    print('No dict_type chosen while using dynamic learning rates...killing job...')
                    exit(0)
                if not 'params_group' in new_params['model_dict']['optimizer_params']:
                    print('No params_group set while using dynamic learning rates...killing job...')
                    exit(0)
                if not 'lr_scale' in new_params['model_dict']['optimizer_params']:
                    print('No lr_scale set while using dynamic learning rates...killing job...')
                    exit(0)
            if not 'shuffle_loader' in new_params['loader_dict']:
                print('WARNING: shuffle_loader not set...setting to false')
                new_params['loader_dict']['shuffle_loader'] = False
            if not 'batch_size' in new_params['loader_dict']:
                print('batch_size not set in loader dictionary...killing job...')
                exit(0)
            else:
                if len(new_params['loader_dict']['batch_size']) != 3:
                    print('batch_size length does not equal 3...killing job...')
                    exit(0)
        check = False
        if 'device' in new_params['device_dict']:
            for option in self.device_options:
                if option == new_params['device_dict']['device']:
                    check = True
                    break
        if check == False:
            print('WARNING: Device not set...setting to cpu')
            new_params['device_dict']['device'] = 'cpu'
        check = False
        if not 'n_models' in new_params['model_dict']:
            print('WARNGING: n_models not set...setting to 1')
            new_params['model_dict']['num_models'] = 1
        if not 'pin_memory' in new_params['device_dict']:
            print('pin_memory not set...setting to false')
            new_params['device_dict']['pin_memory'] = False
        if not 'num_workers' in new_params['loader_dict']:
            print('WARNING: num_workers not set...setting to 0')
            new_params['loader_dict']['num_workers'] = 0

        if not 'data_dir' in new_params['io_dict']:
            print('Data directory not set...killing run...')
            exit(0)
        if not 'main_path' in new_params['io_dict']:
            print('Main path not set...killing run...')
            exit(0)
        if not 'io_dict' in new_params:
            print('No IO dictionary set...killing run...')
            exit(0)
        if not 'device_dict' in new_params:
            print('No device dictionary set...killing run...')
            exit(0)
        if not 'loader_dict' in new_params:
            print('No loader dictionary set...killing run...')
            exit(0)
        set_spawn_method(new_params)
        new_params['device_dict']['system_info'] = get_system_info()
        self.parameters = new_params

        if save_params:
            save_dictionary(fname=os.path.join(self.parameters['io_dict']['main_path'], 'parameters.data'),data=self.parameters)




