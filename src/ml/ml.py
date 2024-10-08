from torch import nn
import numpy as np
import torch
import os
import gc

class ML():
    def __init__(self):
        super().__init__()

        self.parameters = dict(
                               device_dict = dict(
                                   world_size=1,
                                   device='',
                                   ddp_backend='',
                                   run_ddp=False,
                                   pin_memory=False,
                               ),
                               io_dict = dict(
                                   main_path='',
                                   restart_model_name='',
                                   graph_data_dir='',
                                   model_dir='',
                                   model_save_dir='',
                                   results_dir='',
                                   pretrain_dir='',
                                   samples_dir='',
                                   projection_dir='',
                                   remove_old_model=False,
                                   write_indv_pred=False,
                               ),
                               sampling_dict = dict(test_sampling_type = '',
                                                    pretraining_sampling_type = '',
                                                    sampling_type = '',
                                                    train_split = 0.8,
                                                    clusters=1,
                                                    sampling_seed=112358,
                                ),
                                loader_dict=dict(
                                    shuffle_loader=False,
                                    batch_size=[1,1],
                                    num_workers=0
                                ),
                               sodas_dict = dict(
                                    run_sodas=True,
                                    sodas_model = None
                                ),
                               model_dict = dict(
                                   n_models=1,
                                   num_epochs=[1, 1],
                                   train_tolerance=1.0,
                                   model = None,
                                   accumulate_loss='',
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

    def set_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.model = self.parameters['model_dict']['model']

    def set_params(self,new_params,save_params=True):
        if not 'sodas_dict' in new_params:
            print('WARNING: No SODAS dictionary set...')
        if 'sodas_dict' in new_params:
            if not 'sodas_model' in new_params['sodas_dict']:
                print('No sodas model in sodas dictionary...killing job...')
                exit(0)
            if not 'run_sodas' in new_params['sodas_dict']:
                print('WARNING: run_sodas not set in sodas dictionary...setting to True')
                new_params['run_sodas']['gen_encodings'] = True
            if 'run_ddp' in new_params['device_dict']:
                print('run_ddp cannot be used during SODAS sampling...setting to false')
                new_params['device_dict']['run_ddp'] = False
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
            if not 'train_tolerance' in new_params['model_dict']:
                print('WARNGING: training tolerance not set...setting to 1e-3')
                new_params['model_dict']['train_tolerance'] = 1e-3
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

        if not 'graph_data_dir' in new_params['io_dict']:
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

        self.parameters = new_params

        if save_params:
            np.save(os.path.join(self.parameters['io_dict']['main_path'], 'parameter_log.npy'), self.parameters)




