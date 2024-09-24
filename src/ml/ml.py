from torch import nn
import numpy as np
import torch
import os

class ML():
    def __init__(self):
        super().__init__()

        self.parameters = dict(
                               world_size = 1,
                               sampling_seed=112358,
                               is_dihedral = False,
                               remove_old_model = False,
                               interpretable = False,
                               pre_training = False,
                               run_pretrain = False,
                               write_indv_pred = False,
                               restart_training = False,
                               run_sodas_projection = False,
                               sodas_projection = False,
                               run_ddp = False,
                               pin_memory=False,
                               dynamic_lr = False,
                               ddp_backend='',
                               main_path = '',
                               restart_model_name = '',
                               device = '',
                               graph_data_dir = '',
                               model_dir='',
                               model_save_dir='',
                               results_dir = '',
                               pretrain_dir='',
                               samples_dir = '',
                               elements=[],
                               sampling_dict = dict(test_sampling_type = '',
                                                    pretraining_sampling_type = '',
                                                    sampling_type = '',
                                                    train_split = 0.8,
                                                    clusters=1
                                ),
                                loader_dict=dict(
                                    shuffle_loader=False,
                                    batch_size=[1,1],
                                    num_workers=0
                                ),
                               sodas_dict = dict(
                                    gen_graphs=False,
                                    gen_encodings=False,
                                    sodas_model = None,
                                    projection_dir=''
                                ),
                               model_dict = dict(
                                   n_models=1,
                                   num_epochs=[1, 1],
                                   train_tolerance=1.0,
                                   model = None,
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

    def set_model(self):
        del self.model
        self.model = None
        self.model = self.parameters['model_dict']['model']

    def set_params(self,new_params,save_params=True):
        self.parameters = new_params
        if save_params:
            np.save(os.path.join(self.parameters['main_path'], 'parameter_log.npy'), self.parameters)




