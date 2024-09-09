from torch import nn
import numpy as np
import torch
import os

class ML():
    def __init__(self):
        super().__init__()

        self.parameters = dict(gnn_dim = 10,
                               num_convs = 5,
                               num_inputs = 1,
                               num_epochs = 1000,
                               BATCH_SIZE = 1,
                               n_models = 1,
                               world_size = 1,
                               sampling_seed=112358,
                               graph_cutoff = 5.0,
                               LEARN_RATE = 2e-4,
                               lr_scale=1.0,
                               train_tolerance = 1e-5,
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
                               sodas_dict = dict(
                                    gen_graphs=False,
                                    gen_encodings=False,
                                    sodas_model = None,
                                    projection_dir=''
                                ),
                               model_dict = dict(
                                   model = None
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




