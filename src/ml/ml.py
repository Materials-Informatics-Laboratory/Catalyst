from ..graphite.nn.models.alignn import Encoder, Processor, Decoder, ALIGNN
from ..sodas.model.sodas import SODAS
from ..graphite.nn import MLP
from umap import umap_
from torch import nn
import numpy as np
import torch
import os

class PositiveScalarsDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform_atm = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())
        self.transform_bnd = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())
        self.transform_ang = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())

    # Here I assume `data` is not batched.
    def forward(self, data):
        atm_scalars = self.transform_atm(data.h_atm)
        bnd_scalars = self.transform_bnd(data.h_bnd)
        ang_scalars = self.transform_bnd(data.h_ang)
        return (atm_scalars, bnd_scalars, ang_scalars)

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
                               train_tolerance = 1e-5,
                               is_dihedral = False,
                               remove_old_model = True,
                               interpretable = True,
                               pre_training = False,
                               run_pretrain = False,
                               write_indv_pred = False,
                               restart_training = False,
                               run_sodas_projection = False,
                               sodas_projection = True,
                               run_ddp = False,
                               main_path = '',
                               restart_model_name = '',
                               device = 'cpu',
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
                                )
                            )

        self.model = None

    def set_model(self):
        self.model = ALIGNN(
            encoder=Encoder(num_species=self.parameters['num_inputs'],cutoff=self.parameters['graph_cutoff'],
                            dim=self.parameters['gnn_dim']),
            processor=Processor(num_convs=self.parameters['num_convs'], dim=self.parameters['gnn_dim']),
            decoder=PositiveScalarsDecoder(dim=self.parameters['gnn_dim']),
        )

    def set_params(self,new_params):
        self.parameters = new_params
        self.initialize()

    def initialize(self):
        np.save(os.path.join(self.parameters['main_path'], 'parameter_log.npy'),self.parameters)



