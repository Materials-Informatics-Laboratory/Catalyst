from ..utils.loss import loss_setup
from ..utils.distributed import reduce_tensor, combine_dicts_across_gpus
from ..utils.memory import optimizer_to
from .modules.utils.predict import accumulate_predictions
from ...data.utils import load_dictionary
from .modules.utils.data_manager import setup_dataloader
from ..utils.optimizer import set_optimizer

import torch


from datetime import datetime
from pathlib import PurePath
import numpy as np

import numpy as np


import pickle
import random
import torch
import glob
import gzip
import math
import sys
import os
import data

class GNN():
    def __init__(self, model,device):
        super().__init__()

        self.device = device
        self.model = model
        self.model_state = None
        self.model.to(device)
        self.optimizer = None
        self.optimizer_state = None
        self.training_loader = None
        self.validation_loader = None
        self.training_graphs = None
        self.training_samples = None
        self.validation_graphs = None
        self.validation_samples = None

    def print_debug(self):
        print(self.model)
        print(self.training_graphs)
        print(self.validation_graphs)

    def set_model_state(self,model_state):
        self.model_state = model_state

    def set_optimizer_state(self,optimizer_state):
        self.optimizer_state = optimizer_state

    def set_optimizer(self,parameters):
        self.optimizer = set_optimizer(parameters)
        optimizer_to(self.optimizer,self.device)

    def load_model_state(self,state):
        self.model.load_state_dict(self.model_state)
        self.model.to(self.device)

    def train(self,training_dict):
        self.model.train()
        parameters = training_dict['params']
        loss_accum = parameters['model_dict']['accumulate_loss']
        epoch_loss = 0.0

        loss_fn = loss_setup(params=parameters['model_dict']['loss_params'])
        for data in self.training_loader:
            def closure():
                data.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(data)
                preds, y, vec = accumulate_predictions(pred, data, loss_accum)
                preds = preds.to(y.device)
                if vec:
                    loss_list = [0.0] * len(preds)
                    for i in range(len(preds)):
                        loss_list[i] = loss_fn(preds[i], y[i])
                    batch_loss = torch.sum(torch.stack(loss_list))
                else:
                    batch_loss = loss_fn(preds, y)
                nonlocal epoch_loss

                epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()
                return batch_loss

            self.optimizer.step(closure)
        if parameters['device_dict']['run_ddp']:
            epoch_loss = reduce_tensor(torch.tensor(epoch_loss).to(parameters['device_dict']['device'])).item()

        return epoch_loss / (len(self.training_loader) * parameters['device_dict']['world_size'])

    def load_model(self,parameters, data_only=False):
        if data_only:
            return torch.load(parameters['io_dict']['loaded_model_name'])
        else:
            model_data = torch.load(parameters['io_dict']['loaded_model_name'])
            self.model.load_state_dict(model_data['model'])

    def setup_ddp(self, parameters,rank):
        ddp_model = DDP(self.model, device_ids=[rank],
                        find_unused_parameters=parameters['device_dict']['find_unused_parameters'])
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ddp_model)

    def load_training_data(self,params, samples_file, format=0, rank=0):
        if format != 2 and format != -1:
            graph_files = glob.glob(os.path.join(params['io_dict']['data_dir'], '*'))
            samples = load_dictionary(samples_file)
            self.training_samples = samples['training']
            self.validation_samples = samples['validation']

            if format == 0:
                gids = [PurePath(graph).parts[-1].split('.')[0] for graph in graph_files]
                if len(gids) == 0:
                    print('Error: no graph files found...')
                    exit(0)
            else:
                gids = [torch.load(gname)['gid'] for gname in graph_files]

            a, b, c = np.intersect1d(self.training_samples, gids, return_indices=True)
            selected_graphs = [graph_files[cc] for cc in c]
            if format == 0:
                self.training_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    self.training_graphs[i] = torch.load(selected_graphs[i])
            else:
                self.training_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    self.training_graphs[i] = selected_graphs[i]
            a, b, c = np.intersect1d(self.validation_samples, gids, return_indices=True)
            selected_graphs = [graph_files[cc] for cc in c]
            if format == 0:
                self.validation_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    self.validation_graphs[i] = torch.load(selected_graphs[i])
            else:
                self.validation_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    self.validation_graphs[i] = selected_graphs[i]
        elif format == 2:
            graph_file = load_dictionary(glob.glob(os.path.join(params['io_dict']['data_dir'], 'graphs.data'))[0])
            samples = load_dictionary(samples_file)
            self.training_samples = samples['training']
            self.validation_samples = samples['validation']
            gids = [graph.gid for graph in graph_file['graphs']]
            a, b, c = np.intersect1d(self.training_samples, gids, return_indices=True)
            self.training_graphs = [graph_file['graphs'][cc] for cc in c]
            a, b, c = np.intersect1d(self.validation_samples, gids, return_indices=True)
            self.validation_graphs = [graph_file['graphs'][cc] for cc in c]
        else:
            graph_data = load_dictionary(os.path.join(params['io_dict']['data_dir'], 'graphs.data'))
            self.training_graphs = [graph for graph in graph_data['training']]
            self.validation_graphs = [graph for graph in graph_data['validation']]

    def set_dataloader(self,cat,training=True,validation=True,epoch=-1):
        if training:
            loader_params = {
                'epoch':epoch,
                'shuffle':cat.parameters['loader_dict']['shuffle_loader'],
                'batch_size':cat.parameters['loader_dict']['batch_size'][0]
            }
            self.training_loader = setup_dataloader(data=self.training_graphs,cat=cat,loader_params=loader_params)
        if validation:
            loader_params = {
                'epoch': epoch,
                'shuffle': cat.parameters['loader_dict']['shuffle_loader'],
                'batch_size': cat.parameters['loader_dict']['batch_size'][1]
            }
            self.validation_loader = setup_dataloader(data=self.validation_graphs, cat=cat, loader_params=loader_params)

    @torch.no_grad()
    def validate(self,parameters,rank=0):
        self.model.eval()
        loss_fn = loss_setup(params=parameters['model_dict']['loss_params'])
        epoch_loss = 0.0
        loss_accum = parameters['model_dict']['accumulate_loss']
        values = [[],[],[]]
        gids = []
        for data in self.validation_loader:
            data = data.to(self.device, non_blocking=parameters['device_dict']['pin_memory'])
            pred = self.model(data)
            preds, y, vec = accumulate_predictions(pred, data, loss_accum)
            preds = preds.to(y.device)
            if vec:
                loss_list = [0.0] * len(preds)
                for i in range(len(preds)):
                    loss_list[i] = loss_fn(preds[i], y[i])
            else:
                loss_list = [loss_fn(preds, y)]
            batch_loss = torch.sum(torch.stack(loss_list))
            epoch_loss += batch_loss.item()

            values[0].append(preds.tolist())
            values[1].append(y.tolist())
            values[2].append(loss_list)
            gids.append(data.gid)

        test_info = {
            'gids': gids,
            'pred': values[0],
            'y': values[1],
            'loss': values[2],
            'loss_fn': parameters['model_dict']['accumulate_loss'],
            'vec': vec
        }

        if parameters['device_dict']['run_ddp']:
            test_info = combine_dicts_across_gpus(test_info)
        if parameters['io_dict']['write_indv_pred']:
            if rank == 0:
                save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'],'indv_pred.data'),
                                data=test_info)
        if parameters['device_dict']['run_ddp']:
            epoch_loss = reduce_tensor(torch.tensor(epoch_loss).to(parameters['device_dict']['device'])).item()

        return epoch_loss / (len(self.validation_loader) * parameters['device_dict']['world_size'])