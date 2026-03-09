from ..utils.loss import loss_setup
from ..utils.distributed import reduce_tensor, combine_dicts_across_gpus
from ..utils.memory import optimizer_to
from .modules.utils.predict import accumulate_predictions
from ...data.utils import load_dictionary, save_dictionary
from .modules.utils.data_manager import setup_dataloader
from ..utils.optimizer import set_optimizer

import torch
import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.checkpoint = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def print_debug(self):
        print(self.model)
        print(self.training_graphs)
        print(self.validation_graphs)

    def compile_model(self):
        torch._dynamo.config.suppress_errors = True
        self.model = torch.compile(
            self.model,
            mode="default",  # or "max-autotune"
            backend="eager",
            dynamic=False  # True if your input shapes vary
        )
    def send_model(self,device):
        self.model.to(device)
        self.device = device

    def is_ddp(self):
        return isinstance(self.model, DDP)

    def _core_model(self):
        """Return underlying nn.Module (unwrap DDP if needed)."""
        if self.is_ddp():
            return self.model.module
        return self.model

    def set_model_state(self,model_state):
        self.model_state = model_state

    def set_optimizer_state(self,optimizer_state):
        self.optimizer_state = optimizer_state

    def set_optimizer_(self, parameters):
        parameters['model_dict']['optimizer_params']['params_group']['params'] = self.model.parameters()
        self.optimizer = set_optimizer(parameters)
        for param in self.optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

        use_amp = parameters['device_dict'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def save_checkpoint(self, parameters, epoch, rank=0, fname=None):
        """Save model/optimizer state (rank 0 only)."""
        if rank != 0:
            return

        if fname is None:
            fname = os.path.join(
                parameters['io_dict']['model_dir'],
                f"checkpoint_epoch_{epoch}.pt"
            )

        core = self._core_model()
        checkpoint = {
            "epoch": epoch,
            "model_state": core.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "parameters": parameters,  # optional
            "scaler_state": scaler.state_dict(),
        }
        self.checkpoint = checkpoint
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname=None, map_location=None):
        """
        Load model/optimizer state into this GNN.
        Call after self.model has been constructed (and wrapped in DDP if used).
        """
        if map_location is None:
            map_location = self.device if hasattr(self, "device") else "cpu"

        if fname is None:
            checkpoint = self.checkpoint
        else:
            checkpoint = torch.load(fname, map_location=map_location)
        core = self._core_model()
        core.load_state_dict(checkpoint["model_state"])

        if self.optimizer is not None and checkpoint.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scalar is not None and checkpoint.get("scalar_state") is not None:
            self.scalar.load_state_dict(checkpoint["scalar_state"])

        return checkpoint.get("epoch", None)

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

    def train(self, training_dict):
        """
        Modular training loop with optional AMP.

        training_dict:
          'params'   : full parameter dict
          'loss_fn'  : optional custom loss function with signature
                       loss_fn(preds, y, vec, data, loss_params) -> scalar loss tensor
        """
        self.model.train()
        parameters = training_dict['params']
        loss_accum = parameters['model_dict']['accumulate_loss']
        loss_params = parameters['model_dict']['loss_params']
        use_amp = parameters['device_dict'].get('use_amp', False)

        # Choose loss function: custom if provided, else your default
        if 'loss_fn' in training_dict and training_dict['loss_fn'] is not None:
            loss_fn = training_dict['loss_fn']
            # expected signature: loss_fn(preds, y, vec, data, loss_params)
        else:
            base_loss = loss_setup(params=loss_params)

            # wrap to match the modular signature
            def loss_fn(preds, y, vec, data, loss_params=loss_params):
                if vec:
                    loss_list = [base_loss(preds[i], y[i]) for i in range(len(preds))]
                    return torch.sum(torch.stack(loss_list))
                else:
                    return base_loss(preds, y)

        epoch_loss = 0.0

        for data in self.training_loader:
            # move batch to device
            data = data.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # forward + loss under autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = self.model(data)
                preds, y, vec = accumulate_predictions(pred, data, loss_accum)
                preds = preds.to(y.device)

                batch_loss = loss_fn(preds, y, vec, data, loss_params)

            epoch_loss += batch_loss.item()

            # backward with mixed precision
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # DDP: average loss across ranks
        if parameters['device_dict']['run_ddp']:
            epoch_loss = reduce_tensor(torch.tensor(epoch_loss, device=self.device)).item()

        return epoch_loss / (len(self.training_loader) * parameters['device_dict']['world_size'])

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