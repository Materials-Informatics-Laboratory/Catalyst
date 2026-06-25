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
import copy
import glob
import gzip
import math
import sys
import os

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
        self.scalar = torch.amp.GradScaler(enabled=False)

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
        """
        Return the underlying nn.Module.

        Handles:
          - DistributedDataParallel / DataParallel wrappers: .module
          - torch.compile OptimizedModule wrappers: ._orig_mod
        """
        model = self.model

        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        return model



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
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

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

        safe_parameters = copy.deepcopy(parameters)

        if "model_dict" in safe_parameters:
            safe_parameters["model_dict"].pop("model", None)
            safe_parameters["model_dict"].pop("optimizer", None)
            safe_parameters["model_dict"].pop("scheduler", None)
            safe_parameters["model_dict"].pop("loss_fn", None)

        checkpoint = {
            "epoch": epoch,
            "model_state": core.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "parameters": safe_parameters,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
        }

        self.checkpoint = checkpoint
        torch.save(checkpoint, fname)

    def load_checkpoint(
            self,
            fname=None,
            map_location=None,
            load_optimizer=True,
            strict=True,
    ):
        """
        Load model/optimizer/scaler state into this GNN.

        Handles:
        - CUDA checkpoint -> CPU load
        - CPU checkpoint -> CUDA load
        - DDP/DataParallel prefixes: module.*
        - torch.compile prefixes: _orig_mod.*
        - nested prefixes: module._orig_mod.*, _orig_mod.module.*, etc.
        - checkpoints saved as {"model_state": ...}
        - checkpoints saved as {"state_dict": ...}
        - raw state_dict checkpoints
        - scaler_state / scalar_state typo compatibility
        - lazy GenericFeatureEncoder MLP initialization from checkpoint shapes
        """

        import warnings

        if map_location is None:
            map_location = self.device if hasattr(self, "device") else "cpu"

        # ------------------------------------------------------------------
        # Load checkpoint object
        # ------------------------------------------------------------------
        if fname is None:
            checkpoint = self.checkpoint
        else:
            try:
                checkpoint = torch.load(
                    fname,
                    map_location=map_location,
                    weights_only=False,
                )
            except TypeError:
                # Older PyTorch versions do not support weights_only.
                checkpoint = torch.load(fname, map_location=map_location)

        if checkpoint is None:
            raise ValueError(
                "No checkpoint was provided. Pass fname=... or set self.checkpoint first."
            )

        core = self._core_model()

        # ------------------------------------------------------------------
        # Extract model state safely
        # ------------------------------------------------------------------
        def _looks_like_state_dict(obj):
            if not isinstance(obj, dict):
                return False
            if not obj:
                return False
            return all(isinstance(k, str) for k in obj.keys()) and any(
                torch.is_tensor(v) for v in obj.values()
            )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]

        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]

        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            maybe_model = checkpoint["model"]
            if hasattr(maybe_model, "state_dict"):
                model_state = maybe_model.state_dict()
            else:
                raise TypeError(
                    "Checkpoint has key 'model', but checkpoint['model'] does not "
                    "appear to be a torch.nn.Module."
                )

        elif _looks_like_state_dict(checkpoint):
            model_state = checkpoint

        else:
            if isinstance(checkpoint, dict):
                raise KeyError(
                    "Could not find model weights in checkpoint. Expected one of: "
                    "'model_state', 'state_dict', 'model', or a raw state_dict. "
                    f"Available keys: {list(checkpoint.keys())}"
                )
            raise TypeError(
                "Checkpoint must be a dict/state_dict. "
                "It looks like the full model object may have been saved directly."
            )

        if not isinstance(model_state, dict):
            raise TypeError(
                f"Extracted model_state must be a dict/state_dict, got {type(model_state)}."
            )

        # ------------------------------------------------------------------
        # Normalize checkpoint prefixes
        # ------------------------------------------------------------------
        def _strip_known_prefixes_from_key(key):
            """
            Convert:
                module.encoder...
                _orig_mod.encoder...
                module._orig_mod.encoder...
                _orig_mod.module.encoder...
            to:
                encoder...
            """
            new_key = key

            changed = True
            while changed:
                changed = False

                if new_key.startswith("module."):
                    new_key = new_key[len("module."):]
                    changed = True

                if new_key.startswith("_orig_mod."):
                    new_key = new_key[len("_orig_mod."):]
                    changed = True

            return new_key

        def _strip_known_prefixes_from_state_dict(state_dict):
            cleaned = {}

            for key, value in state_dict.items():
                cleaned[_strip_known_prefixes_from_key(key)] = value

            return cleaned

        model_state = _strip_known_prefixes_from_state_dict(model_state)

        # ------------------------------------------------------------------
        # Initialize lazy encoder modules if needed
        # ------------------------------------------------------------------
        def _get_model_device(module):
            try:
                return next(module.parameters()).device
            except StopIteration:
                if hasattr(self, "device"):
                    return torch.device(self.device)
                return torch.device("cpu")

        def _infer_mlp_input_dim_from_state(state_dict, prefix):
            """
            For modules like:
                encoder.embed_g_node = Sequential(MLP(...), LayerNorm(...))

            the first linear weight usually appears as:
                encoder.embed_g_node.0.mlp.0.weight

            with shape:
                [hidden_dim, input_dim]
            """
            candidate_keys = [
                f"{prefix}.0.mlp.0.weight",
                f"{prefix}.mlp.0.weight",
                f"{prefix}.0.weight",
            ]

            for key in candidate_keys:
                if key in state_dict and torch.is_tensor(state_dict[key]):
                    weight = state_dict[key]
                    if weight.dim() >= 2:
                        return int(weight.shape[1])

            return None

        def _maybe_initialize_lazy_generic_encoder(module, state_dict):
            """
            GenericFeatureEncoder initializes embed_g_node/embed_a_node/embed_a_edge
            lazily. During standalone inference, load_checkpoint can run before the
            first forward pass, so those modules may not exist yet.

            If the encoder exposes _ensure_mlp(...), initialize missing MLPs from
            checkpoint weight shapes.
            """
            encoder = getattr(module, "encoder", None)
            if encoder is None:
                return

            if not hasattr(encoder, "_ensure_mlp"):
                return

            device = _get_model_device(module)

            prefix_to_attr = {
                "encoder.embed_g_node": "embed_g_node",
                "encoder.embed_a_node": "embed_a_node",
                "encoder.embed_a_edge": "embed_a_edge",
            }

            for prefix, attr_name in prefix_to_attr.items():
                if getattr(encoder, attr_name, None) is not None:
                    continue

                in_dim = _infer_mlp_input_dim_from_state(state_dict, prefix)
                if in_dim is None:
                    continue

                encoder._ensure_mlp(attr_name, in_dim, device=device)

        _maybe_initialize_lazy_generic_encoder(core, model_state)

        # ------------------------------------------------------------------
        # Load model state
        # ------------------------------------------------------------------
        core_state = core.state_dict()

        if strict:
            try:
                core.load_state_dict(model_state, strict=True)
            except RuntimeError as exc:
                model_keys = list(core_state.keys())
                ckpt_keys = list(model_state.keys())

                raise RuntimeError(
                    "Failed to load checkpoint with strict=True after normalizing "
                    "known wrapper prefixes. This usually means either:\n"
                    "  1. the inference model architecture differs from the training model,\n"
                    "  2. lazy modules were not initialized correctly,\n"
                    "  3. decoder/encoder names changed between training and inference, or\n"
                    "  4. the checkpoint is from a different model.\n\n"
                    f"First model keys: {model_keys[:8]}\n"
                    f"First checkpoint keys: {ckpt_keys[:8]}\n"
                ) from exc

        else:
            # Partial/non-strict load: only load keys that exist and match shape.
            filtered_state = {}
            skipped = []

            for key, value in model_state.items():
                if key not in core_state:
                    skipped.append((key, "missing_in_model"))
                    continue

                if core_state[key].shape != value.shape:
                    skipped.append(
                        (key, f"shape_mismatch checkpoint={tuple(value.shape)} model={tuple(core_state[key].shape)}")
                    )
                    continue

                filtered_state[key] = value

            missing, unexpected = core.load_state_dict(filtered_state, strict=False)

            if skipped:
                warnings.warn(
                    "Checkpoint loaded with strict=False. "
                    f"Loaded {len(filtered_state)} tensors, skipped {len(skipped)} tensors. "
                    f"First skipped entries: {skipped[:8]}"
                )

            if missing:
                warnings.warn(f"Missing model keys after partial load: {missing[:8]}")

            if unexpected:
                warnings.warn(f"Unexpected checkpoint keys after partial load: {unexpected[:8]}")

        # ------------------------------------------------------------------
        # Optimizer/scaler are usually not needed for inference
        # ------------------------------------------------------------------
        if (
                load_optimizer
                and self.optimizer is not None
                and isinstance(checkpoint, dict)
                and checkpoint.get("optimizer_state") is not None
        ):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as exc:
                warnings.warn(
                    "Model weights were loaded, but optimizer_state could not be loaded. "
                    "This is usually fine for inference. "
                    f"Optimizer load error: {exc}"
                )

        # Support both names because older code used scalar/scaler inconsistently.
        scaler_obj = None
        if hasattr(self, "scaler") and self.scaler is not None:
            scaler_obj = self.scaler
        elif hasattr(self, "scalar") and self.scalar is not None:
            scaler_obj = self.scalar

        scaler_state = None
        if isinstance(checkpoint, dict):
            scaler_state = checkpoint.get("scaler_state", None)
            if scaler_state is None:
                scaler_state = checkpoint.get("scalar_state", None)

        if load_optimizer and scaler_obj is not None and scaler_state is not None:
            try:
                scaler_obj.load_state_dict(scaler_state)
            except Exception as exc:
                warnings.warn(
                    "Model weights were loaded, but scaler_state/scalar_state could not be loaded. "
                    "This is usually fine for inference. "
                    f"Scaler load error: {exc}"
                )

        # Keep a reference to the loaded checkpoint.
        if isinstance(checkpoint, dict):
            self.checkpoint = checkpoint
            return checkpoint.get("epoch", None)

        return None


    def load_data(self,params,format=0, rank=0,load_training=True,samples_file=None):
        if format != 2 and format != -1:
            graph_files = glob.glob(os.path.join(params['io_dict']['data_dir'], '*'))
            samples = load_dictionary(samples_file)
            if load_training:
                self.training_samples = samples['training']
            self.validation_samples = samples.get('validation') or samples.get('gids')

            if format == 0:
                gids = [PurePath(graph).parts[-1].split('.')[0] for graph in graph_files]
                if len(gids) == 0:
                    print('Error: no graph files found...')
                    exit(0)
            else:
                gids = [torch.load(gname)['gid'] for gname in graph_files]

            if load_training:
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
            gids = [graph.gid for graph in graph_file['graphs']]
            if load_training:
                self.training_samples = samples['training']
                a, b, c = np.intersect1d(self.training_samples, gids, return_indices=True)
                self.training_graphs = [graph_file['graphs'][cc] for cc in c]
            self.validation_samples = samples['validation']
            a, b, c = np.intersect1d(self.validation_samples, gids, return_indices=True)
            self.validation_graphs = [graph_file['graphs'][cc] for cc in c]
        else:
            graph_data = load_dictionary(os.path.join(params['io_dict']['data_dir'], 'graphs.data'))
            if load_training:
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

        loss = 0.0

        for data in self.training_loader:
            # move batch to device
            data = data.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # forward + loss under autocast
            with torch.amp.autocast(enabled=use_amp,device_type=self.device):
                pred = self.model(data)
                preds, y, vec = accumulate_predictions(pred, data, loss_accum)
                preds = preds.to(y.device)

                batch_loss = loss_fn(preds, y, vec, data, loss_params)

            loss += batch_loss.item()

            # backward with mixed precision
            self.scalar.scale(batch_loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

        # DDP: average loss across ranks
        if parameters['device_dict']['run_ddp']:
            loss = reduce_tensor(torch.tensor(loss, device=self.device)).item()

        return loss / (len(self.training_loader) * parameters['device_dict']['world_size'])

    @torch.no_grad()
    def validate(self,parameters,rank=0):
        self.model.eval()
        loss_fn = loss_setup(params=parameters['model_dict']['loss_params'])
        loss = 0.0
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
            loss += batch_loss.item()

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
            loss = reduce_tensor(torch.tensor(loss).to(parameters['device_dict']['device'])).item()

        return loss / (len(self.validation_loader) * parameters['device_dict']['world_size'])

    @torch.no_grad()
    def predict(self,parameters,rank=0):
        self.model.eval()
        values = []
        gids = []

        for data in self.validation_loader:
            data = data.to(self.device, non_blocking=parameters['device_dict']['pin_memory'])
            pred = self.model(data)
            preds, vec = accumulate_predictions(pred, data, loss_tag='exact',return_y=False)
            preds = preds.to(self.device)

            values.append(preds.tolist())
            gids.append(data.gid)

        test_info = {
            'gids': gids,
            'pred': values,
            'vec': vec
        }
        if parameters['device_dict']['run_ddp']:
            test_info = combine_dicts_across_gpus(test_info)
        return test_info


