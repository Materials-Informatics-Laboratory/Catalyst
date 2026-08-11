from ..utils.loss import loss_setup
from ..utils.distributed import reduce_tensor, combine_dicts_across_gpus
from ..utils.memory import optimizer_to
from .modules.utils.predict import accumulate_predictions
from ...data.utils import load_dictionary, save_dictionary, safe_torch_load
from .modules.utils.data_manager import setup_dataloader
from ..utils.optimizer import set_optimizer

import torch
import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP

from contextlib import nullcontext
from pathlib import PurePath
import numpy as np

import copy
import glob
import os
import re
import warnings


class GNN:
    """
    High-level Catalyst GNN training/testing/checkpointing wrapper.

    This class is intentionally model-agnostic.  It can train/test:

        legacy ALIGNN/order models
        GNNBuilder models
        EquivariantGNN convenience models
        scalar, vector, and scalar_gradient decoder outputs

    The model itself should be a torch.nn.Module.  For the new builder stack,
    typical usage is:

        model = build_model(
            model_type="equivariant",
            output_type="vector",
            output_level="node",
            return_dict=False,
        )

        gnn = GNN(model, device)

    or:

        model = build_model(
            preset="equivariant_scalar_gradient",
            output_type="scalar_gradient",
            output_level="graph",
            return_dict=True,
        )

        gnn = GNN(model, device)

    Required companion update
    -------------------------
    This class expects accumulate_predictions(...) to support dict/tensor
    outputs from the new equivariant decoders.  Use the updated
    modules/utils/predict.py that supports:
        {"scalar": ...}
        {"vector": ...}
        {"gradient": ...}
        direct tensor predictions
        loss_tag="node" for per-node vector targets
    """

    def __init__(self, model, device):
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

        # New canonical name.
        self.scaler = torch.amp.GradScaler(enabled=False)

        # Backward-compatible alias for older code that used self.scalar.
        self.scalar = self.scaler

    # -------------------------------------------------------------------------
    # Model helpers
    # -------------------------------------------------------------------------

    def print_debug(self):
        print(self.model)
        print(self.training_graphs)
        print(self.validation_graphs)

    def compile_model(self):
        torch._dynamo.config.suppress_errors = True
        self.model = torch.compile(
            self.model,
            mode="default",
            backend="eager",
            dynamic=False,
        )

    def send_model(self, device):
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

    def _model_requires_grad_forward(self):
        """
        Return True when inference/validation forward passes require autograd.

        This is needed for scalar_gradient decoders, where the model output
        contains d scalar / d data.pos.  The GNNBuilder marks this with
        prepare_gradient=True.  We also inspect decoder.output_type as a fallback.
        """
        core = self._core_model()

        if bool(getattr(core, "prepare_gradient", False)):
            return True

        decoder = getattr(core, "decoder", None)
        if decoder is not None:
            output_type = str(getattr(decoder, "output_type", "")).lower()
            if output_type == "scalar_gradient":
                return True

        return False

    def _autocast_device_type(self):
        """
        torch.amp.autocast expects device_type as a string such as 'cuda' or 'cpu'.
        """
        device = self.device
        if isinstance(device, torch.device):
            return device.type
        return str(device).split(":")[0]

    def _nonblocking(self, parameters):
        return bool(parameters.get("device_dict", {}).get("pin_memory", False))

    # -------------------------------------------------------------------------
    # State / optimizer
    # -------------------------------------------------------------------------

    def set_model_state(self, model_state):
        self.model_state = model_state

    def set_optimizer_state(self, optimizer_state):
        self.optimizer_state = optimizer_state

    def set_optimizer_(self, parameters):
        parameters["model_dict"]["optimizer_params"]["params_group"]["params"] = self.model.parameters()
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

        use_amp = bool(parameters["device_dict"].get("use_amp", False))
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.scalar = self.scaler

    def save_checkpoint(self, parameters, epoch, rank=0, fname=None):
        """
        Save model/optimizer/scaler state, rank 0 only.

        Updated checkpoint-cleanup behavior
        -----------------------------------
        If parameters["io_dict"]["remove_old_model"] is True, old epoch
        checkpoints in parameters["io_dict"]["model_dir"] are removed after the
        new checkpoint is saved successfully.

        Default behavior with remove_old_model=True:
            keep only the current checkpoint_epoch_{epoch}.pt file.

        Optional io_dict controls:
            checkpoint_keep_last : int, default 1
                Number of newest epoch checkpoints to keep.

            checkpoint_pattern : str, default "checkpoint_epoch_*.pt"
                Glob pattern used to identify ordinary epoch checkpoints.

            remove_old_checkpoints_on_custom_fname : bool, default False
                If False, cleanup only runs for default checkpoint names. This
                prevents a custom checkpoint such as best_model.pt from deleting
                the normal epoch checkpoint history unexpectedly.

            checkpoint_verbose : bool, default False
                Print deleted checkpoint paths.

        Notes
        -----
        The current checkpoint is written first. Cleanup only happens after a
        successful torch.save(...), so a failed save should not delete the
        previous usable checkpoints.
        """
        if rank != 0:
            return

        io_dict = parameters.get("io_dict", {})
        model_dir = io_dict.get("model_dir", None)

        if model_dir is None:
            raise ValueError(
                "parameters['io_dict']['model_dir'] must be set before saving a checkpoint."
            )

        os.makedirs(model_dir, exist_ok=True)

        using_default_fname = fname is None

        if fname is None:
            fname = os.path.join(
                model_dir,
                f"checkpoint_epoch_{epoch}.pt",
            )

        fname = os.path.abspath(os.fspath(fname))

        core = self._core_model()

        safe_parameters = copy.deepcopy(parameters)

        if "model_dict" in safe_parameters:
            safe_parameters["model_dict"].pop("model", None)
            safe_parameters["model_dict"].pop("optimizer", None)
            safe_parameters["model_dict"].pop("scheduler", None)
            safe_parameters["model_dict"].pop("loss_fn", None)

            # Optimizer param groups contain live parameter generators/tensors;
            # do not serialize them inside the config copy.
            optimizer_params = safe_parameters["model_dict"].get("optimizer_params", None)
            if isinstance(optimizer_params, dict):
                params_group = optimizer_params.get("params_group", None)
                if isinstance(params_group, dict):
                    params_group.pop("params", None)

        checkpoint = {
            "epoch": epoch,
            "model_state": core.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "parameters": safe_parameters,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
        }

        self.checkpoint = checkpoint

        # Save first; cleanup only after successful save.
        torch.save(checkpoint, fname)

        remove_old_model = bool(io_dict.get("remove_old_model", False))
        remove_on_custom = bool(io_dict.get("remove_old_checkpoints_on_custom_fname", False))

        if remove_old_model and (using_default_fname or remove_on_custom):
            checkpoint_pattern = io_dict.get("checkpoint_pattern", "checkpoint_epoch_*.pt")
            keep_last = int(io_dict.get("checkpoint_keep_last", 1))
            checkpoint_verbose = bool(io_dict.get("checkpoint_verbose", False))

            self._cleanup_old_checkpoints(
                model_dir=model_dir,
                current_fname=fname,
                pattern=checkpoint_pattern,
                keep_last=keep_last,
                verbose=checkpoint_verbose,
            )

    def _cleanup_old_checkpoints(
        self,
        model_dir,
        current_fname,
        pattern="checkpoint_epoch_*.pt",
        keep_last=1,
        verbose=False,
    ):
        """
        Remove old epoch checkpoints while keeping the newest keep_last files.

        This only targets files matching pattern, so files such as best_model.pt,
        pretraining checkpoints, run_information.npy, etc. are not touched unless
        the pattern explicitly matches them.
        """
        model_dir = os.path.abspath(os.fspath(model_dir))
        current_fname = os.path.abspath(os.fspath(current_fname))
        keep_last = max(int(keep_last), 1)

        candidates = []

        for path in glob.glob(os.path.join(model_dir, pattern)):
            path = os.path.abspath(os.fspath(path))

            if not os.path.isfile(path):
                continue

            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue

            epoch = self._checkpoint_epoch_from_name(path)
            candidates.append(
                {
                    "path": path,
                    "mtime": mtime,
                    "epoch": epoch,
                    "is_current": os.path.normcase(path) == os.path.normcase(current_fname),
                }
            )

        if not candidates:
            return

        # Sort primarily by parsed epoch when available, otherwise mtime. Newest last.
        candidates.sort(
            key=lambda item: (
                item["epoch"] is not None,
                item["epoch"] if item["epoch"] is not None else -1,
                item["mtime"],
            )
        )

        # Always keep the current checkpoint, plus the newest keep_last - 1 others.
        keep_paths = {os.path.normcase(current_fname)}

        newest_first = list(reversed(candidates))
        for item in newest_first:
            if len(keep_paths) >= keep_last:
                break
            keep_paths.add(os.path.normcase(item["path"]))

        for item in candidates:
            path = item["path"]
            norm_path = os.path.normcase(path)

            if norm_path in keep_paths:
                continue

            try:
                os.remove(path)
                if verbose:
                    print(f"Removed old checkpoint: {path}")
            except FileNotFoundError:
                pass
            except OSError as exc:
                warnings.warn(f"Could not remove old checkpoint {path!r}: {exc}")

    @staticmethod
    def _checkpoint_epoch_from_name(path):
        """
        Parse checkpoint epoch from names like checkpoint_epoch_12.pt.
        Returns None when no epoch is found.
        """
        base = os.path.basename(os.fspath(path))
        match = re.match(r"^checkpoint_epoch_(\d+)\.pt$", base)
        if match is None:
            return None
        return int(match.group(1))

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
        - nested prefixes
        - checkpoints saved as {"model_state": ...}
        - checkpoints saved as {"state_dict": ...}
        - raw state_dict checkpoints
        - scaler_state / scalar_state typo compatibility
        - lazy GenericFeatureEncoder MLP initialization from checkpoint shapes
        """

        if map_location is None:
            map_location = self.device if hasattr(self, "device") else "cpu"

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
                checkpoint = torch.load(fname, map_location=map_location)

        if checkpoint is None:
            raise ValueError(
                "No checkpoint was provided. Pass fname=... or set self.checkpoint first."
            )

        core = self._core_model()

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

        def _strip_known_prefixes_from_key(key):
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

        def _get_model_device(module):
            try:
                return next(module.parameters()).device
            except StopIteration:
                if hasattr(self, "device"):
                    return torch.device(self.device)
                return torch.device("cpu")

        def _infer_mlp_input_dim_from_state(state_dict, prefix):
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
            filtered_state = {}
            skipped = []

            for key, value in model_state.items():
                if key not in core_state:
                    skipped.append((key, "missing_in_model"))
                    continue

                if core_state[key].shape != value.shape:
                    skipped.append(
                        (
                            key,
                            f"shape_mismatch checkpoint={tuple(value.shape)} model={tuple(core_state[key].shape)}",
                        )
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

        scaler_obj = self.scaler if self.scaler is not None else self.scalar

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

        if isinstance(checkpoint, dict):
            self.checkpoint = checkpoint
            return checkpoint.get("epoch", None)

        return None

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def load_data(self, params, format=0, rank=0, load_training=True, samples_file=None):
        if format != 2 and format != -1:
            graph_files = glob.glob(os.path.join(params["io_dict"]["data_dir"], "*"))
            samples = load_dictionary(samples_file)

            if load_training:
                self.training_samples = samples["training"]

            self.validation_samples = samples.get("validation")
            if self.validation_samples is None:
                self.validation_samples = samples.get("gids")
            if self.validation_samples is None:
                raise KeyError(
                    "Samples dictionary must contain either a 'validation' or 'gids' entry."
                )

            if format == 0:
                gids = [PurePath(graph).parts[-1].split(".")[0] for graph in graph_files]

                if len(gids) == 0:
                    raise FileNotFoundError(
                        "No graph files were found in "
                        f"{params['io_dict']['data_dir']!r}."
                    )
            else:
                gids = [safe_torch_load(gname, map_location="cpu")["gid"] for gname in graph_files]

            if load_training:
                _, _, c = np.intersect1d(self.training_samples, gids, return_indices=True)
                selected_graphs = [graph_files[cc] for cc in c]

                if format == 0:
                    self.training_graphs = [safe_torch_load(g, map_location="cpu") for g in selected_graphs]
                else:
                    self.training_graphs = selected_graphs

            _, _, c = np.intersect1d(self.validation_samples, gids, return_indices=True)
            selected_graphs = [graph_files[cc] for cc in c]

            if format == 0:
                self.validation_graphs = [safe_torch_load(g, map_location="cpu") for g in selected_graphs]
            else:
                self.validation_graphs = selected_graphs

        elif format == 2:
            graph_file = load_dictionary(glob.glob(os.path.join(params["io_dict"]["data_dir"], "graphs.data"))[0])
            samples = load_dictionary(samples_file)
            gids = [graph.gid for graph in graph_file["graphs"]]

            if load_training:
                self.training_samples = samples["training"]
                _, _, c = np.intersect1d(self.training_samples, gids, return_indices=True)
                self.training_graphs = [graph_file["graphs"][cc] for cc in c]

            self.validation_samples = samples["validation"]
            _, _, c = np.intersect1d(self.validation_samples, gids, return_indices=True)
            self.validation_graphs = [graph_file["graphs"][cc] for cc in c]

        else:
            graph_data = load_dictionary(os.path.join(params["io_dict"]["data_dir"], "graphs.data"))

            if load_training:
                self.training_graphs = [graph for graph in graph_data["training"]]

            self.validation_graphs = [graph for graph in graph_data["validation"]]

    def set_dataloader(self, cat, training=True, validation=True, epoch=-1):
        if training:
            loader_params = {
                "epoch": epoch,
                "shuffle": cat.parameters["loader_dict"]["shuffle_loader"],
                "batch_size": cat.parameters["loader_dict"]["batch_size"][0],
            }
            self.training_loader = setup_dataloader(
                data=self.training_graphs,
                cat=cat,
                loader_params=loader_params,
            )

        if validation:
            loader_params = {
                "epoch": epoch,
                "shuffle": cat.parameters["loader_dict"]["shuffle_loader"],
                "batch_size": cat.parameters["loader_dict"]["batch_size"][1],
            }
            self.validation_loader = setup_dataloader(
                data=self.validation_graphs,
                cat=cat,
                loader_params=loader_params,
            )

    # -------------------------------------------------------------------------
    # Prediction/loss helpers
    # -------------------------------------------------------------------------

    def _prediction_kwargs(self, parameters):
        """
        Collect optional accumulate_predictions kwargs from the parameter dict.

        Recommended config:

            parameters["model_dict"]["prediction_params"] = {
                "output_key": "vector",
                "target_key": "target_vector",
            }

        For per-atom vector learning, also set:

            parameters["model_dict"]["accumulate_loss"] = "node"
        """
        model_dict = parameters.get("model_dict", {})
        loss_params = model_dict.get("loss_params", {}) or {}
        prediction_params = model_dict.get("prediction_params", {}) or {}

        allowed = {
            "channel_mode",
            "normalize_by",
            "legacy_multichannel_shape",
            "output_key",
            "target_key",
            "target_map",
            "prefer_equivariant_key",
        }

        kwargs = {}

        # Allow loss_params to carry these for backward compatibility, but let
        # prediction_params override them.
        for source in (loss_params, prediction_params):
            for key in allowed:
                if key in source:
                    kwargs[key] = source[key]

        return kwargs

    def _accumulate_predictions(self, pred, data, parameters, return_y=True, loss_tag=None):
        if loss_tag is None:
            loss_tag = parameters["model_dict"]["accumulate_loss"]

        kwargs = self._prediction_kwargs(parameters)

        return accumulate_predictions(
            pred,
            data,
            loss_tag,
            return_y=return_y,
            **kwargs,
        )

    def _align_pred_and_target(self, preds, y):
        """
        Put target on the prediction device/dtype when both are tensors.
        """
        if y is None:
            return preds, y

        if torch.is_tensor(preds) and torch.is_tensor(y):
            y = y.to(device=preds.device, dtype=preds.dtype)

            if y.shape != preds.shape and y.numel() == preds.numel():
                y = y.reshape_as(preds)

        return preds, y

    def _default_prediction_loss(self, preds, y, vec, base_loss):
        """
        Robust default loss.

        The old trainer split vector outputs by row/channel when vec=True.
        That is not what we want for node-level vector fields with shape [N, 3].
        If preds and y are tensors, apply the base loss directly.
        """
        if y is None:
            raise ValueError(
                "No target y was returned by accumulate_predictions. "
                "For equivariant outputs, set data.target_vector/data.target_scalar "
                "or provide prediction_params.target_key."
            )

        if torch.is_tensor(preds) and torch.is_tensor(y):
            return base_loss(preds, y)

        if isinstance(preds, (list, tuple)) and isinstance(y, (list, tuple)):
            loss_list = [base_loss(preds[i], y[i]) for i in range(len(preds))]
            return torch.sum(torch.stack(loss_list))

        if vec and isinstance(preds, (list, tuple)):
            loss_list = [base_loss(preds[i], y[i]) for i in range(len(preds))]
            return torch.sum(torch.stack(loss_list))

        return base_loss(preds, y)

    def _make_loss_fn(self, training_or_params):
        """
        Return a loss function with signature:
            loss_fn(preds, y, vec, data, loss_params) -> tensor
        """
        if isinstance(training_or_params, dict) and "params" in training_or_params:
            parameters = training_or_params["params"]
            custom = training_or_params.get("loss_fn", None)
        else:
            parameters = training_or_params
            custom = None

        loss_params = parameters["model_dict"]["loss_params"]

        if custom is not None:
            return custom

        base_loss = loss_setup(params=loss_params)

        def loss_fn(preds, y, vec, data, loss_params=loss_params):
            return self._default_prediction_loss(preds, y, vec, base_loss)

        return loss_fn

    def _tensor_to_python(self, value):
        """
        Detach tensors recursively for saving/printing.
        """
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()

        if isinstance(value, (list, tuple)):
            return [self._tensor_to_python(v) for v in value]

        if isinstance(value, dict):
            return {k: self._tensor_to_python(v) for k, v in value.items()}

        return value

    # -------------------------------------------------------------------------
    # Train / validate / predict
    # -------------------------------------------------------------------------

    def train(self, training_dict):
        """
        Modular training loop with optional AMP.

        Supports:
            legacy list outputs
            direct tensor outputs
            dict outputs from EquivariantDecoder
            per-node vector targets with accumulate_loss='node'

        training_dict:
          'params'   : full parameter dict
          'loss_fn'  : optional custom loss function with signature:
                       loss_fn(preds, y, vec, data, loss_params) -> tensor
        """
        self.model.train()

        parameters = training_dict["params"]
        loss_params = parameters["model_dict"]["loss_params"]
        use_amp = bool(parameters["device_dict"].get("use_amp", False))

        loss_fn = self._make_loss_fn(training_dict)

        loss = 0.0

        if self.training_loader is None:
            raise RuntimeError("training_loader is None. Call set_dataloader(..., training=True) first.")

        for data in self.training_loader:
            data = data.to(self.device, non_blocking=self._nonblocking(parameters))

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                enabled=use_amp,
                device_type=self._autocast_device_type(),
            ):
                pred = self.model(data)
                preds, y, vec = self._accumulate_predictions(
                    pred,
                    data,
                    parameters,
                    return_y=True,
                )
                preds, y = self._align_pred_and_target(preds, y)
                batch_loss = loss_fn(preds, y, vec, data, loss_params)

            loss += float(batch_loss.detach().item())

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if parameters["device_dict"]["run_ddp"]:
            # reduce_tensor already returns the cross-rank mean. Dividing by
            # world_size again would under-report the loss by that factor.
            loss = reduce_tensor(torch.tensor(loss, device=self.device)).item()

        return loss / len(self.training_loader)

    def validate(self, parameters, rank=0):
        """
        Validation loop.

        Unlike the original version, this method is not decorated with
        @torch.no_grad because scalar_gradient models need autograd during
        forward.  Instead, it conditionally uses no_grad or enable_grad.
        """
        self.model.eval()

        loss_params = parameters["model_dict"]["loss_params"]
        loss_fn = self._make_loss_fn(parameters)

        loss = 0.0
        values = [[], [], []]
        gids = []
        vec = False

        if self.validation_loader is None:
            raise RuntimeError("validation_loader is None. Call set_dataloader(..., validation=True) first.")

        grad_context = torch.enable_grad if self._model_requires_grad_forward() else torch.no_grad

        with grad_context():
            for data in self.validation_loader:
                data = data.to(self.device, non_blocking=self._nonblocking(parameters))

                pred = self.model(data)
                preds, y, vec = self._accumulate_predictions(
                    pred,
                    data,
                    parameters,
                    return_y=True,
                )
                preds, y = self._align_pred_and_target(preds, y)

                batch_loss = loss_fn(preds, y, vec, data, loss_params)
                loss += float(batch_loss.detach().item())

                values[0].append(self._tensor_to_python(preds))
                values[1].append(self._tensor_to_python(y))
                values[2].append(self._tensor_to_python(batch_loss))

                if hasattr(data, "gid"):
                    gids.append(self._tensor_to_python(data.gid))
                elif hasattr(data, "frame_index"):
                    gids.append(self._tensor_to_python(data.frame_index))
                else:
                    gids.append(None)

        test_info = {
            "gids": gids,
            "pred": values[0],
            "y": values[1],
            "loss": values[2],
            "loss_fn": parameters["model_dict"]["accumulate_loss"],
            "vec": bool(vec),
        }

        if parameters["device_dict"]["run_ddp"]:
            test_info = combine_dicts_across_gpus(test_info)

        if parameters["io_dict"]["write_indv_pred"]:
            if rank == 0:
                save_dictionary(
                    fname=os.path.join(parameters["io_dict"]["results_dir"], "indv_pred.data"),
                    data=test_info,
                )

        if parameters["device_dict"]["run_ddp"]:
            # reduce_tensor already returns the cross-rank mean.
            loss = reduce_tensor(torch.tensor(loss, device=self.device)).item()

        return loss / len(self.validation_loader)

    def predict(self, parameters, rank=0):
        """
        Prediction loop.

        For scalar_gradient models, this conditionally enables autograd so the
        model can produce gradient outputs during inference.
        """
        self.model.eval()

        values = []
        gids = []
        vec = False

        if self.validation_loader is None:
            raise RuntimeError("validation_loader is None. Call set_dataloader(..., validation=True) first.")

        grad_context = torch.enable_grad if self._model_requires_grad_forward() else torch.no_grad

        with grad_context():
            for data in self.validation_loader:
                data = data.to(self.device, non_blocking=self._nonblocking(parameters))

                pred = self.model(data)
                preds, vec = self._accumulate_predictions(
                    pred,
                    data,
                    parameters,
                    return_y=False,
                    loss_tag="exact",
                )

                values.append(self._tensor_to_python(preds))

                if hasattr(data, "gid"):
                    gids.append(self._tensor_to_python(data.gid))
                elif hasattr(data, "frame_index"):
                    gids.append(self._tensor_to_python(data.frame_index))
                else:
                    gids.append(None)

        test_info = {
            "gids": gids,
            "pred": values,
            "vec": bool(vec),
        }

        if parameters["device_dict"]["run_ddp"]:
            test_info = combine_dicts_across_gpus(test_info)

        return test_info
