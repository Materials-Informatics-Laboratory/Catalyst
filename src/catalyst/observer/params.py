from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Optional

try:
    import GPUtil
except ImportError:  # optional for system metadata only
    GPUtil = None
import psutil

from .._version import __version__
from ..data.utils import save_dictionary
from .validation import (
    CatalystParameterError,
    deep_merge_parameters,
    emit_validation_warnings,
    load_parameter_file,
    normalize_parameters,
    parameter_leaf_paths,
    resolve_parameter_paths,
    validate_config_parameters,
)


class Catalyst:
    """Catalyst runtime configuration and training-observer object.

    Parameters are assembled in a single canonical order::

        defaults < parameter_file < parameters

    The merged configuration is normalized and validated before construction
    completes. Task/model-dependent validation is intentionally staged because
    those objects may be created after ``Catalyst``.

    Parameters
    ----------
    parameter_file
        Optional JSON file. It may contain Catalyst parameters directly or a
        top-level ``catalyst_parameters`` object as used by repository examples.
    parameters
        Optional nested mapping of explicit overrides. These have precedence over
        values read from ``parameter_file``.
    task
        Optional ``GNNTask``. If supplied, task-aware validation is performed in
        the constructor. A task can also be bound later with :meth:`set_task`.
    save_params
        Save the effective parameter dictionary after successful construction.
    strict_unknown
        Reject unknown configuration keys. Open-ended architecture/loss/optimizer
        sub-dictionaries remain extensible.
    """

    def __init__(
        self,
        parameter_file: str | os.PathLike | None = None,
        parameters: Mapping[str, Any] | None = None,
        task: Any = None,
        *,
        save_params: bool = False,
        strict_unknown: bool = True,
    ):
        super().__init__()
        self.version = __version__
        self.accumulate_loss_options = ["exact", "sum", "node"]
        self.device_options = ["cuda", "cpu"]
        self.optimizer_options = [
            "AdamW", "Adadelta", "Adagrad", "Adam", "SparseAdam", "Adamax",
            "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD",
        ]
        self.task = None
        self._validation_state = "config"
        self._validation_warnings: list[str] = []
        self._strict_unknown = bool(strict_unknown)
        self._explicit_parameter_paths: set[tuple[str, ...]] = set()
        self._parameter_file: Optional[str] = None

        merged = self.default_parameters()
        base_dir = None

        if parameter_file is not None:
            file_parameters, base_dir = load_parameter_file(parameter_file)
            self._parameter_file = str(Path(parameter_file).expanduser().resolve())
            self._explicit_parameter_paths.update(parameter_leaf_paths(file_parameters))
            merged = deep_merge_parameters(
                merged,
                file_parameters,
                strict_unknown=self._strict_unknown,
            )

        if parameters is not None:
            if not isinstance(parameters, Mapping):
                raise TypeError("Catalyst parameters= must be a mapping.")
            self._explicit_parameter_paths.update(parameter_leaf_paths(parameters))
            merged = deep_merge_parameters(
                merged,
                parameters,
                strict_unknown=self._strict_unknown,
            )

        merged = resolve_parameter_paths(merged, base_dir)
        self.parameters = normalize_parameters(merged)
        self.parameters["device_dict"]["system_info"] = self.get_system_info(update_parameters=False)
        self.validate_parameters(stage="config")

        if task is not None:
            self.set_task(task)

        self._set_spawn_method_if_needed()
        if save_params:
            self.save_parameters()

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        """Return a fresh complete Catalyst runtime parameter dictionary."""
        return dict(
            device_dict=dict(
                world_size=1,
                device="cpu",
                ddp_backend="gloo",
                run_ddp=False,
                pin_memory=False,
                find_unused_parameters=False,
                system_info=None,
                use_amp=False,
                amp_dtype="float16",
                float32_matmul_precision=None,
                allow_tf32=None,
                ddp_gradient_as_bucket_view=False,
                ddp_static_graph=False,
                ddp_bucket_cap_mb=None,
            ),
            io_dict=dict(
                main_path="",
                loaded_model_name="",
                data_dir="",
                model_dir="",
                results_dir="",
                samples_dir="",
                projection_dir="",
                remove_old_model=False,
                write_indv_pred=False,
                graph_read_format=0,
                training_info_nwrite_steps=1,
                checkpoint_keep_last=None,
                checkpoint_verbose=False,
                checkpoint_pattern="checkpoint_epoch_*.pt",
                remove_old_checkpoints_on_custom_fname=False,
            ),
            sampling_dict=dict(
                sampling_types=["random", "random"],
                split=[0.5, 0.5],
                sampling_seed=112358,
                params_groups=[{"clusters": 1}, {"clusters": 1}],
            ),
            loader_dict=dict(
                shuffle_loader=False,
                batch_size=[1, 1],
                shuffle_steps=10,
                num_workers=0,
                persistent_workers=False,
                prefetch_factor=2,
                prefetch_to_device=False,
                batch_mode="graphs",
                max_nodes=None,
                max_edges=None,
                dynamic_batch_skip_too_big=False,
                dynamic_batch_num_steps=None,
                follow_batch=None,
            ),
            model_dict=dict(
                n_models=1,
                num_epochs=1,
                train_delta=0.001,
                train_tolerance=1.0,
                worsen_tolerance=0.05,
                patience=0,
                max_deltas=4,
                accumulate_loss="exact",
                loss_params={"function": None},
                prediction_params={
                    "target_key": None,
                    "output_key": None,
                    "prefer_equivariant_key": None,
                    "channel_mode": None,
                    "normalize_by": None,
                    "legacy_multichannel_shape": False,
                    "target_map": None,
                },
                model=None,
                strict_loss_policy=True,
                validation_interval=1,
                compile_model=False,
                compile_backend="inductor",
                compile_mode="default",
                compile_dynamic=True,
                compile_suppress_errors=False,
                model_params_group=dict(encoder={}, processor={}, decoder={}),
                interpretable=False,
                restart_training=False,
                batchnorm=False,
                task=None,
                task_out_dim=None,
                task_target_names=None,
                active_learning=False,
                active_learning_params_group={},
                optimizer_params=dict(
                    lr_scale=[1.0, 0.1],
                    dynamic_lr=False,
                    optimizer="",
                    implementation="default",
                    params_group={"lr": 0.001, "lr_decay_factor": 0.5},
                ),
            ),
        )

    def get_system_info(self, *, update_parameters: bool = True):
        system_info = platform.uname()
        memory_info = psutil.virtual_memory()
        try:
            gpus = [] if GPUtil is None else GPUtil.getGPUs()
        except (FileNotFoundError, OSError):
            gpus = []

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
        if gpus:
            info.update(
                gpu_type=gpus[0].name,
                gpu_driver=gpus[0].driver,
                gpu_memory=gpus[0].memoryTotal,
            )
        if update_parameters and hasattr(self, "parameters"):
            self.parameters["device_dict"]["system_info"] = info
        return info

    def _task_conflict(self, path: tuple[str, ...], current: Any, required: Any) -> None:
        if path in self._explicit_parameter_paths and current != required:
            raise CatalystParameterError(
                "The selected GNNTask conflicts with an explicitly supplied Catalyst "
                f"parameter: {'.'.join(path)}={current!r}, but the task requires {required!r}. "
                "Remove the task-controlled setting from JSON/parameters or choose a compatible task."
            )

    def _validate_task_contract(self, task: Any, parameters: Mapping[str, Any]) -> None:
        required_attrs = (
            "name", "target_key", "output_type", "output_level", "out_dim",
            "accumulate_loss", "apply_to_catalyst_parameters",
        )
        missing = [name for name in required_attrs if not hasattr(task, name)]
        if missing:
            raise TypeError(f"task must be a GNNTask-like object; missing attributes {missing}.")

        model_dict = parameters["model_dict"]
        prediction = model_dict.get("prediction_params", {}) or {}
        checks = {
            ("model_dict", "task"): task.name,
            ("model_dict", "task_out_dim"): int(task.out_dim),
            ("model_dict", "accumulate_loss"): task.accumulate_loss,
            ("model_dict", "prediction_params", "target_key"): task.target_key,
        }
        if task.output_key is not None:
            checks[("model_dict", "prediction_params", "output_key")] = task.output_key
        if task.prefer_equivariant_key is not None:
            checks[("model_dict", "prediction_params", "prefer_equivariant_key")] = task.prefer_equivariant_key

        # All GNNTask presets represent supervised target prediction. Legacy
        # list-output models require channel_mode="target"; direct tensor/dict
        # outputs ignore this setting. Treat it as task-controlled so an explicit
        # incompatible configuration is rejected during staged validation.
        checks[("model_dict", "prediction_params", "channel_mode")] = "target"

        if task.name == "graph_multiscalar":
            checks[("model_dict", "prediction_params", "legacy_multichannel_shape")] = False
            checks[("model_dict", "prediction_params", "normalize_by")] = task.normalize_by
            if task.target_names is not None:
                checks[("model_dict", "task_target_names")] = list(task.target_names)

        for path, required in checks.items():
            if path[:2] == ("model_dict", "prediction_params"):
                current = prediction.get(path[-1], None)
            else:
                current = model_dict.get(path[-1], None)
            self._task_conflict(path, current, required)

        if task.name in {"node_vector", "graph_vector"} and int(task.vector_channels) != 1:
            raise CatalystParameterError("Catalyst 2.2 supports exactly one geometric vector channel.")
        if task.name == "graph_multiscalar" and int(task.out_dim) < 2:
            raise CatalystParameterError("graph_multiscalar requires at least two scalar targets.")

    def _copy_parameters_for_update(self) -> dict[str, Any]:
        """Copy configuration while preserving live runtime model references."""
        model = self.parameters.get("model_dict", {}).get("model")
        source = dict(self.parameters)
        source["model_dict"] = dict(source.get("model_dict", {}))
        source["model_dict"]["model"] = None
        optimizer = dict(source["model_dict"].get("optimizer_params", {}))
        params_group = dict(optimizer.get("params_group", {}))
        params_group.pop("params", None)
        optimizer["params_group"] = params_group
        source["model_dict"]["optimizer_params"] = optimizer
        copied = deepcopy(source)
        copied["model_dict"]["model"] = model
        return copied

    def set_task(self, task: Any):
        """Bind a task and atomically apply/validate its backend contract."""
        candidate = self._copy_parameters_for_update()
        self._validate_task_contract(task, candidate)
        task.apply_to_catalyst_parameters(candidate)
        candidate = normalize_parameters(candidate)
        warnings_out = validate_config_parameters(candidate)
        self.parameters = candidate
        self.task = task
        self._validation_state = "task"
        self._validation_warnings = warnings_out
        emit_validation_warnings(warnings_out)
        return self

    def _set_spawn_method_if_needed(self) -> None:
        device_dict = self.parameters["device_dict"]
        if device_dict.get("run_ddp", False) and device_dict.get("ddp_backend") == "nccl":
            import torch.multiprocessing as mp
            mp.set_start_method("spawn", force=True)

    def set_model(self, model):
        from ..ml.utils.memory import change_model_device, clear_torch_memory

        if model is None:
            raise CatalystParameterError("Catalyst.set_model(model) requires a model object.")

        model_task = getattr(model, "_catalyst_task", None)
        if model_task is None and hasattr(model, "model"):
            model_task = getattr(model.model, "_catalyst_task", None)
        if model_task is not None:
            if self.task is None:
                self.set_task(model_task)
            elif self.task != model_task:
                raise CatalystParameterError(
                    f"The model was built for task {model_task.name!r}, but Catalyst is bound to {self.task.name!r}."
                )

        old = self.parameters["model_dict"].get("model")
        if old is not None and old is not model:
            self.parameters["model_dict"]["model"] = None
            clear_torch_memory()
        self.parameters["model_dict"]["model"] = model
        change_model_device(model, self.parameters["device_dict"]["device"])
        self.validate_parameters(stage="model", model=model)
        return self

    def set_params(
        self,
        new_params: Mapping[str, Any] | None = None,
        *,
        parameter_file: str | os.PathLike | None = None,
        save_params: bool = True,
    ):
        """Atomically merge, normalize, and validate parameter updates.

        Invalid updates leave ``self.parameters`` unchanged.  This is the supported
        replacement for direct post-construction mutation of ``cat.parameters``.
        """
        if new_params is None and parameter_file is None:
            raise ValueError("set_params requires new_params or parameter_file.")

        candidate = self._copy_parameters_for_update()
        new_explicit = set(self._explicit_parameter_paths)
        base_dir = None

        if parameter_file is not None:
            file_params, base_dir = load_parameter_file(parameter_file)
            candidate = deep_merge_parameters(candidate, file_params, strict_unknown=self._strict_unknown)
            new_explicit.update(parameter_leaf_paths(file_params))
        if new_params is not None:
            if not isinstance(new_params, Mapping):
                raise TypeError("new_params must be a mapping.")
            candidate = deep_merge_parameters(candidate, new_params, strict_unknown=self._strict_unknown)
            new_explicit.update(parameter_leaf_paths(new_params))

        candidate = resolve_parameter_paths(candidate, base_dir)
        candidate = normalize_parameters(candidate)

        old_explicit = self._explicit_parameter_paths
        self._explicit_parameter_paths = new_explicit
        try:
            warnings_out = validate_config_parameters(candidate)
            if self.task is not None:
                self._validate_task_contract(self.task, candidate)
                self.task.apply_to_catalyst_parameters(candidate)
                warnings_out = validate_config_parameters(candidate)
        except Exception:
            self._explicit_parameter_paths = old_explicit
            raise

        self.parameters = candidate
        self._validation_warnings = warnings_out
        self._validation_state = "task" if self.task is not None else "config"
        emit_validation_warnings(warnings_out)
        self._set_spawn_method_if_needed()
        if save_params:
            self.save_parameters()
        return self

    def validate_parameters(self, *, stage: str = "config", model: Any = None, rank: int | None = None):
        """Run staged validation and return a compact status dictionary."""
        stage = str(stage).lower().strip()
        if stage not in {"config", "task", "model", "ready"}:
            raise ValueError("stage must be one of: config, task, model, ready.")

        warnings_out = validate_config_parameters(self.parameters)

        if stage in {"task", "model", "ready"}:
            if self.task is None:
                raise CatalystParameterError(
                    f"Validation stage {stage!r} requires a GNNTask. Pass task= to Catalyst(...) or call cat.set_task(task)."
                )
            self._validate_task_contract(self.task, self.parameters)

        if stage in {"model", "ready"}:
            actual_model = model if model is not None else self.parameters["model_dict"].get("model")
            if actual_model is None:
                raise CatalystParameterError(f"Validation stage {stage!r} requires a model.")
            model_task = getattr(actual_model, "_catalyst_task", None)
            if model_task is None and hasattr(actual_model, "model"):
                model_task = getattr(actual_model.model, "_catalyst_task", None)
            if model_task is not None and model_task != self.task:
                raise CatalystParameterError(
                    f"Model task {model_task.name!r} does not match Catalyst task {self.task.name!r}."
                )

        if stage == "ready":
            from ..ml.utils.distributed import validate_ddp_configuration

            validate_ddp_configuration(self.parameters, rank=rank)
            if self.parameters["model_dict"]["loss_params"].get("function") is None:
                raise CatalystParameterError("Training requires model_dict.loss_params.function.")
            optimizer = self.parameters["model_dict"]["optimizer_params"].get("optimizer")
            if not optimizer:
                raise CatalystParameterError("Training requires model_dict.optimizer_params.optimizer.")
            samples_dir = self.parameters["io_dict"].get("samples_dir")
            data_dir = self.parameters["io_dict"].get("data_dir")
            if samples_dir in (None, ""):
                raise CatalystParameterError("Training requires io_dict.samples_dir.")
            if data_dir in (None, ""):
                raise CatalystParameterError("Training requires io_dict.data_dir.")

        self._validation_state = stage
        self._validation_warnings = warnings_out
        emit_validation_warnings(warnings_out)
        return self.validation_status()

    def validate_ready_for_training(self, *, model: Any = None, rank: int | None = None):
        """Mandatory final preflight used immediately before Catalyst training."""
        if self.task is None:
            actual_model = model if model is not None else self.parameters["model_dict"].get("model")
            model_task = getattr(actual_model, "_catalyst_task", None) if actual_model is not None else None
            if model_task is None and actual_model is not None and hasattr(actual_model, "model"):
                model_task = getattr(actual_model.model, "_catalyst_task", None)
            if model_task is not None:
                self.set_task(model_task)
            else:
                # Backward-compatible reconstruction for older/direct builder workflows.
                from ..ml.gnn.tasks import task_from_parameters
                self.set_task(task_from_parameters(self.parameters))
        return self.validate_parameters(stage="ready", model=model, rank=rank)

    def validation_status(self) -> dict[str, Any]:
        return {
            "configuration": True,
            "task": self.task.name if self.task is not None else None,
            "model_bound": self.parameters["model_dict"].get("model") is not None,
            "stage": self._validation_state,
            "warnings": list(self._validation_warnings),
        }

    def print_parameters(self) -> None:
        """Print the effective configuration without non-JSON runtime objects."""
        print(json.dumps(self._serializable_parameters(), indent=2, sort_keys=True, default=str))

    def _serializable_parameters(self) -> dict[str, Any]:
        params = self._copy_parameters_for_update()
        params["model_dict"]["model"] = None
        optimizer_group = params["model_dict"].get("optimizer_params", {}).get("params_group", {})
        optimizer_group.pop("params", None)
        return params

    def save_parameters(self, fname: str | os.PathLike | None = None) -> str:
        """Save the final effective configuration used by Catalyst."""
        if fname is None:
            main_path = self.parameters["io_dict"].get("main_path") or "."
            fname = os.path.join(main_path, "parameters.data")
        fname = str(fname)
        os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
        serializable = self._serializable_parameters()
        if Path(fname).suffix.lower() == ".json":
            with open(fname, "w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2, sort_keys=True, default=str)
        else:
            save_dictionary(fname=fname, data=serializable)
        return fname
