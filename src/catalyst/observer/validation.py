"""Central configuration validation for Catalyst runtime parameters.

Validation is intentionally staged:

1. ``config`` validates the merged Catalyst configuration without requiring a task/model.
2. ``task`` validates task-controlled settings once a :class:`GNNTask` is bound.
3. ``ready`` validates the complete training preflight once the model/runtime exists.

The validator is kept free of imports from ``catalyst.ml.gnn`` to avoid circular imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Iterable
import json
import warnings


class CatalystParameterError(ValueError):
    """Raised when Catalyst parameters are invalid or mutually inconsistent."""


# Dictionaries below these paths are intentionally open-ended because they contain
# architecture-, optimizer-, loss-, or sampling-specific keyword arguments.
OPEN_MAPPING_PATHS = {
    ("sampling_dict", "params_groups"),
    ("model_dict", "model_params_group"),
    ("model_dict", "model_params_group", "encoder"),
    ("model_dict", "model_params_group", "processor"),
    ("model_dict", "model_params_group", "decoder"),
    ("model_dict", "optimizer_params", "params_group"),
    ("model_dict", "loss_params"),
    ("model_dict", "active_learning_params_group"),
}


def load_parameter_file(parameter_file: str | Path) -> tuple[dict[str, Any], Path]:
    """Load JSON parameters and return ``(parameters, file_parent)``.

    Example/workflow JSON files may contain a top-level ``catalyst_parameters``
    section.  A JSON file containing only Catalyst parameters is also accepted.
    """
    path = Path(parameter_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Catalyst parameter file does not exist: {path}")
    if path.suffix.lower() != ".json":
        raise CatalystParameterError(
            f"Catalyst parameter_file must be JSON; received {path.name!r}."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CatalystParameterError(
            f"Could not parse Catalyst JSON parameter file {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CatalystParameterError("Catalyst JSON parameters must contain an object at the top level.")
    if "catalyst_parameters" in payload:
        payload = payload["catalyst_parameters"]
    if not isinstance(payload, Mapping):
        raise CatalystParameterError("'catalyst_parameters' must be a JSON object.")
    return deepcopy(dict(payload)), path.parent


def parameter_leaf_paths(mapping: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Return leaf paths explicitly supplied by a user mapping."""
    result: set[tuple[str, ...]] = set()
    for key, value in mapping.items():
        path = prefix + (str(key),)
        if isinstance(value, Mapping) and value:
            result.update(parameter_leaf_paths(value, path))
        else:
            result.add(path)
    return result


def _is_open_path(path: tuple[str, ...]) -> bool:
    return any(path[: len(open_path)] == open_path for open_path in OPEN_MAPPING_PATHS)


def _unknown_key_error(path: tuple[str, ...], key: str, choices: Iterable[str]) -> CatalystParameterError:
    full = ".".join(path + (key,))
    suggestions = get_close_matches(key, list(choices), n=1, cutoff=0.72)
    suffix = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
    return CatalystParameterError(f"Unknown Catalyst parameter {full!r}.{suffix}")


def deep_merge_parameters(
    base: Mapping[str, Any],
    update: Mapping[str, Any],
    *,
    strict_unknown: bool = True,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deep merge a parameter mapping without mutating either input."""
    if not isinstance(update, Mapping):
        raise CatalystParameterError(
            f"Expected a mapping at {'.'.join(path) or '<root>'}; got {type(update).__name__}."
        )

    result = deepcopy(dict(base))
    for key, value in update.items():
        key = str(key)
        current_path = path + (key,)
        if key not in result:
            if strict_unknown and not _is_open_path(path):
                raise _unknown_key_error(path, key, result.keys())
            result[key] = deepcopy(value)
            continue

        existing = result[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge_parameters(
                existing,
                value,
                strict_unknown=strict_unknown,
                path=current_path,
            )
        else:
            result[key] = deepcopy(value)
    return result


def resolve_parameter_paths(parameters: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    """Resolve relative Catalyst I/O paths against the parameter-file directory."""
    if base_dir is None:
        return parameters
    io_dict = parameters.get("io_dict", {})
    for key in ("main_path", "data_dir", "model_dir", "results_dir", "samples_dir", "projection_dir"):
        value = io_dict.get(key)
        if value in (None, ""):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            io_dict[key] = str((base_dir / path).resolve())
        else:
            io_dict[key] = str(path)
    return parameters


def normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize equivalent representations before consistency validation."""
    p = deepcopy(dict(parameters))
    device = p["device_dict"]
    loader = p["loader_dict"]
    model = p["model_dict"]

    device["device"] = str(device.get("device") or "cpu").strip().lower()
    device["ddp_backend"] = str(device.get("ddp_backend") or ("nccl" if device["device"].startswith("cuda") else "gloo")).strip().lower()
    device["amp_dtype"] = str(device.get("amp_dtype") or "float16").strip().lower()
    aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16"}
    device["amp_dtype"] = aliases.get(device["amp_dtype"], device["amp_dtype"])

    loader["batch_mode"] = str(loader.get("batch_mode") or "graphs").strip().lower()
    optimizer = model.setdefault("optimizer_params", {})
    optimizer["implementation"] = str(optimizer.get("implementation") or "default").strip().lower()
    model["compile_backend"] = str(model.get("compile_backend") or "inductor").strip().lower()
    model["compile_mode"] = str(model.get("compile_mode") or "default").strip().lower()

    batch_size = loader.get("batch_size", [1, 1])
    if isinstance(batch_size, int):
        loader["batch_size"] = [batch_size, batch_size]
    elif isinstance(batch_size, tuple):
        loader["batch_size"] = list(batch_size)

    return p


def _require_bool(container: Mapping[str, Any], key: str, path: str) -> None:
    if not isinstance(container.get(key), bool):
        raise CatalystParameterError(f"{path}.{key} must be boolean.")


def validate_config_parameters(parameters: Mapping[str, Any]) -> list[str]:
    """Validate configuration-only constraints and return nonfatal warning strings."""
    warnings_out: list[str] = []
    p = parameters
    device = p["device_dict"]
    io = p["io_dict"]
    sampling = p["sampling_dict"]
    loader = p["loader_dict"]
    model = p["model_dict"]

    if device["device"] != "cpu" and not device["device"].startswith("cuda"):
        raise CatalystParameterError("device_dict.device must be 'cpu', 'cuda', or 'cuda:<index>'.")
    if int(device["world_size"]) < 1:
        raise CatalystParameterError("device_dict.world_size must be >= 1.")
    if device["amp_dtype"] not in {"float16", "bfloat16"}:
        raise CatalystParameterError("device_dict.amp_dtype must be 'float16' or 'bfloat16'.")
    for key in ("run_ddp", "pin_memory", "find_unused_parameters", "use_amp", "ddp_gradient_as_bucket_view", "ddp_static_graph"):
        _require_bool(device, key, "device_dict")

    if device["run_ddp"]:
        if not device["device"].startswith("cuda"):
            raise CatalystParameterError("run_ddp=True requires device_dict.device='cuda' or 'cuda:<index>'.")
        if device["ddp_backend"] != "nccl":
            raise CatalystParameterError("Catalyst CUDA DDP requires device_dict.ddp_backend='nccl'.")
        if int(device["world_size"]) < 2:
            warnings_out.append("run_ddp=True with world_size=1 is valid for debugging but provides no multi-GPU speedup.")
    elif int(device["world_size"]) != 1:
        warnings_out.append("device_dict.world_size is ignored while run_ddp=False.")

    if device["ddp_bucket_cap_mb"] is not None and float(device["ddp_bucket_cap_mb"]) <= 0:
        raise CatalystParameterError("device_dict.ddp_bucket_cap_mb must be positive when provided.")
    if device["float32_matmul_precision"] not in {None, "highest", "high", "medium"}:
        raise CatalystParameterError("device_dict.float32_matmul_precision must be one of None, 'highest', 'high', 'medium'.")

    batch_size = loader.get("batch_size")
    if not isinstance(batch_size, (list, tuple)) or len(batch_size) != 2 or any(int(v) <= 0 for v in batch_size):
        raise CatalystParameterError("loader_dict.batch_size must contain two positive integers: [train, validation].")
    if int(loader["num_workers"]) < 0:
        raise CatalystParameterError("loader_dict.num_workers must be >= 0.")
    if int(loader["shuffle_steps"]) < 1:
        raise CatalystParameterError("loader_dict.shuffle_steps must be >= 1.")
    if loader["persistent_workers"] and int(loader["num_workers"]) == 0:
        raise CatalystParameterError("persistent_workers=True requires loader_dict.num_workers > 0.")
    if loader["prefetch_factor"] is not None and int(loader["prefetch_factor"]) < 1:
        raise CatalystParameterError("loader_dict.prefetch_factor must be >= 1 when provided.")
    if int(loader["num_workers"]) == 0 and loader["prefetch_factor"] not in (None, 2):
        warnings_out.append("loader_dict.prefetch_factor is ignored when num_workers=0.")
    if loader["prefetch_to_device"] and not device["device"].startswith("cuda"):
        raise CatalystParameterError("loader_dict.prefetch_to_device=True requires a CUDA device.")

    mode = loader["batch_mode"]
    if mode not in {"graphs", "nodes", "edges"}:
        raise CatalystParameterError("loader_dict.batch_mode must be 'graphs', 'nodes', or 'edges'.")
    if mode == "nodes":
        if loader["max_nodes"] is None or int(loader["max_nodes"]) <= 0:
            raise CatalystParameterError("batch_mode='nodes' requires loader_dict.max_nodes to be a positive integer.")
        if loader["max_edges"] is not None:
            raise CatalystParameterError("batch_mode='nodes' conflicts with loader_dict.max_edges; set max_edges=None.")
    elif mode == "edges":
        if loader["max_edges"] is None or int(loader["max_edges"]) <= 0:
            raise CatalystParameterError("batch_mode='edges' requires loader_dict.max_edges to be a positive integer.")
        if loader["max_nodes"] is not None:
            raise CatalystParameterError("batch_mode='edges' conflicts with loader_dict.max_nodes; set max_nodes=None.")
    else:
        if loader["max_nodes"] is not None or loader["max_edges"] is not None:
            warnings_out.append("max_nodes/max_edges are ignored while batch_mode='graphs'.")

    if int(model["n_models"]) < 1:
        raise CatalystParameterError("model_dict.n_models must be >= 1.")
    if int(model["num_epochs"]) < 1:
        raise CatalystParameterError("model_dict.num_epochs must be >= 1.")
    if float(model["train_delta"]) < 0 or float(model["train_tolerance"]) < 0:
        raise CatalystParameterError("train_delta and train_tolerance must be >= 0.")
    if float(model["worsen_tolerance"]) < 0:
        raise CatalystParameterError("model_dict.worsen_tolerance must be >= 0.")
    if int(model["max_deltas"]) < 0 or int(model["patience"]) < 0:
        raise CatalystParameterError("model_dict.max_deltas and patience must be >= 0.")
    if int(model["validation_interval"]) < 1:
        raise CatalystParameterError("model_dict.validation_interval must be >= 1.")
    if model["accumulate_loss"] not in {"exact", "sum", "node"}:
        raise CatalystParameterError("model_dict.accumulate_loss must be 'exact', 'sum', or 'node'.")

    implementation = model["optimizer_params"]["implementation"]
    if implementation not in {"default", "auto", "fused", "foreach", "for_loop"}:
        raise CatalystParameterError(
            "model_dict.optimizer_params.implementation must be one of default, auto, fused, foreach, for_loop."
        )
    optimizer_name = model["optimizer_params"].get("optimizer", "")
    if optimizer_name not in {"", None} and not isinstance(optimizer_name, str):
        raise CatalystParameterError("model_dict.optimizer_params.optimizer must be an optimizer name string.")

    if model["compile_model"] is False and (
        model["compile_backend"] != "inductor" or model["compile_mode"] != "default" or model["compile_dynamic"] is not True
    ):
        warnings_out.append("compile_* settings are currently dormant because model_dict.compile_model=False.")
    if not device["use_amp"] and device["amp_dtype"] != "float16":
        warnings_out.append("device_dict.amp_dtype is dormant because use_amp=False.")

    splits = sampling.get("split", [])
    types = sampling.get("sampling_types", [])
    groups = sampling.get("params_groups", [])
    if not (len(splits) == len(types) == len(groups)):
        raise CatalystParameterError("sampling_dict.split, sampling_types, and params_groups must have matching lengths.")
    for split in splits:
        if not 0 <= float(split) <= 1:
            raise CatalystParameterError("Each sampling_dict.split value must lie in [0, 1].")

    for key in ("training_info_nwrite_steps", "checkpoint_keep_last"):
        value = io.get(key)
        if value is not None and int(value) < 1:
            raise CatalystParameterError(f"io_dict.{key} must be >= 1 when provided.")

    return warnings_out


def emit_validation_warnings(messages: Iterable[str]) -> None:
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=3)
