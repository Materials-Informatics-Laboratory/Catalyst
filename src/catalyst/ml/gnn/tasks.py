"""
Task definitions for Catalyst GNN models.

Recommended location:
    catalyst/ml/gnn/tasks.py

This module defines a small task interface that keeps four pieces of a GNN run
consistent:

    1. model output type and output level,
    2. graph target field,
    3. Catalyst prediction/loss accumulation settings,
    4. output/target shape validation.

The task interface does not replace the Catalyst training backend. It only
configures the existing backend consistently.

Task names are intentionally generic:
    graph_scalar
    node_scalar
    node_vector
    graph_vector
    scalar_gradient
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional

import torch
from torch import nn

from catalyst.ml.gnn.modules.models.gnn_builder import build_model


TaskName = Literal[
    "graph_scalar",
    "node_scalar",
    "node_vector",
    "graph_vector",
    "scalar_gradient",
]

OutputType = Literal[
    "scalar",
    "vector",
    "scalar_gradient",
]

OutputLevel = Literal[
    "graph",
    "node",
]


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, Mapping):
        return "{" + ", ".join(str(k) for k in value.keys()) + "}"
    return str(type(value))


def _get_output_from_mapping(output: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in output and output[key] is not None:
            return output[key]

    raise KeyError(
        "Could not find any expected output key in model output. "
        f"Expected one of {list(keys)}, got keys {list(output.keys())}."
    )


class VectorChannelAdapter(nn.Module):
    """
    Convert vector-channel outputs to ordinary vector tensors when needed.

    Accepted input conventions:
        [N, 3]       -> unchanged
        [N, 1, 3]    -> [N, 3] when vector_channels=1

    A tensor such as [N, 3, 3] is rejected by default for node_vector because it
    represents three vector channels, not one complete 3D vector target per node.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        vector_channels: int = 1,
        squeeze_single_channel: bool = True,
        output_keys: tuple[str, ...] = ("vector", "pred", "output"),
    ):
        super().__init__()
        self.model = model
        self.vector_channels = int(vector_channels)
        self.squeeze_single_channel = bool(squeeze_single_channel)
        self.output_keys = tuple(output_keys)

    def forward(self, data):
        output = self.model(data)

        if isinstance(output, Mapping):
            output = _get_output_from_mapping(output, self.output_keys)

        if not torch.is_tensor(output):
            raise TypeError(
                "VectorChannelAdapter expected the wrapped model to return a tensor "
                f"or a dict containing a tensor. Got {_shape(output)}."
            )

        if output.dim() == 2:
            if output.size(-1) != 3:
                raise RuntimeError(
                    "VectorChannelAdapter received a 2D tensor, but the last "
                    f"dimension is not 3. Got shape {_shape(output)}."
                )
            return output

        if output.dim() == 3:
            if output.size(-1) != 3:
                raise RuntimeError(
                    "VectorChannelAdapter expected vector-channel output shape "
                    f"[N, C, 3], but got {_shape(output)}."
                )

            if output.size(1) != self.vector_channels:
                raise RuntimeError(
                    "Unexpected number of vector channels. "
                    f"Got output shape {_shape(output)} but expected "
                    f"{self.vector_channels} vector channel(s)."
                )

            if self.squeeze_single_channel and self.vector_channels == 1:
                return output[:, 0, :]

            return output

        raise RuntimeError(
            "VectorChannelAdapter expected vector output shape [N, 3] or "
            f"[N, C, 3], but got {_shape(output)}."
        )


@dataclass(frozen=True)
class GNNTask:
    """
    Generic GNN task contract.

    The task defines:
        - output_type and output_level for model construction,
        - target_key for reading labels from graph batches,
        - accumulate_loss and prediction_params for the Catalyst backend,
        - expected output/target shapes for a one-batch validation check.
    """

    name: TaskName
    target_key: str
    output_type: OutputType
    output_level: OutputLevel
    accumulate_loss: str
    output_key: Optional[str] = None
    prefer_equivariant_key: Optional[str] = None
    out_dim: int = 1
    vector_channels: int = 1
    requires_vector_adapter: bool = False
    squeeze_single_vector_channel: bool = True

    @classmethod
    def graph_scalar(
        cls,
        *,
        target_key: str = "target_scalar",
        output_key: str = "scalar",
        accumulate_loss: str = "exact",
        out_dim: int = 1,
    ) -> "GNNTask":
        return cls(
            name="graph_scalar",
            target_key=target_key,
            output_type="scalar",
            output_level="graph",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="scalar",
            out_dim=int(out_dim),
        )

    @classmethod
    def node_scalar(
        cls,
        *,
        target_key: str = "target_scalar",
        output_key: str = "scalar",
        accumulate_loss: str = "node",
        out_dim: int = 1,
    ) -> "GNNTask":
        return cls(
            name="node_scalar",
            target_key=target_key,
            output_type="scalar",
            output_level="node",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="scalar",
            out_dim=int(out_dim),
        )

    @classmethod
    def node_vector(
        cls,
        *,
        target_key: str = "target_vector",
        output_key: str = "vector",
        accumulate_loss: str = "node",
        vector_channels: int = 1,
        squeeze_single_vector_channel: bool = True,
    ) -> "GNNTask":
        return cls(
            name="node_vector",
            target_key=target_key,
            output_type="vector",
            output_level="node",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="vector",
            out_dim=int(vector_channels),
            vector_channels=int(vector_channels),
            requires_vector_adapter=True,
            squeeze_single_vector_channel=bool(squeeze_single_vector_channel),
        )

    @classmethod
    def graph_vector(
        cls,
        *,
        target_key: str = "target_vector",
        output_key: str = "vector",
        accumulate_loss: str = "exact",
        vector_channels: int = 1,
        squeeze_single_vector_channel: bool = True,
    ) -> "GNNTask":
        return cls(
            name="graph_vector",
            target_key=target_key,
            output_type="vector",
            output_level="graph",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="vector",
            out_dim=int(vector_channels),
            vector_channels=int(vector_channels),
            requires_vector_adapter=True,
            squeeze_single_vector_channel=bool(squeeze_single_vector_channel),
        )

    @classmethod
    def scalar_gradient(
        cls,
        *,
        target_key: str = "target_vector",
        output_key: str = "gradient",
        accumulate_loss: str = "node",
    ) -> "GNNTask":
        return cls(
            name="scalar_gradient",
            target_key=target_key,
            output_type="scalar_gradient",
            output_level="node",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="gradient",
            out_dim=1,
        )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "GNNTask":
        name = str(name).lower().strip()

        if name == "graph_scalar":
            return cls.graph_scalar(**kwargs)
        if name == "node_scalar":
            return cls.node_scalar(**kwargs)
        if name == "node_vector":
            return cls.node_vector(**kwargs)
        if name == "graph_vector":
            return cls.graph_vector(**kwargs)
        if name == "scalar_gradient":
            return cls.scalar_gradient(**kwargs)

        raise ValueError(
            f"Unknown GNN task {name!r}. Valid options are: "
            "graph_scalar, node_scalar, node_vector, graph_vector, scalar_gradient."
        )

    def with_target_key(self, target_key: str) -> "GNNTask":
        return replace(self, target_key=target_key)

    def apply_to_catalyst_parameters(self, parameters: MutableMapping[str, Any]) -> None:
        """
        Mutate a Catalyst parameter dictionary so the backend reads the correct
        target field and uses the correct accumulation mode.
        """
        model_dict = parameters.setdefault("model_dict", {})
        model_dict["task"] = self.name
        model_dict["accumulate_loss"] = self.accumulate_loss

        prediction_params = dict(model_dict.get("prediction_params", {}))
        prediction_params["target_key"] = self.target_key

        if self.output_key is not None:
            prediction_params["output_key"] = self.output_key

        if self.prefer_equivariant_key is not None:
            prediction_params["prefer_equivariant_key"] = self.prefer_equivariant_key

        model_dict["prediction_params"] = prediction_params

    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "output_type": self.output_type,
            "output_level": self.output_level,
            "out_dim": self.out_dim,
        }

    def wrap_model_if_needed(self, model: nn.Module) -> nn.Module:
        if not self.requires_vector_adapter:
            return model

        return VectorChannelAdapter(
            model,
            vector_channels=self.vector_channels,
            squeeze_single_channel=self.squeeze_single_vector_channel,
            output_keys=tuple(
                key
                for key in (
                    self.output_key,
                    self.prefer_equivariant_key,
                    "vector",
                    "pred",
                    "output",
                )
                if key is not None
            ),
        )

    def validate_prediction_and_target(
        self,
        pred: Any,
        target: Any,
        *,
        allow_graph_scalar_column: bool = True,
    ) -> None:
        if isinstance(pred, Mapping):
            pred = _get_output_from_mapping(
                pred,
                tuple(
                    key
                    for key in (
                        self.output_key,
                        self.prefer_equivariant_key,
                        "pred",
                        "output",
                    )
                    if key is not None
                ),
            )

        if not torch.is_tensor(pred):
            raise TypeError(f"Prediction must be a tensor. Got {_shape(pred)}.")
        if not torch.is_tensor(target):
            raise TypeError(f"Target must be a tensor. Got {_shape(target)}.")

        if self.name == "graph_scalar":
            valid_pred = pred.dim() == 1 or (
                allow_graph_scalar_column and pred.dim() == 2 and pred.size(-1) == 1
            )
            valid_target = target.dim() == 1 or (
                allow_graph_scalar_column and target.dim() == 2 and target.size(-1) == 1
            )

            if not valid_pred or not valid_target:
                raise RuntimeError(
                    "graph_scalar expects pred/target shape [B] or [B, 1]. "
                    f"Got pred={_shape(pred)}, target={_shape(target)}."
                )

            if pred.numel() != target.numel():
                raise RuntimeError(
                    "graph_scalar prediction/target element counts do not match. "
                    f"Got pred={_shape(pred)}, target={_shape(target)}."
                )
            return

        if self.name == "node_scalar":
            if pred.numel() != target.numel():
                raise RuntimeError(
                    "node_scalar prediction/target element counts do not match. "
                    f"Got pred={_shape(pred)}, target={_shape(target)}."
                )
            return

        if self.name in {"node_vector", "graph_vector"}:
            if pred.dim() != 2 or pred.size(-1) != 3:
                raise RuntimeError(
                    f"{self.name} expects prediction shape [N, 3] or [B, 3]. "
                    f"Got pred={_shape(pred)}. If the raw equivariant model returns "
                    "[N, 1, 3], wrap it with VectorChannelAdapter."
                )

            if target.dim() != 2 or target.size(-1) != 3:
                raise RuntimeError(
                    f"{self.name} expects target {self.target_key!r} shape "
                    f"[N, 3] or [B, 3]. Got target={_shape(target)}."
                )

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"{self.name} prediction/target shape mismatch: "
                    f"pred={_shape(pred)}, target={_shape(target)}."
                )
            return

        if self.name == "scalar_gradient":
            if target.dim() != 2 or target.size(-1) != 3:
                raise RuntimeError(
                    "scalar_gradient expects a vector target with shape [N, 3]. "
                    f"Got target={_shape(target)}."
                )
            return

        raise RuntimeError(f"No validation rule implemented for task {self.name!r}.")


def build_task_model(
    *,
    task: GNNTask,
    model_type: Optional[str] = None,
    preset: Optional[str] = None,
    num_species: Optional[int] = None,
    cutoff: Optional[float] = None,
    dim: Optional[int] = None,
    num_convs: Optional[int] = None,
    decoder: Optional[nn.Module] = None,
    return_dict: Optional[bool] = None,
    act: Optional[nn.Module] = None,
    apply_task_model_kwargs: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a model through build_model(...) and apply any task wrapper.

    Two modes are supported.

    Task-native mode:
        build_task_model(
            task=GNNTask.node_vector(...),
            model_type="equivariant",
            num_species=...,
            cutoff=...,
            dim=...,
            num_convs=...,
        )

        The task supplies output_type, output_level, and out_dim.

    Passthrough/preset mode:
        build_task_model(
            task=GNNTask.graph_scalar(...),
            apply_task_model_kwargs=False,
            preset="alignn",
            decoder=CustomReadout(...),
            ...
        )

        The model architecture is not modified by the task. The task still
        configures the Catalyst backend separately via apply_to_catalyst_parameters.

    Important:
        apply_task_model_kwargs is consumed here and is never forwarded to
        build_model(...) or build_gnn_builder(...).
    """
    model_kwargs: Dict[str, Any] = {}

    if preset is not None:
        # build_model(...) prioritizes model_type over preset. If the caller
        # supplies preset="alignn" and model_type="gnn_builder", route by preset
        # and intentionally do not forward model_type.
        if model_type is not None and str(model_type).lower().strip() not in {
            "gnn_builder",
            "generic",
            "generic_gnn",
        }:
            raise ValueError(
                "Cannot combine preset with a non-generic model_type. "
                f"Got preset={preset!r}, model_type={model_type!r}."
            )
        model_kwargs["preset"] = preset
    elif model_type is not None:
        model_kwargs["model_type"] = model_type

    if return_dict is not None:
        model_kwargs["return_dict"] = return_dict

    if num_species is not None:
        model_kwargs["num_species"] = num_species

    if cutoff is not None:
        model_kwargs["cutoff"] = cutoff

    if dim is not None:
        model_kwargs["dim"] = dim

    if num_convs is not None:
        model_kwargs["num_convs"] = num_convs

    if act is not None:
        model_kwargs["act"] = act

    if apply_task_model_kwargs:
        for key, value in task.model_kwargs().items():
            model_kwargs.setdefault(key, value)

    if decoder is not None:
        model_kwargs["decoder"] = decoder

    # Caller kwargs have final precedence, except for task-control keywords that
    # must never leak to build_model(...).
    kwargs = dict(kwargs)
    kwargs.pop("apply_task_model_kwargs", None)
    model_kwargs.update(kwargs)

    model = build_model(**model_kwargs)
    return task.wrap_model_if_needed(model)


def validate_task_batch(
    *,
    task: GNNTask,
    model: nn.Module,
    batch: Any,
    device: Optional[str | torch.device] = None,
    print_summary: bool = True,
) -> None:
    """
    Run one batch through a model and validate output/target compatibility.
    This is intended as a cheap pre-training check.
    """
    if device is not None:
        model = model.to(device)
        batch = batch.to(device)

    model.eval()

    requires_grad = task.name == "scalar_gradient"
    context = torch.enable_grad() if requires_grad else torch.no_grad()

    with context:
        pred = model(batch)

    if not hasattr(batch, task.target_key):
        available = [name for name in dir(batch) if not name.startswith("_")][:50]
        raise AttributeError(
            f"Batch does not contain target field {task.target_key!r}. "
            f"Available public fields include: {available}..."
        )

    target = getattr(batch, task.target_key)
    task.validate_prediction_and_target(pred, target)

    if print_summary:
        print("GNN task batch validation passed.")
        print(f"  task:         {task.name}")
        print(f"  target_key:   {task.target_key}")
        print(f"  output_type:  {task.output_type}")
        print(f"  output_level: {task.output_level}")
        print(f"  pred shape:   {_shape(pred)}")
        print(f"  target shape: {_shape(target)}")


def task_from_parameters(parameters: Mapping[str, Any], *, default_task: str = "graph_scalar") -> GNNTask:
    """
    Reconstruct a generic task object from a Catalyst parameter dictionary.

    If parameters["model_dict"]["task"] is present, it is used directly.
    Otherwise this infers the task from accumulate_loss and prediction_params.
    """
    model_dict = parameters.get("model_dict", {})
    task_name = model_dict.get("task", None)

    prediction_params = model_dict.get("prediction_params", {})
    target_key = prediction_params.get("target_key", None)
    output_key = prediction_params.get("output_key", None)
    prefer_key = prediction_params.get("prefer_equivariant_key", None)
    accumulate_loss = model_dict.get("accumulate_loss", "exact")

    if task_name is not None:
        kwargs: Dict[str, Any] = {}
        if target_key is not None:
            kwargs["target_key"] = target_key
        if output_key is not None:
            kwargs["output_key"] = output_key
        if accumulate_loss is not None:
            kwargs["accumulate_loss"] = accumulate_loss
        return GNNTask.from_name(str(task_name), **kwargs)

    if output_key == "vector" or prefer_key == "vector":
        if accumulate_loss == "node":
            return GNNTask.node_vector(target_key=target_key or "target_vector")
        return GNNTask.graph_vector(target_key=target_key or "target_vector")

    if output_key == "gradient" or prefer_key == "gradient":
        return GNNTask.scalar_gradient(target_key=target_key or "target_vector")

    if accumulate_loss == "node":
        return GNNTask.node_scalar(target_key=target_key or "target_scalar")

    return GNNTask.graph_scalar(target_key=target_key or "target_scalar")


__all__ = [
    "GNNTask",
    "VectorChannelAdapter",
    "build_task_model",
    "validate_task_batch",
    "task_from_parameters",
]
