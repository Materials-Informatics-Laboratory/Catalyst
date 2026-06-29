"""
Task definitions for Catalyst GNN models.

Recommended location:
    catalyst/ml/gnn/tasks.py

Purpose
-------
This file defines a small task interface that connects four things that must
agree for a GNN training run to be valid:

    1. model output type and level,
    2. graph target field,
    3. Catalyst prediction/loss accumulation settings,
    4. output/target shape validation.

The task interface does not replace the Catalyst training backend. It only
configures the existing backend consistently.

Public examples
---------------
Graph-level scalar regression:

    task = GNNTask.graph_scalar(target_key="target_scalar")
    task.apply_to_catalyst_parameters(parameters)

    model = build_task_model(
        task=task,
        model_type="alignn",
        num_species=1,
        cutoff=3.35,
        dim=128,
        num_convs=4,
        decoder=my_graph_scalar_decoder,
    )

Node-level vector regression:

    task = GNNTask.node_vector(target_key="target_vector")
    task.apply_to_catalyst_parameters(parameters)

    model = build_task_model(
        task=task,
        model_type="equivariant",
        num_species=1,
        cutoff=3.35,
        dim=128,
        num_convs=4,
    )

The node_vector task learns one complete 3D vector per node. It does not define
three independent scalar tasks.

Notes on equivariant vector outputs
-----------------------------------
The equivariant vector decoder commonly represents C vector channels as:

    [N, C, 3]

For node_vector with one vector per node, this task builds the base equivariant
model with out_dim=1 and wraps it with an adapter that converts:

    [N, 1, 3] -> [N, 3]

before the Catalyst loss sees the prediction.

Task names are intentionally generic:
    graph_scalar
    node_scalar
    node_vector
    graph_vector
    scalar_gradient

Do not add domain-specific names here.
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
    if isinstance(value, dict):
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
        [N, 1, 3]    -> [N, 3]

    A tensor such as [N, 3, 3] is rejected by default because that represents
    three vector channels, not one vector target per item.
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
    ) -> "GNNTask":
        return cls(
            name="graph_scalar",
            target_key=target_key,
            output_type="scalar",
            output_level="graph",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="scalar",
            out_dim=1,
        )

    @classmethod
    def node_scalar(
        cls,
        *,
        target_key: str = "target_scalar",
        output_key: str = "scalar",
        accumulate_loss: str = "node",
    ) -> "GNNTask":
        return cls(
            name="node_scalar",
            target_key=target_key,
            output_type="scalar",
            output_level="node",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="scalar",
            out_dim=1,
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
            squeeze_single_vector_channel=squeeze_single_vector_channel,
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
            squeeze_single_vector_channel=squeeze_single_vector_channel,
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
        """
        Construct a task from its generic task name.

        Valid names:
            graph_scalar
            node_scalar
            node_vector
            graph_vector
            scalar_gradient
        """
        name = str(name).lower()

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
        model_dict["accumulate_loss"] = self.accumulate_loss

        prediction_params = dict(model_dict.get("prediction_params", {}))
        prediction_params["target_key"] = self.target_key

        if self.output_key is not None:
            prediction_params["output_key"] = self.output_key

        if self.prefer_equivariant_key is not None:
            prediction_params["prefer_equivariant_key"] = self.prefer_equivariant_key

        model_dict["prediction_params"] = prediction_params

    def model_kwargs(self) -> Dict[str, Any]:
        """
        Keyword arguments that should be passed into build_model(...).
        """
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
        """
        Validate prediction/target shapes for this task.

        This should be run on one batch before full training starts.
        """
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
    model_type: str,
    num_species: int,
    cutoff: float,
    dim: int,
    num_convs: int,
    decoder: Optional[nn.Module] = None,
    return_dict: bool = False,
    act: Optional[nn.Module] = None,
    apply_task_model_kwargs: bool = True,
    **kwargs: Any,
) -> nn.Module:
    if act is None:
        act = nn.SiLU()

    model_kwargs: Dict[str, Any] = {
        "model_type": model_type,
        "return_dict": return_dict,
        "num_species": num_species,
        "cutoff": cutoff,
        "dim": dim,
        "num_convs": num_convs,
        "act": act,
    }

    if apply_task_model_kwargs:
        model_kwargs.update(task.model_kwargs())

    if decoder is not None:
        model_kwargs["decoder"] = decoder

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

    with torch.no_grad():
        pred = model(batch)

    if not hasattr(batch, task.target_key):
        raise AttributeError(
            f"Batch does not contain target field {task.target_key!r}. "
            f"Available public fields include: "
            f"{[name for name in dir(batch) if not name.startswith('_')][:40]}..."
        )

    target = getattr(batch, task.target_key)
    task.validate_prediction_and_target(pred, target)

    if print_summary:
        print("GNN task batch validation passed.")
        print(f"  task:        {task.name}")
        print(f"  target_key:  {task.target_key}")
        print(f"  output_type: {task.output_type}")
        print(f"  output_level:{task.output_level}")
        print(f"  pred shape:  {_shape(pred)}")
        print(f"  target shape:{_shape(target)}")


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

    if task_name is not None:
        kwargs: Dict[str, Any] = {}
        if target_key is not None:
            kwargs["target_key"] = target_key
        return GNNTask.from_name(str(task_name), **kwargs)

    output_key = prediction_params.get("output_key", None)
    prefer_key = prediction_params.get("prefer_equivariant_key", None)
    accumulate_loss = model_dict.get("accumulate_loss", "exact")

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
