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
    graph_multiscalar
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
    "graph_multiscalar",
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
        if self.vector_channels != 1:
            raise ValueError(
                "Catalyst 2.2 supports exactly one geometric vector channel per "
                "node_vector/graph_vector task. Use graph_multiscalar for multiple "
                "independent scalar targets. A true multivector task is not yet "
                "part of the supported API."
            )
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


class GraphMultiScalarAdapter(nn.Module):
    """
    Convert order-wise K-channel decoder outputs into ``[B, K]`` predictions.

    Standard Catalyst order decoders return a list containing atom-, bond-, and
    optionally angle-level contributions.  For a graph multiscalar task, every
    last-dimension channel is an independent invariant scalar target.  This
    adapter graph-pools those contributions without treating the channels as a
    geometric vector.

    Direct tensor or mapping outputs from an equivariant/custom decoder are
    accepted unchanged after shape validation.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        num_targets: int,
        normalize_by: Optional[str] = "primary_nodes",
        output_keys: tuple[str, ...] = ("scalar", "pred", "output"),
    ):
        super().__init__()
        self.model = model
        self.num_targets = int(num_targets)
        self.normalize_by = normalize_by
        self.output_keys = tuple(output_keys)

        if self.num_targets < 2:
            raise ValueError(
                "GraphMultiScalarAdapter requires num_targets >= 2. "
                "Use graph_scalar for a single scalar target."
            )

    @staticmethod
    def _num_graphs(data) -> Optional[int]:
        explicit = getattr(data, "num_graphs", None)
        if explicit is not None:
            return int(explicit)

        for batch_attr in ("x_atm_batch", "node_G_batch", "batch"):
            batch = getattr(data, batch_attr, None)
            if torch.is_tensor(batch):
                if batch.numel() == 0:
                    return 0
                return int(batch.max().item()) + 1

        return None

    def _validate_tensor(
        self,
        output: torch.Tensor,
        *,
        data: Any = None,
    ) -> torch.Tensor:
        if output.dim() == 1:
            if output.numel() != self.num_targets:
                raise RuntimeError(
                    "A one-dimensional graph_multiscalar output must contain "
                    f"exactly {self.num_targets} values. Got {_shape(output)}."
                )
            output = output.reshape(1, self.num_targets)

        if output.dim() != 2 or output.size(-1) != self.num_targets:
            raise RuntimeError(
                "graph_multiscalar expects graph-level output shape [B, K], "
                f"with K={self.num_targets}. Got {_shape(output)}."
            )

        if data is not None:
            num_graphs = self._num_graphs(data)
            if num_graphs is not None and output.size(0) != num_graphs:
                raise RuntimeError(
                    "graph_multiscalar received a tensor whose first dimension "
                    "does not match the number of graphs in the batch. This often "
                    "means an entity-level ScalarDecoder output was used instead "
                    "of MultiScalarDecoder. "
                    f"Expected B={num_graphs}, got {_shape(output)}."
                )

        return output

    def forward(self, data):
        output = self.model(data)

        if isinstance(output, Mapping):
            output = _get_output_from_mapping(output, self.output_keys)

        if torch.is_tensor(output):
            return self._validate_tensor(output, data=data)

        if isinstance(output, (list, tuple)):
            # Import lazily to keep the task module lightweight and avoid
            # imposing the legacy accumulator on direct/equivariant tasks.
            from catalyst.ml.gnn.modules.utils.predict import accumulate_predictions

            pooled, multichannel = accumulate_predictions(
                output,
                data,
                loss_tag="exact",
                return_y=False,
                channel_mode="target",
                normalize_by=self.normalize_by,
                legacy_multichannel_shape=False,
            )

            if not multichannel:
                raise RuntimeError(
                    "graph_multiscalar received a single-channel decoder output. "
                    f"Expected {self.num_targets} independent scalar channels."
                )

            return self._validate_tensor(pooled, data=data)

        raise TypeError(
            "GraphMultiScalarAdapter expected the wrapped model to return a "
            "tensor, mapping, or list/tuple of order contributions. "
            f"Got {_shape(output)}."
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
    requires_graph_multiscalar_adapter: bool = False
    squeeze_single_vector_channel: bool = True
    normalize_by: Optional[str] = "primary_nodes"
    target_names: Optional[tuple[str, ...]] = None

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
    def graph_multiscalar(
        cls,
        *,
        num_targets: int,
        target_key: str = "target_scalars",
        output_key: str = "scalar",
        accumulate_loss: str = "exact",
        normalize_by: Optional[str] = "primary_nodes",
        target_names: Optional[Iterable[str]] = None,
    ) -> "GNNTask":
        num_targets = int(num_targets)
        if num_targets < 2:
            raise ValueError(
                "graph_multiscalar requires num_targets >= 2. "
                "Use graph_scalar for a single target."
            )

        names = None if target_names is None else tuple(str(name) for name in target_names)
        if names is not None and len(names) != num_targets:
            raise ValueError(
                "target_names must contain exactly num_targets entries. "
                f"Got {len(names)} names for {num_targets} targets."
            )

        return cls(
            name="graph_multiscalar",
            target_key=target_key,
            output_type="scalar",
            output_level="graph",
            accumulate_loss=accumulate_loss,
            output_key=output_key,
            prefer_equivariant_key="scalar",
            out_dim=num_targets,
            requires_graph_multiscalar_adapter=True,
            normalize_by=normalize_by,
            target_names=names,
        )

    @classmethod
    def graph_scalar_multichannel(cls, **kwargs: Any) -> "GNNTask":
        """Alias for :meth:`graph_multiscalar`."""
        return cls.graph_multiscalar(**kwargs)

    @classmethod
    def scalar_multichannel(cls, **kwargs: Any) -> "GNNTask":
        """Backward-friendly graph-level alias for :meth:`graph_multiscalar`."""
        return cls.graph_multiscalar(**kwargs)

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
        if int(vector_channels) != 1:
            raise ValueError(
                "vector_channels > 1 is not supported by Catalyst 2.2 task validation. "
                "Use vector_channels=1 for a single geometric vector, or "
                "GNNTask.graph_multiscalar(...) for independent scalar channels."
            )
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
        if int(vector_channels) != 1:
            raise ValueError(
                "vector_channels > 1 is not supported by Catalyst 2.2 task validation. "
                "Use vector_channels=1 for a single geometric vector, or "
                "GNNTask.graph_multiscalar(...) for independent scalar channels."
            )
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
        if name in {
            "graph_multiscalar",
            "graph_multi_scalar",
            "graph_scalar_multichannel",
            "scalar_multichannel",
            "multiscalar",
        }:
            return cls.graph_multiscalar(**kwargs)
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
            "graph_scalar, graph_multiscalar, node_scalar, node_vector, "
            "graph_vector, scalar_gradient."
        )

    def with_target_key(self, target_key: str) -> "GNNTask":
        return replace(self, target_key=target_key)

    def apply_to_catalyst_parameters(self, parameters: MutableMapping[str, Any]) -> None:
        """Apply the task-owned backend contract.

        ``parameters`` may be a plain mutable mapping (legacy/internal use) or a
        :class:`catalyst.observer.Catalyst` object.  Passing a Catalyst object is
        preferred for user code because ``Catalyst.set_task`` applies the task
        atomically and performs conflict validation against explicitly supplied
        JSON/constructor parameters.
        """
        if hasattr(parameters, "set_task") and hasattr(parameters, "parameters"):
            parameters.set_task(self)
            return

        model_dict = parameters.setdefault("model_dict", {})
        model_dict["task"] = self.name
        model_dict["task_out_dim"] = self.out_dim
        model_dict["accumulate_loss"] = self.accumulate_loss

        if self.target_names is not None:
            model_dict["task_target_names"] = list(self.target_names)

        prediction_params = dict(model_dict.get("prediction_params", {}))
        prediction_params["target_key"] = self.target_key

        if self.output_key is not None:
            prediction_params["output_key"] = self.output_key

        if self.prefer_equivariant_key is not None:
            prediction_params["prefer_equivariant_key"] = self.prefer_equivariant_key

        # GNNTask describes supervised prediction semantics. Legacy list-output
        # models need channel_mode="target" so their entity/order channels are
        # accumulated into loss-ready target predictions. Direct tensor/dict
        # outputs ignore this option, so making it explicit is harmless there.
        prediction_params["channel_mode"] = "target"

        if self.name == "graph_multiscalar":
            prediction_params["legacy_multichannel_shape"] = False
            prediction_params["normalize_by"] = self.normalize_by

        model_dict["prediction_params"] = prediction_params

    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "output_type": self.output_type,
            "output_level": self.output_level,
            "out_dim": self.out_dim,
        }

    def wrap_model_if_needed(self, model: nn.Module) -> nn.Module:
        if self.requires_graph_multiscalar_adapter:
            return GraphMultiScalarAdapter(
                model,
                num_targets=self.out_dim,
                normalize_by=self.normalize_by,
                output_keys=tuple(
                    key
                    for key in (
                        self.output_key,
                        self.prefer_equivariant_key,
                        "scalar",
                        "pred",
                        "output",
                    )
                    if key is not None
                ),
            )

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

        if self.name == "graph_multiscalar":
            if pred.dim() != 2 or pred.size(-1) != self.out_dim:
                raise RuntimeError(
                    "graph_multiscalar expects prediction shape [B, K], "
                    f"with K={self.out_dim}. Got pred={_shape(pred)}."
                )

            if target.dim() == 1 and target.numel() == pred.numel():
                target = target.reshape_as(pred)

            if target.dim() != 2 or target.size(-1) != self.out_dim:
                raise RuntimeError(
                    f"graph_multiscalar expects target {self.target_key!r} shape "
                    f"[B, K], with K={self.out_dim}. Got target={_shape(target)}."
                )

            if pred.shape != target.shape:
                raise RuntimeError(
                    "graph_multiscalar prediction/target shape mismatch: "
                    f"pred={_shape(pred)}, target={_shape(target)}."
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


def _validate_task_model_request(
    task: GNNTask,
    *,
    model_type: Optional[str],
    preset: Optional[str],
    decoder: Optional[nn.Module],
    apply_task_model_kwargs: bool,
    kwargs: Mapping[str, Any],
) -> None:
    """Validate task/model-builder compatibility before model construction."""
    if not isinstance(task, GNNTask):
        raise TypeError("build_task_model(task=...) requires a GNNTask instance.")

    if apply_task_model_kwargs:
        expected = task.model_kwargs()
        for key, required in expected.items():
            if key in kwargs and kwargs[key] != required:
                raise ValueError(
                    f"Task {task.name!r} requires {key}={required!r}, but "
                    f"build_task_model received {key}={kwargs[key]!r}. "
                    "Task-controlled model settings must not be overridden."
                )

    route = str(preset or model_type or "").lower().strip()
    equivariant_route = (
        route in {"equivariant", "egnn", "equivariant_gnn"}
        or route.startswith("equivariant_")
        or str(kwargs.get("processor_type", "")).lower().strip() == "equivariant"
        or "equivariant" in str(kwargs.get("encoder_type", "")).lower().strip()
        or str(kwargs.get("decoder_type", "")).lower().strip()
        in {"equivariant", "equivariant_decoder", "generic_equivariant"}
    )

    if task.name in {"node_vector", "graph_vector", "scalar_gradient"}:
        if not equivariant_route and decoder is None:
            raise ValueError(
                f"Task {task.name!r} requires an equivariant/vector-capable model route "
                "or an explicitly supplied compatible custom decoder."
            )

    if task.name == "graph_multiscalar" and apply_task_model_kwargs and decoder is None:
        decoder_type = kwargs.get("decoder_type")
        if decoder_type is not None and not equivariant_route:
            if str(decoder_type).lower().strip() not in {"multiscalar", "multi_scalar"}:
                raise ValueError(
                    "graph_multiscalar requires decoder_type='multiscalar' for the "
                    "standard non-equivariant route when no custom decoder is supplied."
                )



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
    _validate_task_model_request(
        task,
        model_type=model_type,
        preset=preset,
        decoder=decoder,
        apply_task_model_kwargs=apply_task_model_kwargs,
        kwargs=kwargs,
    )

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

        if task.name == "graph_multiscalar" and decoder is None:
            route = str(preset or model_type or "").lower().strip()
            equivariant_route = (
                route in {"equivariant", "egnn", "equivariant_gnn"}
                or route.startswith("equivariant_")
                or str(kwargs.get("processor_type", "")).lower().strip() == "equivariant"
                or "equivariant"
                in str(kwargs.get("encoder_type", "")).lower().strip()
                or str(kwargs.get("decoder_type", "")).lower().strip()
                in {"equivariant", "equivariant_decoder", "generic_equivariant"}
            )
            if not equivariant_route:
                model_kwargs.setdefault("decoder_type", "multiscalar")

    if decoder is not None:
        model_kwargs["decoder"] = decoder

    # Caller kwargs have final precedence, except for task-control keywords that
    # must never leak to build_model(...).
    kwargs = dict(kwargs)
    kwargs.pop("apply_task_model_kwargs", None)
    model_kwargs.update(kwargs)

    model = build_model(**model_kwargs)
    wrapped = task.wrap_model_if_needed(model)
    # Metadata is deliberately attached to the returned module so Catalyst can
    # validate task/model compatibility later even when task/model construction
    # occurs before the Catalyst object exists.
    setattr(wrapped, "_catalyst_task", task)
    setattr(wrapped, "_catalyst_model_kwargs", dict(model_kwargs))
    return wrapped


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
    task_out_dim = int(model_dict.get("task_out_dim", 1))
    task_target_names = model_dict.get("task_target_names", None)

    if task_name is not None:
        kwargs: Dict[str, Any] = {}
        if target_key is not None:
            kwargs["target_key"] = target_key
        if output_key is not None:
            kwargs["output_key"] = output_key
        if accumulate_loss is not None:
            kwargs["accumulate_loss"] = accumulate_loss
        if str(task_name).lower().strip() in {
            "graph_multiscalar",
            "graph_multi_scalar",
            "graph_scalar_multichannel",
            "scalar_multichannel",
            "multiscalar",
        }:
            kwargs["num_targets"] = task_out_dim
            if task_target_names is not None:
                kwargs["target_names"] = task_target_names
            kwargs["normalize_by"] = prediction_params.get(
                "normalize_by", "primary_nodes"
            )
        elif str(task_name).lower().strip() in {"graph_scalar", "node_scalar"}:
            kwargs["out_dim"] = task_out_dim
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
    "GraphMultiScalarAdapter",
    "build_task_model",
    "validate_task_batch",
    "task_from_parameters",
]
