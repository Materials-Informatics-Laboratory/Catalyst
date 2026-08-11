import torch
from torch import nn


# =============================================================================
# Public API
# =============================================================================


def accumulate_predictions(
    pred,
    data,
    loss_tag,
    return_y=True,
    channel_mode="target",
    normalize_by="primary_nodes",
    legacy_multichannel_shape=False,
    output_key=None,
    target_key=None,
    target_map=None,
    prefer_equivariant_key=None,
):
    """
    Aggregate model predictions into loss-ready tensors.

    This function supports both the legacy Catalyst outputs and the newer
    generic/equivariant GNN outputs.

    Legacy outputs
    --------------
    The old branch expects ``pred`` to be a list/tuple of entity predictions:

        Atomic graph:
            pred[0] -> atom/entity-G channels
            pred[1] -> bond/entity-A channels
            pred[2] -> angle/edge-A channels, optional

        Generic graph:
            pred[0] -> node_G channels
            pred[1] -> node_A channels
            pred[2] -> edge_A channels, optional

    New generic/equivariant outputs
    -------------------------------
    The new decoder can return either a tensor directly or a dictionary:

        {"scalar": scalar}
        {"vector": vector}
        {"scalar": scalar, "gradient": gradient}

    For direct tensor outputs, this function treats the tensor as already
    decoded and only finds the matching target.

    New useful loss_tag values
    --------------------------
    "node", "raw", "direct", "none":
        Do not graph-aggregate.  Return the decoded prediction and matching
        target directly.  This is the correct mode for per-atom vector targets,
        for example learning the complete force vector per atom.

    "exact":
        For legacy list outputs, graph-aggregate by sample id.
        For new dict/tensor outputs, return direct predictions unless the output
        is explicitly graph-level already.

    "sum":
        For legacy list outputs, sum across the whole batch.
        For new scalar outputs, sum the scalar output and target.

    Parameters
    ----------
    output_key
        Optional key to select from dict outputs, for example "vector",
        "scalar", or "gradient".

    target_key
        Optional explicit data attribute to use as target.

    target_map
        Optional mapping from prediction key to target attribute name, e.g.
        {"vector": "target_vector", "scalar": "target_scalar"}.

    prefer_equivariant_key
        Optional fallback preference for dict outputs.  Same idea as output_key,
        but lower priority.

    Returns
    -------
    If return_y:
        preds, y, vec
    Else:
        preds, vec

    The third return value ``vec`` is kept for compatibility with the existing
    high-level GNN training loop.  It is True for vector-like or multichannel
    predictions.
    """

    # ------------------------------------------------------------------
    # New dict outputs from EquivariantDecoder(return_dict=True).
    # ------------------------------------------------------------------
    if isinstance(pred, dict):
        return _accumulate_dict_prediction(
            pred=pred,
            data=data,
            loss_tag=loss_tag,
            return_y=return_y,
            output_key=output_key,
            target_key=target_key,
            target_map=target_map,
            prefer_equivariant_key=prefer_equivariant_key,
        )

    # ------------------------------------------------------------------
    # New direct tensor outputs from EquivariantDecoder(return_dict=False).
    # Also useful for simple scalar/vector models.
    # ------------------------------------------------------------------
    if torch.is_tensor(pred):
        return _accumulate_direct_tensor_prediction(
            pred=pred,
            data=data,
            loss_tag=loss_tag,
            return_y=return_y,
            target_key=target_key,
            prefer_equivariant_key=prefer_equivariant_key,
        )

    # ------------------------------------------------------------------
    # Tuple output, mainly scalar_gradient with return_dict=False:
    #     (scalar, gradient)
    #
    # To stay compatible with the current high-level GNN.train implementation,
    # select one tensor by default.  Use output_key="scalar" or "gradient".
    # ------------------------------------------------------------------
    if isinstance(pred, tuple):
        pred = _tuple_to_prediction_dict(pred)
        return _accumulate_dict_prediction(
            pred=pred,
            data=data,
            loss_tag=loss_tag,
            return_y=return_y,
            output_key=output_key,
            target_key=target_key,
            target_map=target_map,
            prefer_equivariant_key=prefer_equivariant_key,
        )

    # ------------------------------------------------------------------
    # Legacy list output path.
    # ------------------------------------------------------------------
    if loss_tag == "ensemble":
        raise NotImplementedError(
            "loss_tag='ensemble' was a no-op in the original function. "
            "Add ensemble behavior here if needed."
        )

    if loss_tag not in {"exact", "sum"}:
        raise ValueError(
            f"Unknown loss_tag={loss_tag!r} for legacy list outputs. "
            "Supported legacy tags are 'exact' and 'sum'. "
            "For direct/equivariant tensor outputs, use 'node', 'raw', 'direct', "
            "'none', 'exact', or 'sum'."
        )

    if channel_mode not in {"target", "latent"}:
        raise ValueError(
            f"Unknown channel_mode={channel_mode}. "
            "Use 'target' or 'latent'."
        )

    n_channels = _get_num_channels(pred)
    schema = _get_schema(data)

    if channel_mode == "target":
        preds = _accumulate_as_target(
            pred=pred,
            data=data,
            schema=schema,
            n_channels=n_channels,
            loss_tag=loss_tag,
            normalize_by=normalize_by,
            legacy_multichannel_shape=legacy_multichannel_shape,
        )

        multichannel = n_channels > 1
        y = _get_y(data, loss_tag, multichannel, target_dim=n_channels) if return_y else None

        if return_y:
            return preds, y, multichannel

        return preds, multichannel

    if channel_mode == "latent":
        features, metadata = _accumulate_as_latent_features(
            pred=pred,
            data=data,
            schema=schema,
            n_channels=n_channels,
            loss_tag=loss_tag,
            normalize_by=normalize_by,
        )

        y = _get_y(data, loss_tag, multichannel=False, target_dim=None) if return_y else None

        if return_y:
            return features, y, metadata

        return features, metadata


# =============================================================================
# New generic/equivariant prediction handling
# =============================================================================


_DIRECT_LOSS_TAGS = {"node", "raw", "direct", "none", "exact"}


def _tuple_to_prediction_dict(pred):
    """
    Convert tuple outputs to a dict.

    The current EquivariantDecoder(return_dict=False) returns:

        scalar_gradient -> (scalar, gradient)
    """
    if len(pred) == 2:
        return {"scalar": pred[0], "gradient": pred[1]}

    return {f"output_{i}": value for i, value in enumerate(pred)}


def _accumulate_dict_prediction(
    pred,
    data,
    loss_tag,
    return_y=True,
    output_key=None,
    target_key=None,
    target_map=None,
    prefer_equivariant_key=None,
):
    """
    Handle dict outputs from the new generic/equivariant decoders.
    """
    key = _select_prediction_key(
        pred,
        data=data,
        output_key=output_key,
        target_key=target_key,
        target_map=target_map,
        prefer_equivariant_key=prefer_equivariant_key,
    )

    value = pred[key]
    target_attr = _target_key_for_prediction_key(
        key,
        data=data,
        target_key=target_key,
        target_map=target_map,
    )

    preds = _format_direct_prediction(value, loss_tag=loss_tag, key=key)
    vec = _is_vector_like_prediction(preds, key=key)

    if return_y:
        y = _get_direct_target(
            data,
            target_attr=target_attr,
            pred=preds,
            loss_tag=loss_tag,
            key=key,
        )
        return preds, y, vec

    return preds, vec


def _accumulate_direct_tensor_prediction(
    pred,
    data,
    loss_tag,
    return_y=True,
    target_key=None,
    prefer_equivariant_key=None,
):
    """
    Handle direct tensor outputs.

    This is the path used when EquivariantDecoder(return_dict=False) returns a
    raw vector/scalar tensor.
    """
    key = prefer_equivariant_key or _infer_key_from_tensor_and_data(pred, data, target_key=target_key)

    preds = _format_direct_prediction(pred, loss_tag=loss_tag, key=key)
    vec = _is_vector_like_prediction(preds, key=key)

    if return_y:
        target_attr = target_key or _target_key_for_prediction_key(key, data=data)
        y = _get_direct_target(
            data,
            target_attr=target_attr,
            pred=preds,
            loss_tag=loss_tag,
            key=key,
        )
        return preds, y, vec

    return preds, vec


def _select_prediction_key(
    pred,
    data,
    output_key=None,
    target_key=None,
    target_map=None,
    prefer_equivariant_key=None,
):
    """
    Select which dict output to use for the current loss.

    Priority:
        1. explicit output_key
        2. prefer_equivariant_key
        3. target_key / target_map compatibility
        4. common single-target cases
        5. single-key dictionary
    """
    if output_key is not None:
        if output_key not in pred:
            raise KeyError(f"Requested output_key={output_key!r}, but pred has keys {list(pred)}.")
        return output_key

    if prefer_equivariant_key is not None and prefer_equivariant_key in pred:
        return prefer_equivariant_key

    if target_map:
        for key, mapped_target in target_map.items():
            if key in pred and hasattr(data, mapped_target):
                return key

    if target_key is not None:
        # Match explicit target_key to conventional prediction key.
        reverse = {
            "target_scalar": "scalar",
            "y": "scalar",
            "target_vector": "vector",
            "forces": "vector",
            "force": "vector",
            "target_gradient": "gradient",
            "gradient": "gradient",
        }
        candidate = reverse.get(target_key)
        if candidate in pred:
            return candidate

    # Prefer vector if vector target exists. This is the important path for
    # per-atom force-vector learning with {"vector": ...}.
    if "vector" in pred and _has_any_attr(data, ("target_vector", "forces", "force", "y")):
        return "vector"

    # Prefer gradient if the graph has an explicit gradient target.
    if "gradient" in pred and _has_any_attr(data, ("target_gradient",)):
        return "gradient"

    # If scalar and target_scalar/y exist, use scalar.
    if "scalar" in pred and _has_any_attr(data, ("target_scalar", "y")):
        return "scalar"

    if len(pred) == 1:
        return next(iter(pred.keys()))

    # Scalar-gradient models often return both scalar and gradient. If there is
    # no explicit request, prefer gradient when target_vector exists because in
    # atomistic examples target_vector usually stores force-like labels.
    if "gradient" in pred and _has_any_attr(data, ("target_vector", "forces", "force")):
        return "gradient"

    raise ValueError(
        "Could not infer which prediction key to use. "
        f"Prediction keys: {list(pred)}. "
        "Pass output_key='scalar', 'vector', or 'gradient', or pass target_key=..."
    )


def _target_key_for_prediction_key(key, data, target_key=None, target_map=None):
    """
    Resolve the data target attribute for a prediction key.
    """
    if target_key is not None:
        return target_key

    if target_map and key in target_map:
        return target_map[key]

    candidates = {
        "scalar": ("target_scalar", "y"),
        "vector": ("target_vector", "forces", "force", "y"),
        "gradient": ("target_gradient", "target_vector", "forces", "force", "y"),
    }.get(key, ("y",))

    for attr in candidates:
        if hasattr(data, attr):
            return attr

    # Return the first conventional candidate so the error message is useful.
    return candidates[0]


def _infer_key_from_tensor_and_data(pred, data, target_key=None):
    """
    Infer semantic key for a direct tensor output.
    """
    if target_key is not None:
        if target_key in {"target_vector", "forces", "force"}:
            return "vector"
        if target_key in {"target_gradient", "gradient"}:
            return "gradient"
        if target_key in {"target_scalar", "y"}:
            # y may be scalar or vector; fall through if shape suggests vector.
            pass

    if pred.dim() >= 2 and pred.shape[-1] == 3:
        if _has_any_attr(data, ("target_vector", "forces", "force")):
            return "vector"
        if _has_any_attr(data, ("target_gradient",)):
            return "gradient"

    if _has_any_attr(data, ("target_scalar",)):
        return "scalar"

    if _has_any_attr(data, ("target_vector", "forces", "force")):
        return "vector"

    return "scalar"


def _format_direct_prediction(pred, loss_tag, key):
    """
    Format decoded scalar/vector prediction.

    For node/raw/direct/exact, return as-is except for light scalar cleanup.
    For sum, sum over batch/sample dimensions.
    """
    if not torch.is_tensor(pred):
        raise TypeError(f"Prediction for key={key!r} must be a tensor, got {type(pred)}.")

    if loss_tag in _DIRECT_LOSS_TAGS:
        return pred

    if loss_tag == "sum":
        if pred.dim() == 0:
            return pred
        return pred.sum(dim=0)

    raise ValueError(
        f"Unknown loss_tag={loss_tag!r} for direct/dict prediction. "
        "Supported: node, raw, direct, none, exact, sum."
    )


def _coerce_target_tensor(value):
    """Convert scalar/list/tuple target data to a tensor safely."""
    if torch.is_tensor(value):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.as_tensor(value)

        converted = [_coerce_target_tensor(item) for item in value]
        try:
            return torch.stack(converted)
        except RuntimeError:
            # Preserve PyTorch's normal failure for genuinely ragged targets.
            return torch.as_tensor(value)

    return torch.as_tensor(value)


def _get_direct_target(data, target_attr, pred, loss_tag, key):
    """
    Return target tensor matched to a direct/dict prediction.
    """
    if not hasattr(data, target_attr):
        raise AttributeError(
            f"Could not find target attribute data.{target_attr}. "
            f"Prediction key was {key!r}. Available data attributes include: "
            f"{_safe_data_attr_preview(data)}"
        )

    y = _coerce_target_tensor(getattr(data, target_attr))

    y = y.to(device=pred.device, dtype=pred.dtype)

    if loss_tag in _DIRECT_LOSS_TAGS:
        # Try to reshape only when the number of elements is compatible. This
        # keeps force targets [N, 3] intact and scalar graph targets flexible.
        if y.shape != pred.shape and y.numel() == pred.numel():
            y = y.reshape_as(pred)
        return y

    if loss_tag == "sum":
        if y.dim() == 0:
            return y
        y = y.sum(dim=0)
        if y.shape != pred.shape and y.numel() == pred.numel():
            y = y.reshape_as(pred)
        return y

    raise ValueError(f"Unknown loss_tag: {loss_tag}")


def _is_vector_like_prediction(pred, key=None):
    """
    Compatibility flag for the high-level GNN loop.

    The existing trainer uses this flag to decide whether to apply the loss to
    rows/components separately.  Returning True for vector/gradient keeps the
    old behavior.  If you want one loss over the whole tensor, pass a custom
    loss_fn to GNN.train.
    """
    if key in {"vector", "gradient"}:
        return True

    if torch.is_tensor(pred) and pred.dim() >= 2 and pred.shape[-1] > 1:
        return True

    return False


def _has_any_attr(data, names):
    return any(hasattr(data, name) for name in names)


def _safe_data_attr_preview(data, max_items=40):
    try:
        keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(vars(data).keys())
    except (AttributeError, TypeError, ValueError):
        keys = list(vars(data).keys()) if hasattr(data, "__dict__") else []
    return keys[:max_items]


# =============================================================================
# Legacy accumulation path
# =============================================================================


def _get_num_channels(pred):
    """
    Infer number of channels per entity.
    """
    p0 = pred[0]

    if p0.dim() == 1:
        return 1

    return int(p0.shape[-1])


def _get_schema(data):
    """
    Determine graph convention and associated batch attributes.

    Returns
    -------
    schema : list[dict]
        Each dict contains:
            pred_index
            batch_attr
            name
            is_primary
    """

    # Generic graph convention
    if hasattr(data, "node_G_batch"):
        schema = [
            {
                "pred_index": 0,
                "batch_attr": "node_G_batch",
                "name": "g_node",
                "is_primary": True,
            },
            {
                "pred_index": 1,
                "batch_attr": "node_A_batch",
                "name": "a_node",
                "is_primary": False,
            },
        ]

        if hasattr(data, "edge_A_batch"):
            schema.append(
                {
                    "pred_index": 2,
                    "batch_attr": "edge_A_batch",
                    "name": "a_edge",
                    "is_primary": False,
                }
            )

        return schema

    # Atomic graph convention
    if hasattr(data, "x_atm_batch"):
        schema = [
            {
                "pred_index": 0,
                "batch_attr": "x_atm_batch",
                "name": "atom",
                "is_primary": True,
            },
            {
                "pred_index": 1,
                "batch_attr": "x_bnd_batch",
                "name": "bond",
                "is_primary": False,
            },
        ]

        if hasattr(data, "x_ang_batch"):
            schema.append(
                {
                    "pred_index": 2,
                    "batch_attr": "x_ang_batch",
                    "name": "angle",
                    "is_primary": False,
                }
            )

        return schema

    # PyG/equivariant convention for legacy-style single tensor predictions.
    # This schema is intentionally not used for direct tensor outputs above.
    if hasattr(data, "batch"):
        return [
            {
                "pred_index": 0,
                "batch_attr": "batch",
                "name": "node",
                "is_primary": True,
            }
        ]

    raise AttributeError(
        "Could not determine graph type. Expected either atomic batch "
        "attributes x_atm_batch/x_bnd_batch, generic batch attributes "
        "node_G_batch/node_A_batch, or PyG batch."
    )


def _reshape_prediction(p, n_channels):
    """
    Convert prediction tensor to [N_entity, K].
    """
    if p.dim() == 1:
        return p.reshape(-1, 1)

    return p.reshape(-1, n_channels)


def _num_graphs_from_schema(data, schema):
    """Return one shared graph count for every entity type in a batch."""
    explicit = getattr(data, "num_graphs", None)
    if explicit is not None:
        return int(explicit)
    ptr = getattr(data, "ptr", None)
    if torch.is_tensor(ptr):
        return int(ptr.numel() - 1)

    primary_entries = [entry for entry in schema if entry["is_primary"]]
    if len(primary_entries) != 1:
        raise RuntimeError("Expected exactly one primary node entry in schema.")

    primary_batch = data[primary_entries[0]["batch_attr"]]
    if primary_batch.numel() == 0:
        return 0

    return int(primary_batch.max().item()) + 1


def _sum_by_batch(values, batch, num_graphs=None):
    """Sum values by their original graph/sample ids.

    Passing one shared ``num_graphs`` keeps rows aligned across primary and
    secondary entity tensors, including graphs that contain no bonds/angles.
    """
    batch = batch.to(device=values.device, dtype=torch.long).reshape(-1)

    if values.shape[0] != batch.numel():
        raise ValueError(
            "values and batch must describe the same number of entities. "
            f"Got {values.shape[0]} values and {batch.numel()} batch ids."
        )

    if num_graphs is None:
        num_graphs = 0 if batch.numel() == 0 else int(batch.max().item()) + 1
    num_graphs = int(num_graphs)

    if num_graphs < 0:
        raise ValueError(f"num_graphs must be non-negative, got {num_graphs}.")

    if batch.numel() > 0:
        # Tensor assertions avoid extracting CUDA scalars with .item() in every
        # forward pass and remain friendly to torch.compile.
        torch._assert(torch.all(batch >= 0), "batch contains a negative graph id.")
        torch._assert(
            torch.all(batch < num_graphs),
            "batch contains a graph id outside the requested output range.",
        )

    out = values.new_zeros((num_graphs, values.shape[1]))
    if batch.numel() > 0:
        out.index_add_(0, batch, values)

    return out

def _primary_node_counts(data, schema, loss_tag):
    """
    Number of primary nodes per graph.

    For atomic graphs, primary nodes are atoms.
    For generic graphs, primary nodes are node_G.
    For PyG/equivariant graphs, primary nodes are data.batch nodes.
    """

    primary_entries = [entry for entry in schema if entry["is_primary"]]

    if len(primary_entries) != 1:
        raise RuntimeError("Expected exactly one primary node entry in schema.")

    primary_batch = data[primary_entries[0]["batch_attr"]]

    ones = torch.ones(
        primary_batch.shape[0],
        dtype=torch.float,
        device=primary_batch.device,
    ).reshape(-1, 1)

    if loss_tag == "exact":
        num_graphs = _num_graphs_from_schema(data, schema)
        return _sum_by_batch(
            ones,
            primary_batch,
            num_graphs=num_graphs,
        ).clamp_min(1.0)

    if loss_tag == "sum":
        return ones.sum().clamp_min(1.0)

    raise ValueError(f"Unknown loss_tag: {loss_tag}")


def _apply_size_normalization(aggregated, data, schema, loss_tag, normalize_by):
    """
    Normalize graph-level sums to reduce system-size dependence.
    """

    if normalize_by is None:
        return aggregated

    if normalize_by != "primary_nodes":
        raise ValueError(
            f"Unknown normalize_by={normalize_by}. "
            "Use None or 'primary_nodes'."
        )

    counts = _primary_node_counts(data, schema, loss_tag)

    if loss_tag == "exact":
        # aggregated shape: [num_graphs, K] or [num_graphs, D]
        return aggregated / counts.to(device=aggregated.device)

    if loss_tag == "sum":
        # aggregated shape: [K] or [D]
        return aggregated / counts.to(device=aggregated.device)

    raise ValueError(f"Unknown loss_tag: {loss_tag}")


def _accumulate_as_target(
    pred,
    data,
    schema,
    n_channels,
    loss_tag,
    normalize_by,
    legacy_multichannel_shape,
):
    """
    Original interpretation:
        K channels = K output targets.

    This is what you want for true vector-valued graph-level prediction.
    """

    if loss_tag == "exact":
        total = None
        num_graphs = _num_graphs_from_schema(data, schema)

        for entry in schema:
            pred_index = entry["pred_index"]
            batch_attr = entry["batch_attr"]

            values = _reshape_prediction(pred[pred_index], n_channels)
            batch = data[batch_attr]

            summed = _sum_by_batch(
                values,
                batch,
                num_graphs=num_graphs,
            )

            if total is None:
                total = summed
            else:
                total = total + summed

        total = _apply_size_normalization(
            total,
            data=data,
            schema=schema,
            loss_tag=loss_tag,
            normalize_by=normalize_by,
        )

        if n_channels > 1:
            if legacy_multichannel_shape:
                return total.transpose(0, 1).contiguous()
            return total

        return total.squeeze(-1)

    if loss_tag == "sum":
        total = None

        for p in pred:
            values = _reshape_prediction(p, n_channels)
            summed = values.sum(dim=0)

            if total is None:
                total = summed
            else:
                total = total + summed

        total = _apply_size_normalization(
            total,
            data=data,
            schema=schema,
            loss_tag=loss_tag,
            normalize_by=normalize_by,
        )

        if n_channels > 1:
            return total

        return total.squeeze(-1)

    raise ValueError(f"Unknown loss_tag: {loss_tag}")


def _accumulate_as_latent_features(
    pred,
    data,
    schema,
    n_channels,
    loss_tag,
    normalize_by,
):
    """
    New interpretation:
        K channels = K positive latent feature channels.

    Instead of summing entity types into one K-dimensional target vector,
    this keeps entity types separate and concatenates them:

        z = [z_g_node, z_a_node, z_a_edge]

    or:

        z = [z_atom, z_bond, z_angle]
    """

    pooled_by_type = []
    entity_names = []

    if loss_tag == "exact":
        num_graphs = _num_graphs_from_schema(data, schema)
        for entry in schema:
            pred_index = entry["pred_index"]
            batch_attr = entry["batch_attr"]
            name = entry["name"]

            values = _reshape_prediction(pred[pred_index], n_channels)
            batch = data[batch_attr]

            summed = _sum_by_batch(
                values,
                batch,
                num_graphs=num_graphs,
            )

            summed = _apply_size_normalization(
                summed,
                data=data,
                schema=schema,
                loss_tag=loss_tag,
                normalize_by=normalize_by,
            )

            pooled_by_type.append(summed)
            entity_names.append(name)

        features = torch.cat(pooled_by_type, dim=-1)

    elif loss_tag == "sum":
        for entry in schema:
            pred_index = entry["pred_index"]
            name = entry["name"]

            values = _reshape_prediction(pred[pred_index], n_channels)
            summed = values.sum(dim=0)

            summed = _apply_size_normalization(
                summed,
                data=data,
                schema=schema,
                loss_tag=loss_tag,
                normalize_by=normalize_by,
            )

            pooled_by_type.append(summed)
            entity_names.append(name)

        features = torch.cat(pooled_by_type, dim=-1)

    else:
        raise ValueError(f"Unknown loss_tag: {loss_tag}")

    metadata = {
        "channel_mode": "latent",
        "n_feature_channels_per_entity": n_channels,
        "entity_names": entity_names,
        "feature_dim": features.shape[-1],
        "normalize_by": normalize_by,
        "loss_tag": loss_tag,
    }

    return features, metadata


def _get_y(data, loss_tag, multichannel, target_dim=None):
    """
    Flexible legacy target handling.

    For scalar targets:
        exact -> [num_graphs] or [num_graphs, 1]
        sum   -> scalar

    For vector targets:
        exact -> [num_graphs, target_dim]
        sum   -> [target_dim]
    """

    if not hasattr(data, "y"):
        return None

    y = _coerce_target_tensor(data.y)

    # Make vector targets easier to compare to predictions.
    if target_dim is not None and target_dim > 1:
        y = y.reshape(-1, target_dim)

        if loss_tag == "sum":
            return y.sum(dim=0)

        return y

    # Scalar target
    if loss_tag == "sum":
        return y.flatten().sum()

    return y.flatten()
