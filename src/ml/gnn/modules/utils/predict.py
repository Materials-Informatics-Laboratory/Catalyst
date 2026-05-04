import torch
from torch import nn

def accumulate_predictions(
    pred,
    data,
    loss_tag,
    return_y=True,
    channel_mode="target",
    normalize_by="primary_nodes",
    legacy_multichannel_shape=False,
):
    """
    Aggregate per-entity predictions/features into graph-level quantities.

    Parameters
    ----------
    pred : list[torch.Tensor]
        Model outputs. Expected order:

        Atomic graph:
            pred[0] -> atom/entity-G channels
            pred[1] -> bond/entity-A channels
            pred[2] -> angle/edge-A channels, optional

        Generic graph:
            pred[0] -> node_G channels
            pred[1] -> node_A channels
            pred[2] -> edge_A channels, optional

        Each tensor may have shape:
            [N_entity]        interpreted as [N_entity, 1]
            [N_entity, K]     K scalar channels per entity

    data : torch_geometric.data.Data-like object
        Must contain the relevant batch vectors.

    loss_tag : str
        "exact" or "sum".

        "exact":
            Aggregate each graph/sample in a batch separately.

        "sum":
            Aggregate globally across the whole batch.

    return_y : bool
        Whether to return data.y.

    channel_mode : str
        "target" or "latent".

        "target":
            K is interpreted as the true output dimension.
            This is the original behavior: if K > 1, the prediction is a
            vector-valued target.

        "latent":
            K is interpreted as the number of positive learned feature
            channels. The returned value is a pooled feature vector, not the
            final property y. A separate readout should map it to scalar/vector y.

    normalize_by : str or None
        Controls size normalization.

        None:
            Raw sums. This matches the original behavior most closely, but
            can scale with system size.

        "primary_nodes":
            Divide every entity-type sum by the number of primary graph nodes.
            For Atomic_Graph_Data this uses x_atm_batch.
            For Generic_Graph_Data this uses node_G_batch.

            This is recommended for intensive quantities such as energy per atom
            or enthalpy of mixing per atom.

    legacy_multichannel_shape : bool
        If True and channel_mode == "target", return multichannel exact outputs
        as [K, num_graphs], matching your old multichannel branch.

        If False, return [num_graphs, K], which is usually easier for PyTorch loss
        functions.

    Returns
    -------
    If channel_mode == "target":
        If return_y:
            preds, y, multichannel
        Else:
            preds, multichannel

    If channel_mode == "latent":
        If return_y:
            features, y, metadata
        Else:
            features, metadata

        where features has shape:
            exact: [num_graphs, num_entity_types * K]
            sum:   [num_entity_types * K]
    """

    if loss_tag == "ensemble":
        raise NotImplementedError(
            "loss_tag='ensemble' was a no-op in the original function. "
            "Add ensemble behavior here if needed."
        )

    if loss_tag not in {"exact", "sum"}:
        raise ValueError(f"Unknown loss_tag: {loss_tag}")

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

    raise AttributeError(
        "Could not determine graph type. Expected either atomic batch "
        "attributes x_atm_batch/x_bnd_batch or generic batch attributes "
        "node_G_batch/node_A_batch."
    )


def _reshape_prediction(p, n_channels):
    """
    Convert prediction tensor to [N_entity, K].
    """
    if p.dim() == 1:
        return p.reshape(-1, 1)

    return p.reshape(-1, n_channels)


def _sum_by_batch(values, batch):
    """
    Sum values by graph/sample id.

    values: [N_entity, K]
    batch:  [N_entity]

    returns:
        [num_graphs, K]
    """

    batch = batch.to(device=values.device)

    unique_batch, inverse = torch.unique(
        batch,
        sorted=True,
        return_inverse=True,
    )

    out = values.new_zeros((unique_batch.numel(), values.shape[1]))
    out.index_add_(0, inverse, values)

    return out


def _primary_node_counts(data, schema, loss_tag):
    """
    Number of primary nodes per graph.

    For atomic graphs, primary nodes are atoms.
    For generic graphs, primary nodes are node_G.
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
        return _sum_by_batch(ones, primary_batch).clamp_min(1.0)

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

    This is what you want for true vector-valued prediction.
    """

    if loss_tag == "exact":
        total = None

        for entry in schema:
            pred_index = entry["pred_index"]
            batch_attr = entry["batch_attr"]

            values = _reshape_prediction(pred[pred_index], n_channels)
            batch = data[batch_attr]

            summed = _sum_by_batch(values, batch)

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

    If K=16 and there are two entity types, output dimension is 32.
    If K=16 and there are three entity types, output dimension is 48.
    """

    pooled_by_type = []
    entity_names = []

    if loss_tag == "exact":
        for entry in schema:
            pred_index = entry["pred_index"]
            batch_attr = entry["batch_attr"]
            name = entry["name"]

            values = _reshape_prediction(pred[pred_index], n_channels)
            batch = data[batch_attr]

            summed = _sum_by_batch(values, batch)

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
    Flexible target handling.

    For scalar targets:
        exact -> [num_graphs] or [num_graphs, 1]
        sum   -> scalar

    For vector targets:
        exact -> [num_graphs, target_dim]
        sum   -> [target_dim]
    """

    if not hasattr(data, "y"):
        return None

    y = data.y

    if isinstance(y, (list, tuple)):
        y = torch.stack(tuple(y_i for y_i in y))

    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

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