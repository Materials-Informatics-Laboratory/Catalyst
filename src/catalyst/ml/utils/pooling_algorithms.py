"""Extended scatter/pooling utilities used by Catalyst models.

Catalyst previously imported :mod:`torch_scatter` directly. Modern PyTorch
Geometric exposes a compatible ``scatter`` helper and can use native PyTorch
reductions when the optional compiled ``torch-scatter`` extension is absent.
Using the PyG interface keeps Catalyst installable with a standard ``pip``
workflow while preserving the same pooling API.
"""

import torch
from torch_geometric.utils import scatter as pyg_scatter


def _scatter(src, index, *, dim=-1, dim_size=None, reduce="sum"):
    """Thin wrapper around :func:`torch_geometric.utils.scatter`."""
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def scatter_(src, index, dim=-1, reduce="mean", dim_size=None, eps=1e-9, weights=None):
    """
    Extended scatter function with the same core signature as
    :func:`torch_geometric.utils.scatter`, plus additional reduction modes.

    Args:
        src (Tensor): Source tensor.
        index (LongTensor): Indices for grouping.
        dim (int, optional): Dimension along which to scatter. (default: -1)
        reduce (str, optional): Reduction name.
            Native:
                'sum'/'add', 'mean', 'min', 'max'
            Extended:
                'count'
                'variance'/'var'
                'std'
                'range'
                'softmax'
                'normalized'      (z-score per group)
                'sum_of_squares'
                'mean_abs'
                'l1'
                'l2'
                'geometric_mean'
                'harmonic_mean'
                'midrange'
                'amplitude'
                'logsumexp'
                'softmin'
                'attention'       (requires weights argument)
                'normalized_sum'  (sum / L2 norm of group)
                'any'
                'all'
                'xor'
        dim_size (int, optional): Output size along dim.
        eps (float): Stability constant for divisions and logs. (default: 1e-9)
        weights (Tensor, optional): Weights for 'attention' pooling.

    Returns:
        Tensor
    """

    if reduce == "add":
        reduce = "sum"
    if reduce in ["var", "variance"]:
        reduce = "variance"

    # Native reductions.
    if reduce in ["sum", "mean", "min", "max"]:
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)

    if reduce == "count":
        ones = torch.ones_like(src)
        return _scatter(ones, index, dim=dim, dim_size=dim_size, reduce="sum")

    if reduce == "variance":
        mean = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
        mean_sq = _scatter(src**2, index, dim=dim, dim_size=dim_size, reduce="mean")
        return mean_sq - mean**2

    if reduce == "std":
        var = scatter_(src, index, dim=dim, reduce="variance", dim_size=dim_size, eps=eps)
        return torch.sqrt(var + eps)

    if reduce == "range":
        max_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")
        min_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="min")
        return max_vals - min_vals

    if reduce == "softmax":
        max_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")
        max_per_elem = max_vals.index_select(dim, index)
        src_exp = torch.exp(src - max_per_elem)
        denom = _scatter(src_exp, index, dim=dim, dim_size=dim_size, reduce="sum")
        denom_per_elem = denom.index_select(dim, index)
        weights_ = src_exp / (denom_per_elem + eps)
        return _scatter(src * weights_, index, dim=dim, dim_size=dim_size, reduce="sum")

    if reduce == "normalized":
        mean = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
        std = scatter_(src, index, dim=dim, reduce="std", dim_size=dim_size, eps=eps)
        return (src - mean.index_select(dim, index)) / (std.index_select(dim, index) + eps)

    if reduce == "sum_of_squares":
        return _scatter(src**2, index, dim=dim, dim_size=dim_size, reduce="sum")

    if reduce == "mean_abs":
        return _scatter(torch.abs(src), index, dim=dim, dim_size=dim_size, reduce="mean")

    if reduce == "l1":
        return _scatter(torch.abs(src), index, dim=dim, dim_size=dim_size, reduce="sum")

    if reduce == "l2":
        sum_sq = scatter_(src, index, dim=dim, reduce="sum_of_squares", dim_size=dim_size)
        return torch.sqrt(sum_sq + eps)

    if reduce == "geometric_mean":
        safe_src = torch.clamp(src, min=eps)
        log_vals = torch.log(safe_src)
        mean_log = _scatter(log_vals, index, dim=dim, dim_size=dim_size, reduce="mean")
        return torch.exp(mean_log)

    if reduce == "harmonic_mean":
        safe_src = torch.clamp(src, min=eps)
        inv = 1.0 / safe_src
        inv_sum = _scatter(inv, index, dim=dim, dim_size=dim_size, reduce="sum")
        count = scatter_(src, index, dim=dim, reduce="count", dim_size=dim_size)
        return count / (inv_sum + eps)

    if reduce == "midrange":
        max_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")
        min_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="min")
        return 0.5 * (max_vals + min_vals)

    if reduce == "amplitude":
        return _scatter(torch.abs(src), index, dim=dim, dim_size=dim_size, reduce="max")

    if reduce == "logsumexp":
        max_vals = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")
        max_per_elem = max_vals.index_select(dim, index)
        sum_exp = _scatter(
            torch.exp(src - max_per_elem),
            index,
            dim=dim,
            dim_size=dim_size,
            reduce="sum",
        )
        return max_vals + torch.log(sum_exp + eps)

    if reduce == "softmin":
        return -scatter_(-src, index, dim=dim, reduce="softmax", dim_size=dim_size, eps=eps)

    if reduce == "attention":
        if weights is None:
            raise ValueError("weights tensor must be provided for attention pooling")
        weighted_sum = _scatter(src * weights, index, dim=dim, dim_size=dim_size, reduce="sum")
        weight_sum = _scatter(weights, index, dim=dim, dim_size=dim_size, reduce="sum")
        return weighted_sum / (weight_sum + eps)

    if reduce == "normalized_sum":
        summed = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
        norm = torch.sqrt(
            _scatter(src**2, index, dim=dim, dim_size=dim_size, reduce="sum") + eps
        )
        return summed / norm

    if reduce == "any":
        return (_scatter(src, index, dim=dim, dim_size=dim_size, reduce="max") > 0).to(src.dtype)

    if reduce == "all":
        return (_scatter(src, index, dim=dim, dim_size=dim_size, reduce="min") > 0).to(src.dtype)

    if reduce == "xor":
        counts = _scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
        return (counts % 2).to(src.dtype)

    raise ValueError(f"Unsupported reduce method: {reduce}")
