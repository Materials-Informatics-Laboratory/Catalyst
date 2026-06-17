# scatter_utils.py
import torch
import torch_scatter


def scatter_(src, index, dim=-1, reduce="mean", dim_size=None, eps=1e-9, weights=None):
    """
    Extended scatter function with the same signature as
    torch_geometric.utils.scatter, plus many new reduce options.

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

    # alias normalization
    if reduce == "add":
        reduce = "sum"
    if reduce in ["var", "variance"]:
        reduce = "variance"

    # --- Native ops ---
    if reduce in ["sum", "mean", "min", "max"]:
        op = getattr(torch_scatter, f"scatter_{reduce}")
        out = op(src, index, dim=dim, dim_size=dim_size)
        out = out[0] if isinstance(out, tuple) else out
        return out

    # --- Extended ops ---

    # count
    if reduce == "count":
        ones = torch.ones_like(src)
        return torch_scatter.scatter_add(ones, index, dim=dim, dim_size=dim_size)

    # variance
    if reduce == "variance":
        mean = torch_scatter.scatter_mean(src, index, dim=dim, dim_size=dim_size)
        mean_sq = torch_scatter.scatter_mean(src**2, index, dim=dim, dim_size=dim_size)
        return mean_sq - mean**2

    # std
    if reduce == "std":
        var = scatter_(src, index, dim=dim, reduce="variance", dim_size=dim_size, eps=eps)
        return torch.sqrt(var + eps)

    # range
    if reduce == "range":
        max_vals = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        min_vals = torch_scatter.scatter_min(src, index, dim=dim, dim_size=dim_size)[0]
        return max_vals - min_vals

    # softmax pooling
    if reduce == "softmax":
        max_vals = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        max_per_elem = max_vals.index_select(dim, index)
        src_exp = torch.exp(src - max_per_elem)
        denom = torch_scatter.scatter_add(src_exp, index, dim=dim, dim_size=dim_size)
        denom_per_elem = denom.index_select(dim, index)
        weights_ = src_exp / (denom_per_elem + eps)
        return torch_scatter.scatter_add(src * weights_, index, dim=dim, dim_size=dim_size)

    # z-score pooling
    if reduce == "normalized":
        mean = torch_scatter.scatter_mean(src, index, dim=dim, dim_size=dim_size)
        std = scatter_(src, index, dim=dim, reduce="std", dim_size=dim_size, eps=eps)
        return (src - mean.index_select(dim, index)) / (std.index_select(dim, index) + eps)

    # sum of squares
    if reduce == "sum_of_squares":
        return torch_scatter.scatter_add(src**2, index, dim=dim, dim_size=dim_size)

    # mean absolute value
    if reduce == "mean_abs":
        return torch_scatter.scatter_mean(torch.abs(src), index, dim=dim, dim_size=dim_size)

    # L1 norm
    if reduce == "l1":
        return torch_scatter.scatter_add(torch.abs(src), index, dim=dim, dim_size=dim_size)

    # L2 norm
    if reduce == "l2":
        sum_sq = scatter_(src, index, dim=dim, reduce="sum_of_squares", dim_size=dim_size)
        return torch.sqrt(sum_sq + eps)

    # geometric mean
    if reduce == "geometric_mean":
        safe_src = torch.clamp(src, min=eps)
        log_vals = torch.log(safe_src)
        mean_log = torch_scatter.scatter_mean(log_vals, index, dim=dim, dim_size=dim_size)
        return torch.exp(mean_log)

    # harmonic mean
    if reduce == "harmonic_mean":
        safe_src = torch.clamp(src, min=eps)
        inv = 1.0 / safe_src
        inv_sum = torch_scatter.scatter_add(inv, index, dim=dim, dim_size=dim_size)
        count = scatter_(src, index, dim=dim, reduce="count", dim_size=dim_size)
        return count / (inv_sum + eps)

    # midrange = (max + min)/2
    if reduce == "midrange":
        max_vals = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        min_vals = torch_scatter.scatter_min(src, index, dim=dim, dim_size=dim_size)[0]
        return 0.5 * (max_vals + min_vals)

    # amplitude = max(abs(x))
    if reduce == "amplitude":
        return torch_scatter.scatter_max(torch.abs(src), index, dim=dim, dim_size=dim_size)[0]

    # logsumexp
    if reduce == "logsumexp":
        max_vals = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        max_per_elem = max_vals.index_select(dim, index)
        sum_exp = torch_scatter.scatter_add(torch.exp(src - max_per_elem), index, dim=dim, dim_size=dim_size)
        return max_vals + torch.log(sum_exp + eps)

    # softmin
    if reduce == "softmin":
        return -scatter_(-src, index, dim=dim, reduce="softmax", dim_size=dim_size, eps=eps)

    # attention pooling (requires weights)
    if reduce == "attention":
        if weights is None:
            raise ValueError("weights tensor must be provided for attention pooling")
        weighted_sum = torch_scatter.scatter_add(src * weights, index, dim=dim, dim_size=dim_size)
        weight_sum = torch_scatter.scatter_add(weights, index, dim=dim, dim_size=dim_size)
        return weighted_sum / (weight_sum + eps)

    # normalized sum (sum / L2 norm of group)
    if reduce == "normalized_sum":
        summed = torch_scatter.scatter_add(src, index, dim=dim, dim_size=dim_size)
        norm = torch.sqrt(torch_scatter.scatter_add(src**2, index, dim=dim, dim_size=dim_size) + eps)
        return summed / norm

    # boolean-style
    if reduce == "any":
        return (torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)[0] > 0).to(src.dtype)

    if reduce == "all":
        return (torch_scatter.scatter_min(src, index, dim=dim, dim_size=dim_size)[0] > 0).to(src.dtype)

    if reduce == "xor":
        counts = torch_scatter.scatter_add(src, index, dim=dim, dim_size=dim_size)
        return (counts % 2).to(src.dtype)

    raise ValueError(f"Unsupported reduce method: {reduce}")
