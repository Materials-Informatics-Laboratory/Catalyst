import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from typing import Tuple

from ....nn.mlp import MLP


class PNAConv(MessagePassing):
    """
    Catalyst-style PNA convolution.

    Uses multiple aggregators and simple degree scalers, while preserving
    the Catalyst convention:

        forward(x, edge_index, edge_attr) -> x, edge_attr
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        aggr_scheme: str = "add",
        act=nn.SiLU(),
        avg_degree: float = 10.0,
        use_degree_scalers: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(aggr=aggr_scheme)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.aggr_scheme = aggr_scheme
        self.act = act
        self.avg_degree = float(avg_degree)
        self.use_degree_scalers = use_degree_scalers
        self.eps = eps

        # Build messages from source node + edge features.
        self.message_mlp = nn.Sequential(
            MLP(hs=[node_dim + edge_dim] + [node_dim] * 3, act=act),
            nn.LayerNorm(node_dim),
        )

        # Aggregators: mean, max, min, std.
        n_aggregators = 4

        # Scalers: identity, amplification, attenuation.
        n_scalers = 3 if use_degree_scalers else 1

        node_input_dim = node_dim + node_dim * n_aggregators * n_scalers

        self.node_mlp = nn.Sequential(
            MLP(hs=[node_input_dim] + [node_dim] * 3, act=act),
            nn.LayerNorm(node_dim),
        )

        self.edge_mlp = nn.Sequential(
            MLP(hs=[node_dim * 2 + edge_dim] + [edge_dim] * 3, act=act),
            nn.LayerNorm(edge_dim),
        )

    def _aggregate(self, msg: Tensor, index: Tensor, dim_size: int) -> Tensor:
        mean = scatter(msg, index=index, dim=0, dim_size=dim_size, reduce="mean")
        max_ = scatter(msg, index=index, dim=0, dim_size=dim_size, reduce="max")
        min_ = scatter(msg, index=index, dim=0, dim_size=dim_size, reduce="min")

        mean_sq = scatter(msg * msg, index=index, dim=0, dim_size=dim_size, reduce="mean")
        std = torch.sqrt(torch.clamp(mean_sq - mean * mean, min=0.0) + self.eps)

        return torch.cat([mean, max_, min_, std], dim=-1)

    def _degree_scalers(self, agg: Tensor, degree: Tensor) -> Tensor:
        if not self.use_degree_scalers:
            return agg

        avg_log_degree = torch.log(
            torch.tensor(
                self.avg_degree + 1.0,
                dtype=agg.dtype,
                device=agg.device,
            )
        )

        log_degree = torch.log(degree + 1.0).clamp(min=self.eps)

        amplification = log_degree / avg_log_degree.clamp(min=self.eps)
        attenuation = avg_log_degree / log_degree

        return torch.cat(
            [
                agg,
                agg * amplification,
                agg * attenuation,
            ],
            dim=-1,
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        i, j = edge_index

        # Edge-aware messages.
        msg_input = torch.cat([x[j], edge_attr], dim=-1)
        msg = self.message_mlp(msg_input)

        # Multi-aggregator PNA-style pooling.
        agg = self._aggregate(msg, index=i, dim_size=x.size(0))

        # Degree scalers.
        ones = torch.ones((i.size(0), 1), dtype=x.dtype, device=x.device)
        degree = scatter(ones, index=i, dim=0, dim_size=x.size(0), reduce="add")
        agg = self._degree_scalers(agg, degree)

        # Residual node update.
        node_input = torch.cat([x, agg], dim=-1)
        x = x + self.node_mlp(node_input)

        # Residual edge update.
        edge_input = torch.cat([x[i], x[j], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_input)

        return x, edge_attr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, edge_dim={self.edge_dim}, "
            f"avg_degree={self.avg_degree}, "
            f"use_degree_scalers={self.use_degree_scalers})"
        )