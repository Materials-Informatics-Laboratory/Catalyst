import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from typing import Tuple

from ....nn.mlp import MLP


class EdgeConditionedConv(MessagePassing):
    """
    Catalyst-style edge-conditioned convolution.

    This is an efficient NNConv-like layer where edge features generate
    a node-dimensional filter/gate for messages.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        aggr_scheme: str = "add",
        act=nn.SiLU(),
    ):
        super().__init__(aggr=aggr_scheme)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.aggr_scheme = aggr_scheme
        self.act = act

        self.node_src = nn.Linear(node_dim, node_dim)
        self.node_root = nn.Linear(node_dim, node_dim)

        # Edge-conditioned filter. Produces one gate per node feature.
        self.edge_filter = nn.Sequential(
            MLP(hs=[edge_dim] + [node_dim] * 2, act=act),
            nn.LayerNorm(node_dim),
            nn.Sigmoid(),
        )

        # Optional additive edge-to-message term.
        self.edge_to_msg = nn.Linear(edge_dim, node_dim)

        # Node update.
        self.node_mlp = nn.Sequential(
            MLP(hs=[node_dim] + [node_dim] * 3, act=act),
            nn.LayerNorm(node_dim),
        )

        # Edge update.
        self.edge_mlp = nn.Sequential(
            MLP(hs=[node_dim * 2 + edge_dim] + [edge_dim] * 3, act=act),
            nn.LayerNorm(edge_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.node_src, self.node_root, self.edge_to_msg]:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        i, j = edge_index

        # Edge-conditioned message.
        filt = self.edge_filter(edge_attr)
        msg = filt * self.node_src(x[j]) + self.edge_to_msg(edge_attr)

        # Aggregate onto receiver/central node i.
        agg = scatter(
            msg,
            index=i,
            dim=0,
            dim_size=x.size(0),
            reduce=self.aggr_scheme,
        )

        # Residual node update.
        node_input = self.node_root(x) + agg
        x = x + self.node_mlp(node_input)

        # Residual edge update.
        edge_input = torch.cat([x[i], x[j], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_input)

        return x, edge_attr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node_dim={self.node_dim}, edge_dim={self.edge_dim}, "
            f"aggr_scheme='{self.aggr_scheme}')"
        )