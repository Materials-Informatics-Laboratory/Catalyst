import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from typing import List, Tuple

from ....nn.mlp import MLP


class GINEConv(MessagePassing):
    """
    Catalyst-style GINE convolution.

    Input:
        x:         [n_nodes, node_dim]
        edge_index:[2, n_edges]
        edge_attr: [n_edges, edge_dim]

    Output:
        x:         [n_nodes, node_dim]
        edge_attr: [n_edges, edge_dim]
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        aggr_scheme: str = "add",
        act=nn.SiLU(),
        train_eps: bool = True,
        eps: float = 0.0,
    ):
        super().__init__(aggr=aggr_scheme)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.aggr_scheme = aggr_scheme
        self.act = act

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))

        # Project edge features into node-feature space for GINE message.
        self.edge_to_node = nn.Linear(edge_dim, node_dim)

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
        torch.nn.init.xavier_uniform_(self.edge_to_node.weight)
        self.edge_to_node.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        i, j = edge_index

        # GINE-style edge-aware messages.
        msg = self.act(x[j] + self.edge_to_node(edge_attr))

        # Match your MGN convention: aggregate messages onto node i.
        agg = scatter(
            msg,
            index=i,
            dim=0,
            dim_size=x.size(0),
            reduce=self.aggr_scheme,
        )

        # Residual node update.
        node_input = (1.0 + self.eps) * x + agg
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