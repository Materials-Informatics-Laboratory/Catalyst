"""
Catalyst standard graph decoders.

Recommended location:
    catalyst/ml/gnn/modules/decoders/standard_decoders.py

Purpose
-------
This file pulls the existing decoder/readout logic out of the old ALIGNN model
file and makes it plug-and-play with the new GenericGNN / OrderProcessor format.

The decoders are generic enough to live in a standard decoder module because
they operate on hidden graph fields, not directly on atomistic raw features.

They support both:

    New order-style hidden names:
        h_1, h_2, h_3

and legacy Catalyst hidden names:

    Atomic_Graph_Data:
        h_atm, h_bnd, h_ang

    Generic_Graph_Data:
        h_g_node, h_a_node, h_a_edge

Backward-compatible aliases are provided:

    Decoder = ScalarDecoder
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from ....nn.mlp import MLP


# =============================================================================
# Helpers
# =============================================================================


def _has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def _first_attr(data, *names: str, required: bool = True):
    for name in names:
        if _has_attr(data, name):
            return getattr(data, name)

    if required:
        raise AttributeError(f"Could not find any of these graph attributes: {names}")

    return None


def _order_hidden_features(data) -> List[torch.Tensor]:
    """
    Return available hidden features in order: 1-body, 2-body, optional 3-body.
    """
    h_1 = _first_attr(data, "h_1", "h_atm", "h_g_node")
    h_2 = _first_attr(data, "h_2", "h_bnd", "h_a_node")
    h_3 = _first_attr(data, "h_3", "h_ang", "h_a_edge", required=False)

    features = [h_1, h_2]
    if h_3 is not None:
        features.append(h_3)

    return features


# =============================================================================
# Standard decoders
# =============================================================================


class ScalarDecoder(nn.Module):
    """
    Generic scalar/vector decoder for hidden graph orders.

    This is the extracted and generalized version of the old Decoder class.

    If combine=True:
        returns torch.cat([decoded_h1, decoded_h2, decoded_h3], dim=0)

    If combine=False:
        returns [decoded_h1, decoded_h2, decoded_h3]
    """

    def __init__(self, in_dim: int, out_dim: int, act=nn.SiLU(), combine: bool = True):
        super().__init__()
        self.node_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.combine = combine
        self.act_func = act

        self.decoder = MLP([in_dim, in_dim, out_dim], act=self.act_func)

    def forward(self, data):
        outputs = [self.decoder(h) for h in _order_hidden_features(data)]

        if self.combine:
            return torch.cat(outputs, dim=0)

        return outputs


class PositiveScalarsDecoder(nn.Module):
    """
    Generic positive scalar decoder for hidden graph orders.

    This is the extracted and generalized version of the old PositiveScalarsDecoder.

    It returns a list of positive scalar tensors, one per available order.
    """

    def __init__(self, dim: int, act=nn.SiLU()):
        super().__init__()
        self.dim = dim
        self.act_func = act

        self.transform_1 = nn.Sequential(
            MLP([dim, dim, 1], act=self.act_func),
            nn.Softplus(),
        )
        self.transform_2 = nn.Sequential(
            MLP([dim, dim, 1], act=self.act_func),
            nn.Softplus(),
        )
        self.transform_3 = nn.Sequential(
            MLP([dim, dim, 1], act=self.act_func),
            nn.Softplus(),
        )

        # Legacy attribute names for checkpoint/backward compatibility.
        self.transform_g_node = self.transform_1
        self.transform_a_node = self.transform_2
        self.transform_a_edge = self.transform_3

    def forward(self, data):
        h = _order_hidden_features(data)

        outputs = [
            self.transform_1(h[0]),
            self.transform_2(h[1]),
        ]

        if len(h) > 2:
            outputs.append(self.transform_3(h[2]))

        return outputs


class MultiScalarDecoder(nn.Module):
    """
    Decode ``K`` independent invariant scalar channels for each graph order.

    The output shape for every available order is ``[N_order, K]``.  Channels
    share the upstream GNN representation but are ordinary scalar regression
    targets: they have no Cartesian/equivariant interpretation and are never
    passed through ``VectorChannelAdapter``.

    Parameters
    ----------
    dim
        Hidden feature dimension.
    num_targets
        Number of independent scalar properties to predict.
    positive
        Apply Softplus to every output channel when all targets are known to be
        nonnegative.  The default is signed/unconstrained regression.
    """

    def __init__(
        self,
        dim: int,
        num_targets: int,
        act=nn.SiLU(),
        positive: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_targets = int(num_targets)
        self.out_dim = self.num_targets
        self.K = self.num_targets  # Compatibility with PositiveKChannelDecoder.
        self.act_func = act
        self.positive = bool(positive)

        if self.num_targets < 2:
            raise ValueError(
                "MultiScalarDecoder requires num_targets >= 2. "
                "Use ScalarDecoder/PositiveScalarsDecoder for one target."
            )

        def make_transform():
            layers = [MLP([self.dim, self.dim, self.num_targets], act=self.act_func)]
            if self.positive:
                layers.append(nn.Softplus())
            return nn.Sequential(*layers)

        self.transform_1 = make_transform()
        self.transform_2 = make_transform()
        self.transform_3 = make_transform()

        # Legacy/order aliases used by downstream analysis code.
        self.transform_g_node = self.transform_1
        self.transform_a_node = self.transform_2
        self.transform_a_edge = self.transform_3

    def forward(self, data):
        h = _order_hidden_features(data)

        outputs = [
            self.transform_1(h[0]),
            self.transform_2(h[1]),
        ]

        if len(h) > 2:
            outputs.append(self.transform_3(h[2]))

        return outputs


class PositiveKChannelDecoder(nn.Module):
    """
    Generic positive K-channel decoder for hidden graph orders.

    This is the extracted and generalized version of the old PositiveKChannelDecoder.
    """

    def __init__(self, dim: int, act=nn.SiLU(), K: int = 16):
        super().__init__()
        self.dim = dim
        self.K = K
        self.act_func = act

        self.transform_1 = nn.Sequential(
            MLP([dim, dim, K], act=self.act_func),
            nn.Softplus(),
        )
        self.transform_2 = nn.Sequential(
            MLP([dim, dim, K], act=self.act_func),
            nn.Softplus(),
        )
        self.transform_3 = nn.Sequential(
            MLP([dim, dim, K], act=self.act_func),
            nn.Softplus(),
        )

        # Legacy attribute names for checkpoint/backward compatibility.
        self.transform_g_node = self.transform_1
        self.transform_a_node = self.transform_2
        self.transform_a_edge = self.transform_3

    def forward(self, data):
        h = _order_hidden_features(data)

        outputs = [
            self.transform_1(h[0]),
            self.transform_2(h[1]),
        ]

        if len(h) > 2:
            outputs.append(self.transform_3(h[2]))

        return outputs


class PositiveFeatureReadout(nn.Module):
    """
    Maps pooled positive feature vectors to scalar or vector outputs.

    This is used after accumulate_predictions(..., channel_mode='latent').

    If out_dim = 1:
        scalar prediction.

    If out_dim > 1:
        true vector-valued prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int = 1,
        nonnegative_weights: bool = True,
        signed: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.nonnegative_weights = nonnegative_weights
        self.signed = signed

        if signed:
            # Positive-minus-positive decomposition:
            # y = b + W_plus z - W_minus z
            self.weight_plus_raw = nn.Parameter(torch.zeros(feature_dim, out_dim))
            self.weight_minus_raw = nn.Parameter(torch.zeros(feature_dim, out_dim))
        else:
            self.weight_raw = nn.Parameter(torch.zeros(feature_dim, out_dim))

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, features: torch.Tensor, return_contributions: bool = False):
        """
        features:
            [num_graphs, feature_dim]
            or [feature_dim]

        returns:
            y:
                [num_graphs, out_dim]
                or [out_dim]
        """
        if self.signed:
            w_plus = torch.nn.functional.softplus(self.weight_plus_raw)
            w_minus = torch.nn.functional.softplus(self.weight_minus_raw)

            y_plus = features @ w_plus
            y_minus = features @ w_minus
            y = self.bias + y_plus - y_minus

            if return_contributions:
                contrib_plus = features.unsqueeze(-1) * w_plus
                contrib_minus = features.unsqueeze(-1) * w_minus

                return y, {
                    "w_plus": w_plus,
                    "w_minus": w_minus,
                    "contrib_plus": contrib_plus,
                    "contrib_minus": contrib_minus,
                    "net_contrib": contrib_plus - contrib_minus,
                }

            return y

        if self.nonnegative_weights:
            w = torch.nn.functional.softplus(self.weight_raw)
        else:
            w = self.weight_raw

        y = self.bias + features @ w

        if return_contributions:
            contributions = features.unsqueeze(-1) * w

            return y, {
                "weights": w,
                "contributions": contributions,
            }

        return y


# Backward-compatible name from old alignn.py.
Decoder = ScalarDecoder


__all__ = [
    "ScalarDecoder",
    "Decoder",
    "PositiveScalarsDecoder",
    "MultiScalarDecoder",
    "PositiveKChannelDecoder",
    "PositiveFeatureReadout",
]
