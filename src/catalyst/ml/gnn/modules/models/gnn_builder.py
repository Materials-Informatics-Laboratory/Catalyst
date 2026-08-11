"""
Catalyst GNN builder.

Recommended location:
    catalyst/ml/gnn/modules/models/gnn_builder.py

This file replaces the old generic_gnn.py naming with a clearer separation:

    GNN
        High-level training/testing/checkpointing engine outside this file.

    GNNBuilder
        torch.nn.Module that assembles:
            encoder -> processor -> decoder

    EquivariantGNN
        Convenience wrapper around GNNBuilder that preselects:
            EquivariantAtomicEncoder
            EquivariantProcessor
            EquivariantDecoder

The public construction path should be:

    model = build_model(...)

and then:

    trainer = GNN(model, device)

Compatibility
-------------
The old GenericGNN names are kept as aliases so existing imports can migrate
gradually:

    GenericGNN = GNNBuilder
    GenericGNNPreset = GNNBuilderPreset
    build_generic_gnn = build_gnn_builder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..conv.conv_factory import build_activation
from ..processors.order_processor import OrderProcessor
from ..processors.scalar_processor import ScalarProcessor


# =============================================================================
# Graph alias helpers
# =============================================================================


def _has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def _first_attr(data, *names: str, required: bool = False):
    for name in names:
        if _has_attr(data, name):
            return getattr(data, name)

    if required:
        raise AttributeError(f"Could not find any of these graph attributes: {names}")

    return None


def attach_order_input_aliases(data):
    """
    Attach generic order-style raw feature aliases without deleting legacy fields.

    Atomic legacy:
        x_atm -> x_1
        x_bnd -> x_2
        x_ang -> x_3

    Generic legacy:
        node_G -> x_1
        node_A -> x_2
        edge_A -> x_3

    PyG-like:
        x -> x_1
        edge_attr -> x_2

    Edges:
        edge_index_G -> edge_index_2
        edge_index -> edge_index_2
        edge_index_A -> edge_index_3
    """
    if not _has_attr(data, "x_1"):
        value = _first_attr(data, "x_atm", "node_G", "x")
        if value is not None:
            data.x_1 = value

    if not _has_attr(data, "x_2"):
        value = _first_attr(data, "x_bnd", "node_A", "edge_attr")
        if value is not None:
            data.x_2 = value

    if not _has_attr(data, "x_3"):
        value = _first_attr(data, "x_ang", "edge_A")
        if value is not None:
            data.x_3 = value

    if not _has_attr(data, "edge_index_2"):
        value = _first_attr(data, "edge_index_G", "edge_index")
        if value is not None:
            data.edge_index_2 = value

    if not _has_attr(data, "edge_index_3"):
        value = _first_attr(data, "edge_index_A")
        if value is not None:
            data.edge_index_3 = value

    return data


def attach_equivariant_input_aliases(data):
    """
    Attach PyG/equivariant aliases expected by equivariant processors.

    Required equivariant fields are normally provided by the updated graph
    builders:

        z, pos, edge_index, cell, pbc, shifts

    This helper only fills safe aliases; it does not invent geometry beyond
    zero shifts for nonperiodic/single-graph fallback cases.
    """
    if not _has_attr(data, "edge_index"):
        value = _first_attr(data, "edge_index_G", "edge_index_2")
        if value is not None:
            data.edge_index = value

    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        if _has_attr(data, "z"):
            data.num_nodes = int(data.z.size(0))
        elif _has_attr(data, "pos"):
            data.num_nodes = int(data.pos.size(0))
        else:
            value = _first_attr(data, "x_atm", "node_G", "x_1", "x")
            if value is not None:
                data.num_nodes = int(value.size(0))

    if _has_attr(data, "pos") and _has_attr(data, "edge_index") and not _has_attr(data, "shifts"):
        data.shifts = torch.zeros(
            (data.edge_index.size(1), 3),
            dtype=torch.long,
            device=data.edge_index.device,
        )

    return data


def attach_input_aliases(data):
    """
    Attach all safe aliases used by the scalar/order/equivariant branches.
    """
    data = attach_order_input_aliases(data)
    data = attach_equivariant_input_aliases(data)
    return data


def attach_order_hidden_aliases(data):
    """
    Attach generic order-style hidden aliases from legacy hidden fields.
    """
    if not _has_attr(data, "h_1"):
        value = _first_attr(data, "h_atm", "h_g_node", "h_node", "h_scalar")
        if value is not None:
            data.h_1 = value

    if not _has_attr(data, "h_2"):
        value = _first_attr(data, "h_bnd", "h_a_node", "h_edge")
        if value is not None:
            data.h_2 = value

    if not _has_attr(data, "h_3"):
        value = _first_attr(data, "h_ang", "h_a_edge")
        if value is not None:
            data.h_3 = value

    return data


def attach_equivariant_hidden_aliases(data):
    """
    Attach equivariant hidden aliases from generic/order hidden fields.
    """
    if not _has_attr(data, "h_scalar"):
        value = _first_attr(data, "h_1", "h_atm", "h_g_node", "h_node")
        if value is not None:
            data.h_scalar = value

    if _has_attr(data, "h_scalar"):
        data.h_1 = data.h_scalar
        data.h_node = data.h_scalar

    return data


def attach_hidden_aliases(data):
    data = attach_order_hidden_aliases(data)
    data = attach_equivariant_hidden_aliases(data)
    return data


def backfill_legacy_hidden_names(data):
    """
    Backfill old Catalyst names from order-style/equivariant hidden names.

    This keeps existing decoders/readout utilities working while the package is
    migrated to the new modular names.
    """
    if _has_attr(data, "h_scalar"):
        data.h_1 = data.h_scalar
        data.h_node = data.h_scalar

    if _has_attr(data, "h_1"):
        if _has_attr(data, "x_atm"):
            data.h_atm = data.h_1
        if _has_attr(data, "node_G"):
            data.h_g_node = data.h_1
        if _has_attr(data, "x"):
            data.h_node = data.h_1

    if _has_attr(data, "h_2"):
        if _has_attr(data, "x_bnd"):
            data.h_bnd = data.h_2
        if _has_attr(data, "node_A"):
            data.h_a_node = data.h_2
        if _has_attr(data, "edge_attr"):
            data.h_edge = data.h_2

    if _has_attr(data, "h_3"):
        if _has_attr(data, "x_ang"):
            data.h_ang = data.h_3
        if _has_attr(data, "edge_A"):
            data.h_a_edge = data.h_3

    return data


def prepare_gradient_input(
    data,
    *,
    gradient_input_key: str = "pos",
    clone: bool = False,
):
    """
    Enable gradients on the input tensor used by scalar_gradient decoders.

    This must happen before encoder -> processor -> decoder, because equivariant
    processors recompute edge geometry from positions inside their forward pass.
    """
    if not _has_attr(data, gradient_input_key):
        raise AttributeError(
            f"Cannot prepare scalar-gradient input because data.{gradient_input_key} is missing."
        )

    value = getattr(data, gradient_input_key)

    if clone:
        value = value.detach().clone()

    value.requires_grad_(True)
    setattr(data, gradient_input_key, value)
    return data


# =============================================================================
# Core model builder
# =============================================================================


class GNNBuilder(nn.Module):
    """
    Composable GNN model:

        data -> encoder -> processor -> decoder

    This is an nn.Module intended to be passed into the higher-level Catalyst GNN
    training/testing/checkpointing engine.

    Parameters
    ----------
    encoder, processor, decoder
        The three model components.
    prepare_gradient
        If True, enable gradients on ``gradient_input_key`` before the forward
        pass.  If None, this is inferred automatically for decoders with
        output_type == "scalar_gradient".
    force_enable_grad
        If True, wraps forward in torch.enable_grad() when prepare_gradient is
        active.  This allows scalar_gradient models to work even inside an
        outer @torch.no_grad validation wrapper.
    """

    def __init__(
        self,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        *,
        name: str = "GNNBuilder",
        prepare_gradient: Optional[bool] = None,
        gradient_input_key: str = "pos",
        clone_gradient_input: bool = False,
        force_enable_grad: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.name = name

        decoder_output_type = str(getattr(decoder, "output_type", "")).lower()
        if prepare_gradient is None:
            prepare_gradient = decoder_output_type == "scalar_gradient"

        self.prepare_gradient = bool(prepare_gradient)
        self.gradient_input_key = str(gradient_input_key)
        self.clone_gradient_input = bool(clone_gradient_input)
        self.force_enable_grad = bool(force_enable_grad)

    def _forward_impl(self, data):
        data = attach_input_aliases(data)

        if self.prepare_gradient:
            data = prepare_gradient_input(
                data,
                gradient_input_key=self.gradient_input_key,
                clone=self.clone_gradient_input,
            )

        data = self.encoder(data)
        data = attach_hidden_aliases(data)

        data = self.processor(data)
        data = attach_hidden_aliases(data)
        data = backfill_legacy_hidden_names(data)

        return self.decoder(data)

    def forward(self, data):
        if self.prepare_gradient and self.force_enable_grad:
            with torch.enable_grad():
                return self._forward_impl(data)

        return self._forward_impl(data)


# Backward-compatible alias for existing imports.
GenericGNN = GNNBuilder


# =============================================================================
# Default component builders
# =============================================================================


def _lazy_import_default_components():
    """
    Import default modular encoders/decoders lazily.
    """
    from ..encoders.atomic_encoders import AtomicGraphEncoder
    from ..encoders.atomic_encoders import GenericFeatureEncoder
    from ..encoders.atomic_encoders import Encoder_atomic
    from ..encoders.atomic_encoders import Encoder_generic

    from ..decoders.standard_decoders import ScalarDecoder
    from ..decoders.standard_decoders import Decoder
    from ..decoders.standard_decoders import PositiveScalarsDecoder
    from ..decoders.standard_decoders import MultiScalarDecoder
    from ..decoders.standard_decoders import PositiveKChannelDecoder
    from ..decoders.standard_decoders import PositiveFeatureReadout

    from ..encoders.equivariant_encoders import EquivariantAtomicEncoder
    from ..decoders.equivariant_decoders import EquivariantDecoder

    return {
        "AtomicGraphEncoder": AtomicGraphEncoder,
        "GenericFeatureEncoder": GenericFeatureEncoder,
        "Encoder_atomic": Encoder_atomic,
        "Encoder_generic": Encoder_generic,
        "ScalarDecoder": ScalarDecoder,
        "Decoder": Decoder,
        "PositiveScalarsDecoder": PositiveScalarsDecoder,
        "MultiScalarDecoder": MultiScalarDecoder,
        "PositiveKChannelDecoder": PositiveKChannelDecoder,
        "PositiveFeatureReadout": PositiveFeatureReadout,
        "EquivariantAtomicEncoder": EquivariantAtomicEncoder,
        "EquivariantDecoder": EquivariantDecoder,
    }


def build_default_encoder(
    encoder_type: str,
    *,
    dim: int,
    act=None,
    num_species: Optional[int] = None,
    cutoff: float = 5.0,
    dihedral: bool = False,
    basis: str = "gaussian",
    params_group=None,
    **encoder_kwargs,
) -> nn.Module:
    """
    Build one of the standard Catalyst encoders.
    """
    components = _lazy_import_default_components()
    encoder_type = str(encoder_type).lower().strip()
    act = build_activation(act)

    if encoder_type in {"atomic", "atomistic", "alignn_atomic", "encoder_atomic"}:
        if num_species is None:
            raise ValueError("num_species is required for the atomic/ALIGNN encoder.")

        return components["AtomicGraphEncoder"](
            num_species=num_species,
            cutoff=cutoff,
            act=act,
            dim=dim,
            dihedral=dihedral,
            params_group=params_group,
            **encoder_kwargs,
        )

    if encoder_type in {"generic", "feature", "generic_feature", "encoder_generic"}:
        return components["GenericFeatureEncoder"](
            act=act,
            dim=dim,
            basis=basis,
            params_group=params_group,
            **encoder_kwargs,
        )

    if encoder_type in {
        "equivariant",
        "equivariant_atomic",
        "atomic_equivariant",
        "egnn_atomic",
        "equivariant_atomistic",
    }:
        encoder_cls = components["EquivariantAtomicEncoder"]
        if encoder_cls is None:
            raise ImportError(
                "EquivariantAtomicEncoder is not available. Install:\n"
                "  catalyst/ml/gnn/modules/encoders/equivariant_encoders.py"
            )

        return encoder_cls(
            dim=dim,
            num_species=num_species,
            **encoder_kwargs,
        )

    raise ValueError(
        f"Unsupported encoder_type={encoder_type!r}. "
        "Supported defaults are: 'atomic', 'generic', and 'equivariant_atomic'."
    )


def build_default_decoder(
    decoder_type: str,
    *,
    dim: int,
    out_dim: int = 1,
    act=None,
    combine: bool = True,
    K: int = 16,
    output_type: str = "scalar",
    output_level: str = "graph",
    return_dict: bool = False,
    gradient_input_key: str = "pos",
    gradient_sign: str = "negative",
    **decoder_kwargs,
) -> nn.Module:
    """
    Build one of the standard Catalyst decoders.
    """
    components = _lazy_import_default_components()
    decoder_type = str(decoder_type).lower().strip()
    act = build_activation(act)

    if decoder_type in {"scalar", "decoder", "standard"}:
        return components["ScalarDecoder"](
            in_dim=dim,
            out_dim=out_dim,
            act=act,
            combine=combine,
            **decoder_kwargs,
        )

    if decoder_type in {"positive", "positive_scalar", "positive_scalars"}:
        return components["PositiveScalarsDecoder"](
            dim=dim,
            act=act,
            **decoder_kwargs,
        )

    if decoder_type in {
        "multiscalar",
        "multi_scalar",
        "scalar_channels",
        "independent_scalars",
    }:
        return components["MultiScalarDecoder"](
            dim=dim,
            num_targets=out_dim,
            act=act,
            **decoder_kwargs,
        )

    if decoder_type in {"positive_k", "positive_k_channel", "positive_kchannel"}:
        return components["PositiveKChannelDecoder"](
            dim=dim,
            act=act,
            K=K,
            **decoder_kwargs,
        )

    if decoder_type in {"equivariant", "equivariant_decoder", "generic_equivariant"}:
        decoder_cls = components["EquivariantDecoder"]
        if decoder_cls is None:
            raise ImportError(
                "EquivariantDecoder is not available. Install:\n"
                "  catalyst/ml/gnn/modules/decoders/equivariant_decoders.py"
            )

        return decoder_cls(
            dim=dim,
            output_type=output_type,
            output_level=output_level,
            out_dim=out_dim,
            act=act,
            return_dict=return_dict,
            gradient_input_key=gradient_input_key,
            gradient_sign=gradient_sign,
            **decoder_kwargs,
        )

    raise ValueError(
        f"Unsupported decoder_type={decoder_type!r}. "
        "Supported defaults are: 'scalar', 'positive', 'multiscalar', "
        "'positive_k', and 'equivariant'."
    )


def build_default_processor(
    processor_type: str,
    *,
    dim: int,
    num_convs: int,
    conv_type: str = "mesh",
    aggr_scheme: str = "add",
    encode_3body: bool = True,
    threebody_conv_type: Optional[str] = None,
    act=None,
    cutoff: float = 5.0,
    equivariant_type: str = "egnn",
    rbf_dim: int = 32,
    conv_kwargs: Optional[Dict[str, Any]] = None,
    threebody_conv_kwargs: Optional[Dict[str, Any]] = None,
    **processor_kwargs,
) -> nn.Module:
    """
    Build one of the standard Catalyst processors.

    processor_type="order"
        ALIGNN-style 1/2/3-body processor.

    processor_type="scalar"
        One-graph scalar/invariant processor.

    processor_type="equivariant"
        Geometry-aware equivariant processor such as EGNN.
    """
    processor_type = str(processor_type).lower().strip()
    act = build_activation(act)

    if processor_type in {"order", "alignn", "three_body", "3body", "line_graph"}:
        return OrderProcessor(
            dim=dim,
            num_convs=num_convs,
            conv_type=conv_type,
            aggr_scheme=aggr_scheme,
            encode_3body=encode_3body,
            threebody_conv_type=threebody_conv_type,
            act=act,
            conv_kwargs=conv_kwargs,
            threebody_conv_kwargs=threebody_conv_kwargs,
            **processor_kwargs,
        )

    if processor_type in {"scalar", "standard", "one_graph", "1graph", "message_passing"}:
        return ScalarProcessor(
            dim=dim,
            num_convs=num_convs,
            conv_type=conv_type,
            aggr_scheme=aggr_scheme,
            act=act,
            conv_kwargs=conv_kwargs,
            **processor_kwargs,
        )

    if processor_type in {"equivariant", "egnn", "equivariant_processor"}:
        from ..processors.equivariant_processor import EquivariantProcessor

        return EquivariantProcessor(
            dim=dim,
            num_convs=num_convs,
            equivariant_type=equivariant_type,
            cutoff=cutoff,
            rbf_dim=rbf_dim,
            act=act,
            **processor_kwargs,
        )

    raise ValueError(
        f"Unsupported processor_type={processor_type!r}. "
        "Supported defaults are: 'order', 'scalar', and 'equivariant'."
    )


# =============================================================================
# Convenience wrappers
# =============================================================================


class EquivariantGNN(GNNBuilder):
    """
    Convenience wrapper for the equivariant branch.

    This is still a GNNBuilder and still works with the high-level GNN training
    class.  It simply preselects the equivariant encoder/processor/decoder so
    users can call one model constructor.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        num_convs: int = 4,
        cutoff: float = 5.0,
        num_species: Optional[int] = None,
        equivariant_type: str = "egnn",
        output_type: str = "vector",
        output_level: str = "node",
        out_dim: int = 1,
        act=None,
        rbf_dim: int = 32,
        dropout: float = 0.0,
        return_dict: bool = False,
        gradient_input_key: str = "pos",
        gradient_sign: str = "negative",
        prepare_gradient: Optional[bool] = None,
        clone_gradient_input: bool = False,
        force_enable_grad: bool = True,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "EquivariantGNN",
    ):
        act = build_activation(act)
        encoder_kwargs = dict(encoder_kwargs or {})
        processor_kwargs = dict(processor_kwargs or {})
        decoder_kwargs = dict(decoder_kwargs or {})

        encoder = build_default_encoder(
            "equivariant_atomic",
            dim=dim,
            num_species=num_species,
            dropout=dropout,
            norm=True,
            **encoder_kwargs,
        )

        processor = build_default_processor(
            "equivariant",
            dim=dim,
            num_convs=num_convs,
            cutoff=cutoff,
            equivariant_type=equivariant_type,
            rbf_dim=rbf_dim,
            act=act,
            dropout=dropout,
            **processor_kwargs,
        )

        decoder = build_default_decoder(
            "equivariant",
            dim=dim,
            out_dim=out_dim,
            act=act,
            output_type=output_type,
            output_level=output_level,
            return_dict=return_dict,
            gradient_input_key=gradient_input_key,
            gradient_sign=gradient_sign,
            **decoder_kwargs,
        )

        super().__init__(
            encoder=encoder,
            processor=processor,
            decoder=decoder,
            name=name,
            prepare_gradient=prepare_gradient,
            gradient_input_key=gradient_input_key,
            clone_gradient_input=clone_gradient_input,
            force_enable_grad=force_enable_grad,
        )


# =============================================================================
# Presets and factories
# =============================================================================


@dataclass(frozen=True)
class GNNBuilderPreset:
    name: str
    encoder_type: str
    conv_type: str = "mesh"
    decoder_type: str = "positive"
    processor_type: str = "order"
    encode_3body: bool = True
    threebody_conv_type: Optional[str] = None
    equivariant_type: str = "egnn"
    output_type: str = "scalar"
    output_level: str = "graph"


# Backward-compatible alias.
GenericGNNPreset = GNNBuilderPreset


PRESETS: Dict[str, GNNBuilderPreset] = {
    "alignn": GNNBuilderPreset(
        name="ALIGNN",
        encoder_type="atomic",
        conv_type="gated_gcn",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "mgn": GNNBuilderPreset(
        name="MeshGraphNets",
        encoder_type="atomic",
        conv_type="mesh",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "mesh": GNNBuilderPreset(
        name="MeshGraphNets",
        encoder_type="atomic",
        conv_type="mesh",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "gated_gcn": GNNBuilderPreset(
        name="GatedGCN",
        encoder_type="atomic",
        conv_type="gated_gcn",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "gine": GNNBuilderPreset(
        name="GINE",
        encoder_type="atomic",
        conv_type="gine",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "edge_conditioned": GNNBuilderPreset(
        name="EdgeConditioned",
        encoder_type="atomic",
        conv_type="edge_conditioned",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "nnconv": GNNBuilderPreset(
        name="NNConv",
        encoder_type="atomic",
        conv_type="edge_conditioned",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "pna": GNNBuilderPreset(
        name="PNA",
        encoder_type="atomic",
        conv_type="pna",
        decoder_type="positive",
        processor_type="order",
        encode_3body=True,
    ),
    "generic_gnn": GNNBuilderPreset(
        name="GNNBuilder",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="order",
        encode_3body=True,
    ),
    "gnn_builder": GNNBuilderPreset(
        name="GNNBuilder",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="order",
        encode_3body=True,
    ),
    "scalar_gnn": GNNBuilderPreset(
        name="ScalarGNN",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="scalar",
        encode_3body=False,
    ),
    "scalar": GNNBuilderPreset(
        name="ScalarGNN",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="scalar",
        encode_3body=False,
    ),
    "equivariant": GNNBuilderPreset(
        name="EquivariantGNN",
        encoder_type="equivariant_atomic",
        decoder_type="equivariant",
        processor_type="equivariant",
        encode_3body=False,
        equivariant_type="egnn",
        output_type="vector",
        output_level="node",
    ),
    "egnn": GNNBuilderPreset(
        name="EquivariantGNN",
        encoder_type="equivariant_atomic",
        decoder_type="equivariant",
        processor_type="equivariant",
        encode_3body=False,
        equivariant_type="egnn",
        output_type="vector",
        output_level="node",
    ),
    "equivariant_vector": GNNBuilderPreset(
        name="EquivariantVectorGNN",
        encoder_type="equivariant_atomic",
        decoder_type="equivariant",
        processor_type="equivariant",
        encode_3body=False,
        equivariant_type="egnn",
        output_type="vector",
        output_level="node",
    ),
    "equivariant_scalar": GNNBuilderPreset(
        name="EquivariantScalarGNN",
        encoder_type="equivariant_atomic",
        decoder_type="equivariant",
        processor_type="equivariant",
        encode_3body=False,
        equivariant_type="egnn",
        output_type="scalar",
        output_level="graph",
    ),
    "equivariant_scalar_gradient": GNNBuilderPreset(
        name="EquivariantScalarGradientGNN",
        encoder_type="equivariant_atomic",
        decoder_type="equivariant",
        processor_type="equivariant",
        encode_3body=False,
        equivariant_type="egnn",
        output_type="scalar_gradient",
        output_level="graph",
    ),
}


def build_gnn_builder(
    *,
    encoder: Optional[nn.Module] = None,
    processor: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    encoder_type: str = "atomic",
    conv_type: str = "mesh",
    decoder_type: str = "positive",
    processor_type: str = "order",
    name: str = "GNNBuilder",
    num_species: Optional[int] = None,
    cutoff: float = 5.0,
    dim: int = 128,
    num_convs: int = 3,
    out_dim: int = 1,
    act=None,
    aggr_scheme: str = "add",
    encode_3body: bool = True,
    threebody_conv_type: Optional[str] = None,
    dihedral: bool = False,
    basis: str = "gaussian",
    combine: bool = True,
    K: int = 16,
    equivariant_type: str = "egnn",
    rbf_dim: int = 32,
    output_type: str = "scalar",
    output_level: str = "graph",
    return_dict: bool = False,
    gradient_input_key: str = "pos",
    gradient_sign: str = "negative",
    prepare_gradient: Optional[bool] = None,
    clone_gradient_input: bool = False,
    force_enable_grad: bool = True,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    conv_kwargs: Optional[Dict[str, Any]] = None,
    threebody_conv_kwargs: Optional[Dict[str, Any]] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
) -> GNNBuilder:
    """
    Build a GNNBuilder from explicit modules or string component specs.
    """
    act = build_activation(act)
    encoder_kwargs = dict(encoder_kwargs or {})
    conv_kwargs = dict(conv_kwargs or {})
    threebody_conv_kwargs = dict(threebody_conv_kwargs or {})
    processor_kwargs = dict(processor_kwargs or {})
    decoder_kwargs = dict(decoder_kwargs or {})

    if encoder is None:
        encoder = build_default_encoder(
            encoder_type,
            dim=dim,
            act=act,
            num_species=num_species,
            cutoff=cutoff,
            dihedral=dihedral,
            basis=basis,
            **encoder_kwargs,
        )

    if processor is None:
        processor = build_default_processor(
            processor_type,
            dim=dim,
            num_convs=num_convs,
            conv_type=conv_type,
            aggr_scheme=aggr_scheme,
            encode_3body=encode_3body,
            threebody_conv_type=threebody_conv_type,
            act=act,
            cutoff=cutoff,
            equivariant_type=equivariant_type,
            rbf_dim=rbf_dim,
            conv_kwargs=conv_kwargs,
            threebody_conv_kwargs=threebody_conv_kwargs,
            **processor_kwargs,
        )

    if decoder is None:
        decoder = build_default_decoder(
            decoder_type,
            dim=dim,
            out_dim=out_dim,
            act=act,
            combine=combine,
            K=K,
            output_type=output_type,
            output_level=output_level,
            return_dict=return_dict,
            gradient_input_key=gradient_input_key,
            gradient_sign=gradient_sign,
            **decoder_kwargs,
        )

    return GNNBuilder(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        name=name,
        prepare_gradient=prepare_gradient,
        gradient_input_key=gradient_input_key,
        clone_gradient_input=clone_gradient_input,
        force_enable_grad=force_enable_grad,
    )


# Backward-compatible alias.
build_generic_gnn = build_gnn_builder


def build_equivariant_gnn(**kwargs) -> EquivariantGNN:
    """
    Convenience constructor for the equivariant branch.
    """
    return EquivariantGNN(**kwargs)


def build_preset(
    preset: str,
    *,
    encoder_type: Optional[str] = None,
    conv_type: Optional[str] = None,
    decoder_type: Optional[str] = None,
    processor_type: Optional[str] = None,
    encode_3body: Optional[bool] = None,
    threebody_conv_type: Optional[str] = None,
    equivariant_type: Optional[str] = None,
    output_type: Optional[str] = None,
    output_level: Optional[str] = None,
    **kwargs,
) -> GNNBuilder:
    preset_key = str(preset).lower().strip()

    if preset_key not in PRESETS:
        raise ValueError(
            f"Unknown preset={preset!r}. Supported presets are: {sorted(PRESETS)}"
        )

    spec = PRESETS[preset_key]

    return build_gnn_builder(
        name=spec.name,
        encoder_type=encoder_type or spec.encoder_type,
        conv_type=conv_type or spec.conv_type,
        decoder_type=decoder_type or spec.decoder_type,
        processor_type=processor_type or spec.processor_type,
        encode_3body=spec.encode_3body if encode_3body is None else encode_3body,
        threebody_conv_type=(
            threebody_conv_type
            if threebody_conv_type is not None
            else spec.threebody_conv_type
        ),
        equivariant_type=equivariant_type or spec.equivariant_type,
        output_type=output_type or spec.output_type,
        output_level=output_level or spec.output_level,
        **kwargs,
    )


def build_alignn_model(**kwargs) -> GNNBuilder:
    return build_preset("alignn", **kwargs)


def build_mgn_model(**kwargs) -> GNNBuilder:
    return build_preset("mgn", **kwargs)


def build_mesh_model(**kwargs) -> GNNBuilder:
    return build_preset("mesh", **kwargs)


def build_gated_gcn_model(**kwargs) -> GNNBuilder:
    return build_preset("gated_gcn", **kwargs)


def build_gine_model(**kwargs) -> GNNBuilder:
    return build_preset("gine", **kwargs)


def build_edge_conditioned_model(**kwargs) -> GNNBuilder:
    return build_preset("edge_conditioned", **kwargs)


def build_nnconv_model(**kwargs) -> GNNBuilder:
    return build_preset("nnconv", **kwargs)


def build_pna_model(**kwargs) -> GNNBuilder:
    return build_preset("pna", **kwargs)


def build_generic_feature_model(**kwargs) -> GNNBuilder:
    return build_preset("generic_gnn", **kwargs)


def build_straight_through_model(
    conv_type: str,
    *,
    encoder_type: str = "atomic",
    decoder_type: str = "positive",
    processor_type: str = "order",
    encode_3body: bool = True,
    **kwargs,
) -> GNNBuilder:
    """
    Convenience constructor for direct conv comparisons.
    """
    return build_gnn_builder(
        encoder_type=encoder_type,
        conv_type=conv_type,
        decoder_type=decoder_type,
        processor_type=processor_type,
        encode_3body=encode_3body,
        name=f"{conv_type}_GNNBuilder",
        **kwargs,
    )


def build_model(
    *,
    preset: Optional[str] = None,
    model_type: Optional[str] = None,
    **kwargs,
) -> GNNBuilder:
    """
    Primary public model builder.

    Examples
    --------
    Legacy ALIGNN/order branch:

        model = build_model(
            preset="alignn",
            num_species=5,
            cutoff=6.0,
            dim=128,
            num_convs=3,
        )

    Equivariant vector branch:

        model = build_model(
            preset="equivariant_vector",
            dim=128,
            num_convs=4,
            cutoff=5.0,
            output_type="vector",
            output_level="node",
        )

    Explicit equivariant branch:

        model = build_model(
            encoder_type="equivariant_atomic",
            processor_type="equivariant",
            decoder_type="equivariant",
            output_type="vector",
            output_level="node",
            dim=128,
            num_convs=4,
            cutoff=5.0,
        )

    Convenience wrapper branch:

        model = build_model(
            model_type="equivariant",
            output_type="vector",
            output_level="node",
            dim=128,
            num_convs=4,
            cutoff=5.0,
        )
    """
    # Be tolerant of config dictionaries that contain these keys in kwargs.
    if preset is None and "preset" in kwargs:
        preset = kwargs.pop("preset")
    elif "preset" in kwargs:
        kwargs.pop("preset")

    if model_type is None and "model_type" in kwargs:
        model_type = kwargs.pop("model_type")
    elif "model_type" in kwargs:
        kwargs.pop("model_type")

    if model_type is not None:
        key = str(model_type).lower().strip()
        if key in {"equivariant", "egnn", "equivariant_gnn"}:
            return build_equivariant_gnn(**kwargs)
        if key in {"gnn_builder", "generic", "generic_gnn"}:
            return build_gnn_builder(**kwargs)
        raise ValueError(
            f"Unknown model_type={model_type!r}. Supported: 'gnn_builder', 'equivariant'."
        )

    if preset is not None:
        return build_preset(preset, **kwargs)

    return build_gnn_builder(**kwargs)


def build_model_from_config(config: Dict[str, Any]) -> GNNBuilder:
    """
    Build a model from a JSON/YAML-style dictionary.
    """
    config = dict(config)
    preset = config.pop("preset", None)
    model_type = config.pop("model_type", None)
    return build_model(preset=preset, model_type=model_type, **config)


# =============================================================================
# Backward-compatible wrapper classes
# =============================================================================


class ALIGNNPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_alignn_model(**kwargs)
        super().__init__(
            model.encoder,
            model.processor,
            model.decoder,
            name=model.name,
            prepare_gradient=model.prepare_gradient,
            gradient_input_key=model.gradient_input_key,
            clone_gradient_input=model.clone_gradient_input,
            force_enable_grad=model.force_enable_grad,
        )


class MeshGraphNetsPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_mgn_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class GatedGCNPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_gated_gcn_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class GINEPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_gine_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class EdgeConditionedPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_edge_conditioned_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class PNAPreset(GNNBuilder):
    def __init__(self, **kwargs):
        model = build_pna_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


__all__ = [
    "GNNBuilder",
    "GenericGNN",
    "EquivariantGNN",
    "GNNBuilderPreset",
    "GenericGNNPreset",
    "PRESETS",
    "OrderProcessor",
    "ScalarProcessor",
    "attach_order_input_aliases",
    "attach_equivariant_input_aliases",
    "attach_input_aliases",
    "attach_order_hidden_aliases",
    "attach_equivariant_hidden_aliases",
    "attach_hidden_aliases",
    "backfill_legacy_hidden_names",
    "prepare_gradient_input",
    "build_default_encoder",
    "build_default_decoder",
    "build_default_processor",
    "build_gnn_builder",
    "build_generic_gnn",
    "build_equivariant_gnn",
    "build_preset",
    "build_alignn_model",
    "build_mgn_model",
    "build_mesh_model",
    "build_gated_gcn_model",
    "build_gine_model",
    "build_edge_conditioned_model",
    "build_nnconv_model",
    "build_pna_model",
    "build_generic_feature_model",
    "build_straight_through_model",
    "build_model",
    "build_model_from_config",
    "ALIGNNPreset",
    "MeshGraphNetsPreset",
    "GatedGCNPreset",
    "GINEPreset",
    "EdgeConditionedPreset",
    "PNAPreset",
]
