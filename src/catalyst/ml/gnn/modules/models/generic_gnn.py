"""
Composable Catalyst GenericGNN model.

Recommended location:
    catalyst/ml/gnn/modules/models/generic_gnn.py

This file should only handle model assembly:

    data -> encoder -> processor -> decoder -> output

Processor logic lives in:
    catalyst/ml/gnn/modules/processors/order_processor.py
    catalyst/ml/gnn/modules/processors/scalar_processor.py

Conv construction / string aliases live in:
    catalyst/ml/gnn/modules/conv/factory.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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

    Basic PyG-like:
        x -> x_1
        edge_attr -> x_2

    Generic legacy:
        node_G -> x_1
        node_A -> x_2
        edge_A -> x_3

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


def attach_order_hidden_aliases(data):
    """
    Attach generic order-style hidden aliases from legacy hidden fields.
    """
    if not _has_attr(data, "h_1"):
        value = _first_attr(data, "h_atm", "h_g_node", "h_node")
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


def backfill_legacy_hidden_names(data):
    """
    Backfill old Catalyst names from order-style h_1/h_2/h_3.

    This keeps existing decoders/readout utilities working while the package is
    migrated to the new modular names.
    """
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


# =============================================================================
# Core model
# =============================================================================


class GenericGNN(nn.Module):
    """
    Generic composable GNN:

        data -> encoder -> processor -> decoder

    The encoder is responsible for converting raw graph fields into hidden
    fields. The processor updates hidden fields. The decoder maps hidden fields
    to the requested output.
    """

    def __init__(
        self,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        name: str = "GenericGNN",
    ):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.name = name

    def forward(self, data):
        data = attach_order_input_aliases(data)

        data = self.encoder(data)
        data = attach_order_hidden_aliases(data)

        data = self.processor(data)
        data = attach_order_hidden_aliases(data)
        data = backfill_legacy_hidden_names(data)

        return self.decoder(data)


# =============================================================================
# Default component builders
# =============================================================================


def _lazy_import_default_components():
    """
    Import default modular encoders/decoders.

    Expected locations:
        catalyst/ml/gnn/modules/encoders/atomic_encoders.py
        catalyst/ml/gnn/modules/decoders/standard_decoders.py
    """
    from ..encoders.atomic_encoders import AtomicGraphEncoder
    from ..encoders.atomic_encoders import GenericFeatureEncoder
    from ..encoders.atomic_encoders import Encoder_atomic
    from ..encoders.atomic_encoders import Encoder_generic

    from ..decoders.standard_decoders import ScalarDecoder
    from ..decoders.standard_decoders import Decoder
    from ..decoders.standard_decoders import PositiveScalarsDecoder
    from ..decoders.standard_decoders import PositiveKChannelDecoder
    from ..decoders.standard_decoders import PositiveFeatureReadout

    return {
        "AtomicGraphEncoder": AtomicGraphEncoder,
        "GenericFeatureEncoder": GenericFeatureEncoder,
        "Encoder_atomic": Encoder_atomic,
        "Encoder_generic": Encoder_generic,
        "ScalarDecoder": ScalarDecoder,
        "Decoder": Decoder,
        "PositiveScalarsDecoder": PositiveScalarsDecoder,
        "PositiveKChannelDecoder": PositiveKChannelDecoder,
        "PositiveFeatureReadout": PositiveFeatureReadout,
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
            raise ValueError(
                "num_species is required for the atomic/ALIGNN encoder."
            )

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

    raise ValueError(
        f"Unsupported encoder_type={encoder_type!r}. "
        "Supported defaults are: 'atomic' and 'generic'."
    )


def build_default_decoder(
    decoder_type: str,
    *,
    dim: int,
    out_dim: int = 1,
    act=None,
    combine: bool = True,
    K: int = 16,
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

    if decoder_type in {"positive_k", "positive_k_channel", "positive_kchannel"}:
        return components["PositiveKChannelDecoder"](
            dim=dim,
            act=act,
            K=K,
            **decoder_kwargs,
        )

    raise ValueError(
        f"Unsupported decoder_type={decoder_type!r}. "
        "Supported defaults are: 'scalar', 'positive', and 'positive_k'."
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
    conv_kwargs: Optional[Dict[str, Any]] = None,
    threebody_conv_kwargs: Optional[Dict[str, Any]] = None,
    **processor_kwargs,
) -> nn.Module:
    """
    Build one of the standard Catalyst processors.

    processor_type="order"
        ALIGNN-style 1/2/3-body processor. Updates h_2/h_3 through edge_index_3
        and h_1/h_2 through edge_index_2.

    processor_type="scalar"
        Simple one-graph scalar/invariant processor. Updates only h_1/h_2 through
        edge_index_2.
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

    raise ValueError(
        f"Unsupported processor_type={processor_type!r}. "
        "Supported defaults are: 'order' and 'scalar'."
    )


# =============================================================================
# Presets and factories
# =============================================================================


@dataclass(frozen=True)
class GenericGNNPreset:
    name: str
    encoder_type: str
    conv_type: str
    decoder_type: str = "positive"
    processor_type: str = "order"
    encode_3body: bool = True
    threebody_conv_type: Optional[str] = None


PRESETS: Dict[str, GenericGNNPreset] = {
    "alignn": GenericGNNPreset(
        name="ALIGNN",
        encoder_type="atomic",
        conv_type="gated_gcn",
        decoder_type="positive",
        encode_3body=True,
    ),
    "mgn": GenericGNNPreset(
        name="MeshGraphNets",
        encoder_type="atomic",
        conv_type="mesh",
        decoder_type="positive",
        encode_3body=True,
    ),
    "mesh": GenericGNNPreset(
        name="MeshGraphNets",
        encoder_type="atomic",
        conv_type="mesh",
        decoder_type="positive",
        encode_3body=True,
    ),
    "gated_gcn": GenericGNNPreset(
        name="GatedGCN",
        encoder_type="atomic",
        conv_type="gated_gcn",
        decoder_type="positive",
        encode_3body=True,
    ),
    "gine": GenericGNNPreset(
        name="GINE",
        encoder_type="atomic",
        conv_type="gine",
        decoder_type="positive",
        encode_3body=True,
    ),
    "edge_conditioned": GenericGNNPreset(
        name="EdgeConditioned",
        encoder_type="atomic",
        conv_type="edge_conditioned",
        decoder_type="positive",
        encode_3body=True,
    ),
    "nnconv": GenericGNNPreset(
        name="NNConv",
        encoder_type="atomic",
        conv_type="edge_conditioned",
        decoder_type="positive",
        encode_3body=True,
    ),
    "pna": GenericGNNPreset(
        name="PNA",
        encoder_type="atomic",
        conv_type="pna",
        decoder_type="positive",
        encode_3body=True,
    ),
    "generic_gnn": GenericGNNPreset(
        name="GenericGNN",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="order",
        encode_3body=True,
    ),
    "scalar_gnn": GenericGNNPreset(
        name="ScalarGNN",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="scalar",
        encode_3body=False,
    ),
    "scalar": GenericGNNPreset(
        name="ScalarGNN",
        encoder_type="generic",
        conv_type="mesh",
        decoder_type="scalar",
        processor_type="scalar",
        encode_3body=False,
    ),
}


def build_generic_gnn(
    *,
    encoder: Optional[nn.Module] = None,
    processor: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    encoder_type: str = "atomic",
    conv_type: str = "mesh",
    decoder_type: str = "positive",
    processor_type: str = "order",
    name: str = "GenericGNN",
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
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    conv_kwargs: Optional[Dict[str, Any]] = None,
    threebody_conv_kwargs: Optional[Dict[str, Any]] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
) -> GenericGNN:
    """
    Build a GenericGNN from modules or strings.

    You can pass explicit encoder/processor/decoder modules, or let this builder
    construct the defaults.
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
            **decoder_kwargs,
        )

    return GenericGNN(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        name=name,
    )


def build_preset(
    preset: str,
    *,
    encoder_type: Optional[str] = None,
    conv_type: Optional[str] = None,
    decoder_type: Optional[str] = None,
    processor_type: Optional[str] = None,
    encode_3body: Optional[bool] = None,
    threebody_conv_type: Optional[str] = None,
    **kwargs,
) -> GenericGNN:
    preset_key = str(preset).lower().strip()

    if preset_key not in PRESETS:
        raise ValueError(
            f"Unknown preset={preset!r}. Supported presets are: {sorted(PRESETS)}"
        )

    spec = PRESETS[preset_key]

    return build_generic_gnn(
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
        **kwargs,
    )


def build_alignn_model(**kwargs) -> GenericGNN:
    return build_preset("alignn", **kwargs)


def build_mgn_model(**kwargs) -> GenericGNN:
    return build_preset("mgn", **kwargs)


def build_mesh_model(**kwargs) -> GenericGNN:
    return build_preset("mesh", **kwargs)


def build_gated_gcn_model(**kwargs) -> GenericGNN:
    return build_preset("gated_gcn", **kwargs)


def build_gine_model(**kwargs) -> GenericGNN:
    return build_preset("gine", **kwargs)


def build_edge_conditioned_model(**kwargs) -> GenericGNN:
    return build_preset("edge_conditioned", **kwargs)


def build_nnconv_model(**kwargs) -> GenericGNN:
    return build_preset("nnconv", **kwargs)


def build_pna_model(**kwargs) -> GenericGNN:
    return build_preset("pna", **kwargs)


def build_generic_feature_model(**kwargs) -> GenericGNN:
    return build_preset("generic_gnn", **kwargs)


def build_straight_through_model(
    conv_type: str,
    *,
    encoder_type: str = "atomic",
    decoder_type: str = "positive",
    processor_type: str = "order",
    encode_3body: bool = True,
    **kwargs,
) -> GenericGNN:
    """
    Convenience constructor for direct conv comparisons.
    """
    return build_generic_gnn(
        encoder_type=encoder_type,
        conv_type=conv_type,
        decoder_type=decoder_type,
        processor_type=processor_type,
        encode_3body=encode_3body,
        name=f"{conv_type}_GenericGNN",
        **kwargs,
    )


def build_model(
    *,
    preset: Optional[str] = None,
    **kwargs,
) -> GenericGNN:
    """
    Primary public model builder.

    This is the preferred user-facing constructor for examples and normal use.

    Examples
    --------
    Fully explicit modular model:

        model = build_model(
            encoder_type="generic",
            conv_type="mesh",
            decoder_type="positive",
            processor_type="scalar",
            dim=128,
            num_convs=3,
        )

    Preset model:

        model = build_model(
            preset="alignn",
            num_species=5,
            cutoff=6.0,
            dim=128,
            num_convs=3,
        )

    Important
    ---------
    The ``preset`` keyword is consumed here. It must not be forwarded into
    build_generic_gnn(...), because build_generic_gnn does not accept a preset
    argument.
    """
    # Be tolerant of config dictionaries that still contain a preset key inside
    # kwargs, e.g. build_model(**{"preset": "alignn", ...}).
    if preset is None and "preset" in kwargs:
        preset = kwargs.pop("preset")
    elif "preset" in kwargs:
        # Avoid forwarding duplicate/unknown preset into build_generic_gnn.
        kwargs.pop("preset")

    if preset is not None:
        return build_preset(preset, **kwargs)

    return build_generic_gnn(**kwargs)


def build_model_from_config(config: Dict[str, Any]) -> GenericGNN:
    """
    Build a model from a JSON/YAML-style dictionary.

    This accepts either:
        {"preset": "alignn", ...}
    or:
        {"encoder_type": "generic", "conv_type": "mesh", ...}
    """
    config = dict(config)
    preset = config.pop("preset", None)
    return build_model(preset=preset, **config)


# =============================================================================
# Backward-compatible wrapper classes
# =============================================================================


class ALIGNNPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_alignn_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class MeshGraphNetsPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_mgn_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class GatedGCNPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_gated_gcn_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class GINEPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_gine_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class EdgeConditionedPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_edge_conditioned_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


class PNAPreset(GenericGNN):
    def __init__(self, **kwargs):
        model = build_pna_model(**kwargs)
        super().__init__(model.encoder, model.processor, model.decoder, name=model.name)


__all__ = [
    "GenericGNN",
    "GenericGNNPreset",
    "PRESETS",
    "OrderProcessor",
    "ScalarProcessor",
    "attach_order_input_aliases",
    "attach_order_hidden_aliases",
    "backfill_legacy_hidden_names",
    "build_default_encoder",
    "build_default_decoder",
    "build_default_processor",
    "build_generic_gnn",
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
