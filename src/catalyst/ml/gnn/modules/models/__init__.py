"""
Model-level exports for Catalyst GNN modules.

This package exposes model containers and model-builder utilities.

Encoders, decoders, processors, and convolution layers should generally be
imported from their own subpackages when needed.  The model-level entry point is
now ``gnn_builder.py``:

    GNNBuilder
        Generic encoder -> processor -> decoder model container.

    EquivariantGNN
        Convenience wrapper around GNNBuilder for equivariant models.

    build_model(...)
        Primary public model-construction function.

The older GenericGNN names are still exported as compatibility aliases.
"""

from .gnn_builder import (
    GNNBuilder,
    GenericGNN,
    EquivariantGNN,
    GNNBuilderPreset,
    GenericGNNPreset,
    PRESETS,
    OrderProcessor,
    ScalarProcessor,
    attach_order_input_aliases,
    attach_equivariant_input_aliases,
    attach_input_aliases,
    attach_order_hidden_aliases,
    attach_equivariant_hidden_aliases,
    attach_hidden_aliases,
    backfill_legacy_hidden_names,
    prepare_gradient_input,
    build_default_encoder,
    build_default_decoder,
    build_default_processor,
    build_gnn_builder,
    build_generic_gnn,
    build_equivariant_gnn,
    build_preset,
    build_alignn_model,
    build_mgn_model,
    build_mesh_model,
    build_gated_gcn_model,
    build_gine_model,
    build_edge_conditioned_model,
    build_nnconv_model,
    build_pna_model,
    build_generic_feature_model,
    build_straight_through_model,
    build_model,
    build_model_from_config,
    ALIGNNPreset,
    MeshGraphNetsPreset,
    GatedGCNPreset,
    GINEPreset,
    EdgeConditionedPreset,
    PNAPreset,
)

# Optional compatibility import.
# Keep this only if models/alignn.py still exists as a compatibility shim.
try:
    from .alignn import ALIGNN
except ImportError:
    ALIGNN = None


__all__ = [
    # New primary names.
    "GNNBuilder",
    "GNNBuilderPreset",
    "EquivariantGNN",

    # Backward-compatible names.
    "GenericGNN",
    "GenericGNNPreset",

    # Presets / processors.
    "PRESETS",
    "OrderProcessor",
    "ScalarProcessor",

    # Alias helpers.
    "attach_order_input_aliases",
    "attach_equivariant_input_aliases",
    "attach_input_aliases",
    "attach_order_hidden_aliases",
    "attach_equivariant_hidden_aliases",
    "attach_hidden_aliases",
    "backfill_legacy_hidden_names",
    "prepare_gradient_input",

    # Component builders.
    "build_default_encoder",
    "build_default_decoder",
    "build_default_processor",

    # Model builders.
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

    # Preset wrapper classes.
    "ALIGNNPreset",
    "MeshGraphNetsPreset",
    "GatedGCNPreset",
    "GINEPreset",
    "EdgeConditionedPreset",
    "PNAPreset",

    # Optional legacy ALIGNN shim.
    "ALIGNN",
]
