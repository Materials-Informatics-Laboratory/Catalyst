"""
Model-level exports for Catalyst GNN modules.

This package exposes model containers and model-builder utilities.

Primary entry points:
    build_model(...)
    build_task_model(...) from catalyst.ml.gnn.tasks
    GNNBuilder
    EquivariantGNN
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

from .alignn import ALIGNN

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

__all__.append("ALIGNN")
