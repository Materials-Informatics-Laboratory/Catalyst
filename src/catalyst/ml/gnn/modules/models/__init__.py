"""
Model-level exports for Catalyst GNN modules.

This package should expose model containers and model-builder utilities.
Encoders, decoders, processors, and conv layers should be imported from their
own subpackages when needed.
"""

from .generic_gnn import (
    GenericGNN,
    GenericGNNPreset,
    PRESETS,
    build_default_encoder,
    build_default_decoder,
    build_generic_gnn,
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
    "GenericGNN",
    "GenericGNNPreset",
    "PRESETS",
    "build_default_encoder",
    "build_default_decoder",
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
    "build_model_from_config",
    "ALIGNNPreset",
    "MeshGraphNetsPreset",
    "GatedGCNPreset",
    "GINEPreset",
    "EdgeConditionedPreset",
    "PNAPreset",
    "ALIGNN",
]