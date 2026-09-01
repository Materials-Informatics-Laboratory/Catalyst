"""Regression tests for config-driven example packaging/layout."""

from __future__ import annotations

import json
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2]


def test_pre_training_workflow_ships_adjacent_json_config():
    workflow_dir = EXAMPLES_ROOT / "gnn_examples" / "pre_training_workflow"
    script = workflow_dir / "example_workflow.py"
    config = workflow_dir / "catalyst_example_config.json"

    assert script.is_file(), f"Missing pre-training workflow script: {script}"
    assert config.is_file(), f"Missing pre-training workflow config: {config}"

    payload = json.loads(config.read_text(encoding="utf-8"))
    assert payload["workflow"]["generate_graphs"] is True
    assert payload["workflow"]["project_graphs"] is True
    assert payload["workflow"]["generate_samples"] is True
    assert "catalyst_parameters" in payload


def test_pre_training_workflow_config_has_no_supervised_training_switches():
    config = (
        EXAMPLES_ROOT
        / "gnn_examples"
        / "pre_training_workflow"
        / "catalyst_example_config.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    workflow = payload["workflow"]

    obsolete = {
        "train",
        "retrain",
        "test",
        "plot_test",
        "plot_training",
        "ranking",
        "predictions",
    }
    assert obsolete.isdisjoint(workflow), (
        "The pre-training workflow is graph generation/projection/sampling only; "
        f"remove obsolete supervised switches: {sorted(obsolete & set(workflow))}"
    )


def test_pre_training_workflow_loader_batch_sizes_are_config_valid():
    config = (
        EXAMPLES_ROOT
        / "gnn_examples"
        / "pre_training_workflow"
        / "catalyst_example_config.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    batch_size = payload["catalyst_parameters"]["loader_dict"]["batch_size"]

    assert isinstance(batch_size, list)
    assert len(batch_size) == 2
    assert all(isinstance(value, int) and value > 0 for value in batch_size), (
        "Catalyst CONFIG-stage validation requires loader_dict.batch_size to contain "
        "two positive integers: [train, validation]."
    )


def test_pre_training_workflow_uses_explicit_numpy_index_size_checks():
    script = (
        EXAMPLES_ROOT
        / "gnn_examples"
        / "pre_training_workflow"
        / "example_workflow.py"
    )
    text = script.read_text(encoding="utf-8")

    assert "if test_idx:" not in text
    assert "if train_idx:" not in text
    assert "if test_idx.size > 0:" in text
    assert "if train_idx.size > 0:" in text
    assert "test_idx = np.asarray(test_idx, dtype=int).reshape(-1)" in text
    assert "remaining_idx = np.asarray(remaining_idx, dtype=int).reshape(-1)" in text
    assert "train_idx = np.asarray(train_idx, dtype=int).reshape(-1)" in text
    assert "valid_idx = np.asarray(valid_idx, dtype=int).reshape(-1)" in text
