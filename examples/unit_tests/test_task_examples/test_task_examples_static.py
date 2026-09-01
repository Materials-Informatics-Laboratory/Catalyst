"""Static coverage checks for the full GNNTask example matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = EXAMPLES_ROOT / "gnn_examples" / "task_examples"
FORCE_EXAMPLE = EXAMPLES_ROOT / "gnn_examples" / "alignn_examples" / "force" / "al_fcc_equivariant_force_catalyst_backend.py"

EXPECTED_TASK_EXAMPLES = {
    "01_graph_scalar": ("graph_scalar_example.py", "graph_scalar"),
    "02_graph_multiscalar": ("graph_multiscalar_example.py", "graph_multiscalar"),
    "03_node_scalar": ("node_scalar_example.py", "node_scalar"),
    "04_graph_vector": ("graph_vector_example.py", "graph_vector"),
    "05_scalar_gradient": ("scalar_gradient_example.py", "scalar_gradient"),
}


def test_each_task_example_has_its_own_directory_script_and_json_config():
    for dirname, (script_name, _task_name) in EXPECTED_TASK_EXAMPLES.items():
        example_dir = TASK_DIR / dirname
        assert example_dir.is_dir()
        assert (example_dir / script_name).is_file()
        config_path = example_dir / "config.json"
        assert config_path.is_file()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert isinstance(config, dict)
        assert "task" in config
        assert "model" in config
        assert "training" in config
        assert "output_dir" in config
    assert FORCE_EXAMPLE.is_file()


def test_each_new_task_example_reads_config_builds_model_and_uses_full_backend():
    for dirname, (script_name, task_name) in EXPECTED_TASK_EXAMPLES.items():
        path = TASK_DIR / dirname / script_name
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        assert "load_example_config(" in text
        assert f"GNNTask.{task_name}(" in text
        assert "build_task_model(" in text
        assert "run_backend_task_example(" in text
        assert "plot_kind=" in text
        assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in ast.walk(tree))


def test_task_config_splits_leave_nonempty_test_sets():
    for dirname, (_script_name, _task_name) in EXPECTED_TASK_EXAMPLES.items():
        config = json.loads((TASK_DIR / dirname / "config.json").read_text(encoding="utf-8"))
        n_total = int(config["dataset"]["num_graphs"])
        n_train = int(config["training"]["n_train"])
        n_validation = int(config["training"]["n_validation"])
        assert n_train > 0
        assert n_validation > 0
        assert n_train + n_validation < n_total


def test_node_vector_is_covered_by_the_full_force_workflow():
    text = FORCE_EXAMPLE.read_text(encoding="utf-8")
    assert "GNNTask.node_vector(" in text
    assert "build_task_model(" in text
    assert "run_training" in text
    assert "run_inference" in text


def test_all_six_public_task_names_have_a_full_example_path():
    covered = {value[1] for value in EXPECTED_TASK_EXAMPLES.values()} | {"node_vector"}
    assert covered == {
        "graph_scalar",
        "graph_multiscalar",
        "node_scalar",
        "node_vector",
        "graph_vector",
        "scalar_gradient",
    }


def test_generic_workflow_is_now_generation_projection_and_sampling_only():
    path = EXAMPLES_ROOT / "gnn_examples" / "pre_training_workflow" / "example_workflow.py"
    text = path.read_text(encoding="utf-8")
    assert "generic_graph_gen" in text
    assert "SODAS" in text
    assert "run_sampling" in text
    assert "run_training" not in text
    assert "run_inference" not in text
