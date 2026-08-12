from __future__ import annotations

from pathlib import Path
import py_compile
import runpy

import pytest


# Anchor to the examples tree rather than assuming a repository checkout.
# This supports both repo/examples/unit_tests/... and standalone unit_tests/... layouts.
EXAMPLES_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = EXAMPLES_ROOT / "gnn_examples" / "smoke"
SMOKE_SCRIPTS = sorted(SMOKE_DIR.glob("*.py"))


def test_smoke_examples_are_discovered():
    names = [path.name for path in SMOKE_SCRIPTS]
    assert names == [
        "01_generic_graph_scalar_smoke.py",
        "02_al_fcc_alignn_graph_scalar_smoke.py",
        "03_al_fcc_equivariant_node_vector_smoke.py",
        "04_al_fcc_alignn_graph_multiscalar_smoke.py",
        "05_al_fcc_train_checkpoint_inference_smoke.py",
        "06_generic_node_scalar_smoke.py",
        "07_equivariant_graph_vector_smoke.py",
        "08_equivariant_scalar_gradient_smoke.py",
    ]


@pytest.mark.parametrize("path", SMOKE_SCRIPTS, ids=lambda p: p.stem)
def test_smoke_example_compiles(path: Path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", SMOKE_SCRIPTS, ids=lambda p: p.stem)
def test_smoke_example_executes(path: Path):
    # runpy keeps all smoke scripts in the same pytest process. This avoids
    # recompiling Numba-backed graph helpers in a fresh subprocess for every
    # example while still executing each file exactly as `python file.py` would.
    runpy.run_path(str(path), run_name="__main__")
