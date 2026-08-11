from __future__ import annotations

from pathlib import Path
import py_compile
import runpy

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_DIR = REPO_ROOT / "examples" / "gnn_examples" / "smoke"
SMOKE_SCRIPTS = sorted(SMOKE_DIR.glob("*.py"))


def test_smoke_examples_are_discovered():
    names = [path.name for path in SMOKE_SCRIPTS]
    assert names == [
        "01_generic_graph_scalar_smoke.py",
        "02_al_fcc_alignn_graph_scalar_smoke.py",
        "03_al_fcc_equivariant_node_vector_smoke.py",
        "04_al_fcc_alignn_graph_multiscalar_smoke.py",
        "05_al_fcc_train_checkpoint_inference_smoke.py",
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
