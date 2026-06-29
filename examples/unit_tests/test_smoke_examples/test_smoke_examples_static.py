from __future__ import annotations

from pathlib import Path
import py_compile


def test_smoke_examples_compile():
    repo = Path(__file__).resolve().parents[1]
    example_dir = repo / "examples" / "gnn_examples" / "smoke"

    for path in sorted(example_dir.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
