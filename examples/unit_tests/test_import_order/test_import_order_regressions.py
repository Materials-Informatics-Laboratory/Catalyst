"""Subprocess import-order tests guarding against circular-import regressions."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "code",
    [
        "import catalyst.properties; import catalyst.utilities; from catalyst.graph.alignnd import alignn_gen",
        "import catalyst.utilities; import catalyst.properties; from catalyst.graph.alignnd import alignn_gen",
        "import catalyst.ml.utils.loss; import catalyst.ml.gnn; from catalyst.ml.gnn.GNN import GNN",
        "import catalyst.ml.gnn; import catalyst.ml.utils.loss; from catalyst.ml.gnn.GNN import GNN",
    ],
)
def test_supported_import_orders_do_not_trigger_circular_imports(code):
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
