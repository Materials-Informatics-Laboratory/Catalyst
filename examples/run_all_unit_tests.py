"""Convenience wrapper for running the complete Catalyst example unit-test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parent


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(EXAMPLES_ROOT / "unit_tests"),
        "-v",
    ]
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=str(EXAMPLES_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
