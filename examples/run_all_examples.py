"""Run every user-facing Catalyst example in a deterministic order.

Usage
-----
python examples/run_all_examples.py
python examples/run_all_examples.py --continue-on-error
python examples/run_all_examples.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parent


def discover_examples() -> list[Path]:
    scripts = []
    for path in EXAMPLES_ROOT.rglob("*.py"):
        rel = path.relative_to(EXAMPLES_ROOT)
        if path == Path(__file__).resolve():
            continue
        if path.parent == EXAMPLES_ROOT and path.name.startswith("run_all_"):
            continue
        if "unit_tests" in rel.parts:
            continue
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        # Catalyst 2.2 task examples live one-per-subdirectory.  Ignore the
        # superseded flat 01_*.py ... 05_*.py files if an update was overlaid
        # without first deleting them.
        task_root = EXAMPLES_ROOT / "gnn_examples" / "task_examples"
        if path.parent == task_root and path.name[:3] in {"01_", "02_", "03_", "04_", "05_"}:
            continue
        scripts.append(path)

    def sort_key(path: Path):
        rel = path.relative_to(EXAMPLES_ROOT).as_posix()
        if "/smoke/" in f"/{rel}":
            group = 0
        elif "/task_examples/" in f"/{rel}":
            group = 1
        elif "/graph_examples/" in f"/{rel}":
            group = 2
        elif "generic_gnn_example" in rel:
            group = 3
        else:
            group = 4
        return group, rel

    return sorted(scripts, key=sort_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List discovered examples and exit.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed example and report all failures at the end.",
    )
    args = parser.parse_args()

    scripts = discover_examples()
    if args.list:
        for path in scripts:
            print(path.relative_to(EXAMPLES_ROOT))
        return 0

    failures = []
    started = time.perf_counter()
    for index, path in enumerate(scripts, start=1):
        rel = path.relative_to(EXAMPLES_ROOT)
        print("\n" + "=" * 78)
        print(f"[{index}/{len(scripts)}] Running {rel}")
        print("=" * 78)
        result = subprocess.run([sys.executable, str(path)], cwd=str(EXAMPLES_ROOT.parent))
        if result.returncode != 0:
            failures.append((rel, result.returncode))
            if not args.continue_on_error:
                break

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("Example-suite summary")
    print(f"Discovered: {len(scripts)}")
    print(f"Failures:   {len(failures)}")
    print(f"Elapsed:    {elapsed:.1f} s")
    if failures:
        for path, code in failures:
            print(f"  FAILED ({code}): {path}")
        return 1
    print("All examples completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
