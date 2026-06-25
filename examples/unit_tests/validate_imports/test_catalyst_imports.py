#!/usr/bin/env python
"""
Catalyst import smoke test.

Purpose
-------
This script checks whether a local Catalyst install is importable on a user's
machine. It is designed to catch common installation/package-layout problems,
for example:

    - `import catalyst` imports the wrong package.
    - `from catalyst.graph.alignnd import alignn_gen` fails.
    - Subpackages are missing because `__init__.py` files or setup.cfg are wrong.
    - Required dependencies are missing from the active Python environment.
    - Checkpoint/graph examples accidentally import from `catalyst.src...`.

Usage
-----
Run directly:

    python test_catalyst_imports.py

Run with pytest:

    pytest -q test_catalyst_imports.py

Optional deeper test:

    python test_catalyst_imports.py --include-examples --strict

Notes
-----
By default, this script skips Catalyst examples because example modules may load
configs, read data, train models, or execute workflow code at import time. Use
`--include-examples` only when you want a deeper source-tree test.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional


# These are the core imports that should work in a modern Catalyst install.
CRITICAL_IMPORTS = [
    "catalyst",
    "catalyst.graph",
    "catalyst.graph.graph",
    "catalyst.graph.alignnd",
    "catalyst.graph.generic_build",
    "catalyst.data.utils",
    "catalyst.observer.params",
    "catalyst.utilities.sampling",
    "catalyst.ml.gnn.GNN",
    "catalyst.ml.gnn.modules.models.alignn",
    "catalyst.ml.training",
    "catalyst.ml.inference",
    "catalyst.characterization.sodas.model.sodas",
]


# These catch cases where modules import but expected public names are missing.
CRITICAL_SYMBOLS = [
    ("catalyst.graph.alignnd", "alignn_gen"),
    ("catalyst.graph.alignnd", "alignnd"),
    ("catalyst.graph.alignnd", "realignnd"),
    ("catalyst.graph.alignnd", "atomic_alignnd"),
    ("catalyst.graph.generic_build", "generic_graph_gen"),
    ("catalyst.graph.graph", "Atomic_Graph_Data"),
    ("catalyst.graph.graph", "Generic_Graph_Data"),
    ("catalyst.graph.graph", "line_graph"),
    ("catalyst.ml.gnn.GNN", "GNN"),
    ("catalyst.ml.gnn.modules.models.alignn", "ALIGNN"),
    ("catalyst.ml.gnn.modules.models.alignn", "Encoder_atomic"),
    ("catalyst.ml.gnn.modules.models.alignn", "Processor"),
    ("catalyst.ml.inference", "run_inference"),
    ("catalyst.ml.training", "run_training"),
    ("catalyst.characterization.sodas.model.sodas", "SODAS"),
    ("catalyst.observer.params", "Catalyst"),
]


DEFAULT_EXCLUDE_PATTERNS = [
    # Avoid workflow/example side effects unless requested.
    r"(^|\.)examples(\.|$)",
    r"(^|\.)Example(\.|$)",
    r"(^|\.)examples_",
    # Avoid notebooks or generated scratch modules if present.
    r"(^|\.)notebooks?(\.|$)",
    r"(^|\.)scratch(\.|$)",
    # Avoid tests while running this test.
    r"(^|\.)tests?(\.|$)",
]


@dataclass
class ImportResult:
    name: str
    ok: bool
    seconds: float
    category: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    missing_module: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class SymbolResult:
    module: str
    symbol: str
    ok: bool
    error_message: Optional[str] = None


def _looks_like_missing_optional_dependency(exc: BaseException) -> tuple[bool, Optional[str]]:
    """
    Return (is_optional_dependency_failure, missing_module_name).

    ModuleNotFoundError can mean either:
      - Catalyst itself is packaged incorrectly, e.g. no catalyst.graph.
      - An external dependency is missing, e.g. torch_scatter.

    Missing `catalyst...` modules are classified as internal failures.
    Anything else is classified as a missing external/optional dependency.
    """
    if not isinstance(exc, ModuleNotFoundError):
        return False, None

    missing_name = getattr(exc, "name", None)
    if not missing_name:
        return False, None

    if missing_name == "catalyst" or missing_name.startswith("catalyst."):
        return False, missing_name

    return True, missing_name


def import_one(name: str) -> ImportResult:
    start = time.perf_counter()
    try:
        importlib.import_module(name)
        seconds = time.perf_counter() - start
        return ImportResult(
            name=name,
            ok=True,
            seconds=seconds,
            category="ok",
        )
    except BaseException as exc:  # noqa: BLE001 - intentional smoke-test capture
        seconds = time.perf_counter() - start
        is_missing_dependency, missing_name = _looks_like_missing_optional_dependency(exc)
        category = "missing_dependency" if is_missing_dependency else "import_failure"

        return ImportResult(
            name=name,
            ok=False,
            seconds=seconds,
            category=category,
            error_type=type(exc).__name__,
            error_message=str(exc),
            missing_module=missing_name,
            traceback_text=traceback.format_exc(),
        )


def check_symbol(module_name: str, symbol_name: str) -> SymbolResult:
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol_name):
            return SymbolResult(
                module=module_name,
                symbol=symbol_name,
                ok=False,
                error_message=f"Missing symbol: {module_name}.{symbol_name}",
            )
        return SymbolResult(module=module_name, symbol=symbol_name, ok=True)
    except BaseException as exc:  # noqa: BLE001
        return SymbolResult(
            module=module_name,
            symbol=symbol_name,
            ok=False,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def discover_catalyst_modules(
    catalyst_pkg: ModuleType,
    include_examples: bool = False,
    extra_exclude_regex: Optional[str] = None,
) -> list[str]:
    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
    if include_examples:
        exclude_patterns = [
            pattern
            for pattern in exclude_patterns
            if "examples" not in pattern and "Example" not in pattern
        ]

    if extra_exclude_regex:
        exclude_patterns.append(extra_exclude_regex)

    compiled_patterns = [re.compile(pattern) for pattern in exclude_patterns]

    module_names: list[str] = []
    for module_info in pkgutil.walk_packages(
        catalyst_pkg.__path__,
        prefix=f"{catalyst_pkg.__name__}.",
    ):
        name = module_info.name

        if any(pattern.search(name) for pattern in compiled_patterns):
            continue

        module_names.append(name)

    return sorted(set(module_names))


def print_environment(catalyst_pkg: ModuleType) -> None:
    print("=" * 88)
    print("Catalyst import smoke test")
    print("=" * 88)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Working directory : {Path.cwd()}")
    print(f"catalyst.__file__ : {getattr(catalyst_pkg, '__file__', None)}")
    print(f"catalyst.__path__ : {list(getattr(catalyst_pkg, '__path__', []))}")
    print(f"catalyst version  : {getattr(catalyst_pkg, '__version__', '<not defined>')}")
    print("=" * 88)


def print_import_table(title: str, results: list[ImportResult], max_failures_to_show: int = 50) -> None:
    ok_count = sum(result.ok for result in results)
    fail_count = len(results) - ok_count
    dependency_count = sum(result.category == "missing_dependency" for result in results)
    internal_count = sum(result.category == "import_failure" for result in results)

    print()
    print(title)
    print("-" * 88)
    print(f"Total: {len(results)} | OK: {ok_count} | Failed: {fail_count}")
    if fail_count:
        print(f"Failure split: internal/import={internal_count}, missing_dependency={dependency_count}")

    failures = [result for result in results if not result.ok]
    if not failures:
        return

    print()
    print("Failures:")
    for result in failures[:max_failures_to_show]:
        print(f"  [{result.category}] {result.name}")
        print(f"    {result.error_type}: {result.error_message}")
        if result.missing_module:
            print(f"    missing module: {result.missing_module}")

    if len(failures) > max_failures_to_show:
        print(f"  ... {len(failures) - max_failures_to_show} more failures not shown")


def print_symbol_table(results: list[SymbolResult]) -> None:
    ok_count = sum(result.ok for result in results)
    fail_count = len(results) - ok_count

    print()
    print("Critical symbol checks")
    print("-" * 88)
    print(f"Total: {len(results)} | OK: {ok_count} | Failed: {fail_count}")

    for result in results:
        if not result.ok:
            print(f"  [missing] {result.module}.{result.symbol}")
            print(f"    {result.error_message}")


def write_report(
    path: Path,
    critical_results: list[ImportResult],
    walk_results: list[ImportResult],
    symbol_results: list[SymbolResult],
) -> None:
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "working_directory": str(Path.cwd()),
        "critical_imports": [asdict(result) for result in critical_results],
        "walk_imports": [asdict(result) for result in walk_results],
        "critical_symbols": [asdict(result) for result in symbol_results],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print()
    print(f"Wrote detailed JSON report to: {path}")


def run_smoke_test(
    include_examples: bool = False,
    strict: bool = False,
    walk_package: bool = True,
    extra_exclude_regex: Optional[str] = None,
    report_path: Optional[Path] = None,
) -> int:
    importlib.invalidate_caches()

    try:
        catalyst_pkg = importlib.import_module("catalyst")
    except BaseException as exc:  # noqa: BLE001
        print("FAILED: Could not import top-level package `catalyst`.")
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Check that you installed from the Catalyst repository root, for example:")
        print("    python -m pip install -e .")
        return 1

    print_environment(catalyst_pkg)

    critical_results = [import_one(name) for name in CRITICAL_IMPORTS]
    print_import_table("Critical import checks", critical_results)

    symbol_results = [check_symbol(module, symbol) for module, symbol in CRITICAL_SYMBOLS]
    print_symbol_table(symbol_results)

    walk_results: list[ImportResult] = []
    if walk_package:
        module_names = discover_catalyst_modules(
            catalyst_pkg,
            include_examples=include_examples,
            extra_exclude_regex=extra_exclude_regex,
        )

        print()
        print("Package discovery")
        print("-" * 88)
        print(f"Discovered {len(module_names)} Catalyst modules to import.")
        if not include_examples:
            print("Examples are skipped by default. Use --include-examples to include them.")

        walk_results = [import_one(name) for name in module_names]
        print_import_table("Full package-walk import checks", walk_results)

    if report_path is not None:
        write_report(report_path, critical_results, walk_results, symbol_results)

    critical_failed = any(not result.ok for result in critical_results)
    symbols_failed = any(not result.ok for result in symbol_results)
    internal_walk_failed = any(result.category == "import_failure" for result in walk_results)
    dependency_walk_failed = any(result.category == "missing_dependency" for result in walk_results)

    print()
    print("Summary")
    print("-" * 88)

    if critical_failed or symbols_failed:
        print("FAILED: One or more critical Catalyst imports/symbols failed.")
        return 1

    if internal_walk_failed:
        print("FAILED: One or more Catalyst package modules failed for internal reasons.")
        return 1

    if strict and dependency_walk_failed:
        print("FAILED: Missing optional/external dependencies found and --strict was set.")
        return 1

    if dependency_walk_failed:
        print("PASSED with missing optional/external dependencies.")
        print("Use --strict if missing optional dependencies should fail the test.")
        return 0

    print("PASSED: Catalyst imports look healthy.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a local Catalyst install.")
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Also import Catalyst example modules. Off by default to avoid workflow side effects.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any external/optional dependency is missing.",
    )
    parser.add_argument(
        "--no-walk",
        action="store_true",
        help="Only run the critical import/symbol checks, not the full package walk.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Additional regex pattern for modules to skip during the package walk.",
    )
    parser.add_argument(
        "--report",
        default="catalyst_import_report.json",
        help="Path to write a detailed JSON report. Use '' to disable.",
    )

    args = parser.parse_args()

    report_path = Path(args.report) if args.report else None
    return run_smoke_test(
        include_examples=args.include_examples,
        strict=args.strict,
        walk_package=not args.no_walk,
        extra_exclude_regex=args.exclude_regex,
        report_path=report_path,
    )


# Pytest entry point.
def test_catalyst_import_smoke() -> None:
    exit_code = run_smoke_test(
        include_examples=False,
        strict=False,
        walk_package=True,
        report_path=None,
    )
    assert exit_code == 0


if __name__ == "__main__":
    raise SystemExit(main())
