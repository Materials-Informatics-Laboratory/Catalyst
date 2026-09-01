from __future__ import annotations

from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2]


def _user_facing_example_files() -> list[Path]:
    files: list[Path] = []
    for path in EXAMPLES_ROOT.rglob("*.py"):
        if "unit_tests" in path.parts:
            continue
        if path.name == "__init__.py":
            continue
        files.append(path)
    return sorted(files)


def test_examples_never_open_interactive_matplotlib_windows() -> None:
    offenders = []
    for path in _user_facing_example_files():
        text = path.read_text(encoding="utf-8")
        if "plt.show(" in text:
            offenders.append(str(path.relative_to(EXAMPLES_ROOT)))

    assert not offenders, (
        "User-facing examples must save figures instead of opening interactive "
        f"Matplotlib windows. Offenders: {offenders}"
    )


def test_plotting_examples_use_agg_and_figures_directory() -> None:
    offenders = []

    for path in _user_facing_example_files():
        text = path.read_text(encoding="utf-8")
        if "import matplotlib.pyplot as plt" not in text:
            continue

        pyplot_index = text.index("import matplotlib.pyplot as plt")
        agg_marker = 'matplotlib.use("Agg")'

        if agg_marker not in text or text.index(agg_marker) > pyplot_index:
            offenders.append(
                f"{path.relative_to(EXAMPLES_ROOT)}: Agg backend is not set before pyplot"
            )

        if "figures" not in text:
            offenders.append(
                f"{path.relative_to(EXAMPLES_ROOT)}: no figures directory is defined"
            )

    assert not offenders, "\n".join(offenders)
