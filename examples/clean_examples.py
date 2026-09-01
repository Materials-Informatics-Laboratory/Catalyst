from pathlib import Path

# Put this script directly inside the Catalyst "examples" directory.
# Running it immediately deletes everything recursively except .py and .json files.

ROOT = Path(__file__).resolve().parent
KEEP_EXTENSIONS = {".py", ".json"}

if ROOT.name.lower() != "examples":
    raise RuntimeError(
        f"This script must be located directly inside the examples directory.\n"
        f"Current directory: {ROOT}"
    )

files_removed = 0
directories_removed = 0
bytes_removed = 0

print("=" * 78)
print("Cleaning Catalyst examples directory")
print(f"Root: {ROOT}")
print("Keeping only .py and .json files")
print("=" * 78)

# Remove all files except Python and JSON.
for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    if path.suffix.lower() in KEEP_EXTENSIONS:
        continue

    try:
        bytes_removed += path.stat().st_size
    except OSError:
        pass

    print(f"Removing: {path.relative_to(ROOT)}")
    path.unlink()
    files_removed += 1

# Remove directories that are now empty, deepest first.
directories = sorted(
    [path for path in ROOT.rglob("*") if path.is_dir()],
    key=lambda path: len(path.parts),
    reverse=True,
)

for directory in directories:
    try:
        if not any(directory.iterdir()):
            print(f"Removing empty directory: {directory.relative_to(ROOT)}")
            directory.rmdir()
            directories_removed += 1
    except FileNotFoundError:
        pass

print("=" * 78)
print("Cleanup complete")
print(f"Files removed: {files_removed}")
print(f"Empty directories removed: {directories_removed}")
print(f"Data removed: {bytes_removed / (1024 ** 2):.2f} MiB")
print("=" * 78)
