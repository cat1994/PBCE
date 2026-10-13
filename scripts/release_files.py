"""Explicit publication scope shared by verification and archive creation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "LICENSE.txt", "THIRD_PARTY_NOTICES.md", "CMakeLists.txt",
              "Makefile", ".gitignore", ".gitattributes", "requirements-plot.txt", "VALIDATION.md")
SOURCE_DIRS = ("src", "bin", "pbce", "tests", "scripts", "docs", "games", "data")


def release_files():
    files = [ROOT / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        for path in (ROOT / name).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Release must not contain symlinks: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo", ".tmp", ".log"}:
                continue
            # Only the curated data package is public; local runs are excluded.
            if name == "data" and relative.parts[1] not in {"raw", "README.md", "manifest.json"}:
                continue
            files.append(path)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or invalid release file: {path}")
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())
