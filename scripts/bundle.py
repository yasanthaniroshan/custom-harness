"""Package the project for Kaggle without git: `python scripts/bundle.py` -> harness-src.zip.

Fallback for when the notebook cannot clone the GitHub repo. Upload the zip as a Kaggle Dataset
(Datasets -> New Dataset, or `kaggle datasets version` for an update) and attach it to the
notebook. Only source, configs and metadata are included (no .venv, outputs or caches).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INCLUDED_PATHS = ["src", "configs", "scripts", "pyproject.toml", "uv.lock", "README.md"]
EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache", "outputs"}


def files() -> list[Path]:
    selected = []
    for name in INCLUDED_PATHS:
        path = PROJECT_ROOT / name
        if path.is_file():
            selected.append(path)
        elif path.is_dir():
            selected += [file for file in path.rglob("*")
                         if file.is_file() and not EXCLUDED_DIRECTORY_NAMES & set(file.parts)
                         and file.suffix != ".pyc"]
    return sorted(selected)


def main(destination: str = "harness-src.zip") -> Path:
    archive_path = PROJECT_ROOT / destination
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files():
            archive.write(file, file.relative_to(PROJECT_ROOT).as_posix())
    return archive_path


if __name__ == "__main__":
    archive = main(*sys.argv[1:2])
    print(f"wrote {archive} ({archive.stat().st_size / 1024:.0f} KiB)")
