#!/usr/bin/env python3
"""Check publication files and readable data records without source checksums."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

from release_files import ROOT, release_files


def verify(full=False):
    files = release_files()
    actual = {path.relative_to(ROOT).as_posix(): path for path in files}
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    listed = {"data/" + entry["path"] for entry in manifest["files"]}
    packaged = {name for name in actual if name.startswith("data/raw/")}
    if listed != packaged or len(listed) != 35:
        raise ValueError("Expected exactly the 30 fixed and 5 adaptive manifest files")
    if full:
        spec = importlib.util.spec_from_file_location("plot_pbce", ROOT / "bin/plot_PBCE.py")
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
    for entry in manifest["files"]:
        path = ROOT / "data" / entry["path"]
        if path.stat().st_size != entry["gzip_bytes"]:
            raise ValueError(f"Compressed data mismatch: {path.name}")
        with gzip.open(path, "rb") as stream:
            size = sum(len(chunk) for chunk in iter(lambda: stream.read(1024 * 1024), b""))
        if size != entry["source_bytes"]:
            raise ValueError(f"Raw data mismatch: {path.name}")
        if full:
            if entry["mode"] == "fixed":
                _, metadata = plot.fixed_records(path)
                if metadata["rounds"] != entry["iteration_count"]:
                    raise ValueError(f"Round count mismatch: {path.name}")
            else:
                plot.adaptive_records(path)
    print(f"Checked {len(files)} publication files and all 35 readable data archives" + ("; full parsing passed" if full else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also parse record structure, fixed/checkpoint metrics and run completion")
    args = parser.parse_args()
    try:
        verify(args.full)
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"Release verification failed: {error}")


if __name__ == "__main__":
    main()
