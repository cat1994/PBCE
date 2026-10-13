#!/usr/bin/env python3
"""Verify the complete source/data inventory and lossless raw-data hashes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from release_files import ROOT, release_files


def digest_stream(stream):
    value = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
        size += len(chunk)
    return value.hexdigest(), size


def digest(path):
    with path.open("rb") as stream:
        return digest_stream(stream)[0]


def verify(full=False):
    entries = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if name in entries:
            raise ValueError(f"Duplicate checksum entry: {name}")
        entries[name] = checksum
    files = release_files()
    actual = {path.relative_to(ROOT).as_posix(): path for path in files}
    if set(actual) != set(entries):
        raise ValueError(f"Release inventory differs: {sorted(set(actual) ^ set(entries))}")
    for name, path in actual.items():
        if digest(path) != entries[name]:
            raise ValueError(f"SHA256 mismatch: {name}")
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
        if digest(path) != entry["gzip_sha256"] or path.stat().st_size != entry["gzip_bytes"]:
            raise ValueError(f"Compressed data mismatch: {path.name}")
        with gzip.open(path, "rb") as stream:
            checksum, size = digest_stream(stream)
        if checksum != entry["source_sha256"] or size != entry["source_bytes"]:
            raise ValueError(f"Raw data mismatch: {path.name}")
        if full:
            if entry["mode"] == "fixed":
                _, metadata = plot.fixed_records(path)
                if metadata["rounds"] != entry["iteration_count"]:
                    raise ValueError(f"Round count mismatch: {path.name}")
            else:
                plot.adaptive_records(path)
    print(f"Verified {len(files)} release files and all 35 lossless data archives" + ("; full parsing passed" if full else ""))


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
