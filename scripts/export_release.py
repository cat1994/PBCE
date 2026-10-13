#!/usr/bin/env python3
"""Export only the verified public source, license, tests and bundled data."""
import argparse
from pathlib import Path
import zipfile

from release_files import ROOT, release_files
from verify_release import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/pbce-anonymous.zip")
    args = parser.parse_args()
    verify()
    output = args.output.resolve()
    files = release_files()
    if output in [path.resolve() for path in files]:
        parser.error("output must not replace a source/data file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                info = zipfile.ZipInfo("PBCE/" + path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Release archive: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
