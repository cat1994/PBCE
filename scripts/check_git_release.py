#!/usr/bin/env python3
"""Read-only audit of Git publication scope against the verified release files."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


def git(root, *args):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise ValueError(f"Git {args[0]} failed; use a non-bare repository root")
    return result.stdout


def names(raw):
    return {os.fsdecode(name) for name in raw.split(b"\0") if name}


def check_repository(root, expected_files, staged=False):
    """Check a repository without staging, committing, or changing Git settings.

    expected_files contains repository-relative POSIX paths, including SHA256SUMS.
    In worktree mode, deleted tracked files are treated as planned removals.
    Staged mode requires index contents to match the worktree under Git's normal
    text/line-ending conversion rules, rather than comparing raw CRLF bytes.
    """
    root = Path(root)
    expected = set(expected_files)
    if not expected or any(
        not name or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts or name != PurePosixPath(name).as_posix()
        for name in expected
    ):
        raise ValueError("Expected inventory must contain normalized relative file paths")
    if git(root, "rev-parse", "--is-inside-work-tree").strip() != b"true":
        raise ValueError("Use a non-bare repository root")
    if git(root, "rev-parse", "--show-prefix").strip():
        raise ValueError("Run the audit against the repository root, not a subdirectory")

    tracked = set()
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, _object_id, stage = header.split()
        name = os.fsdecode(raw_name)
        if stage != b"0":
            raise ValueError(f"Unresolved index conflict: {name}")
        if mode not in {b"100644", b"100755"}:
            raise ValueError(f"Symlinks and gitlinks are not publishable: {name}")
        tracked.add(name)

    if staged:
        actual = tracked
    else:
        untracked = names(git(root, "ls-files", "--others", "--exclude-standard", "-z"))
        actual = {name for name in tracked | untracked if os.path.lexists(root / name)}

    extra, missing = actual - expected, expected - actual
    if extra or missing:
        details = []
        if extra:
            details.append("unexpected files: " + ", ".join(sorted(extra)))
        if missing:
            details.append("missing files: " + ", ".join(sorted(missing)))
        raise ValueError("Publication inventory differs; " + "; ".join(details))
    for name in sorted(expected):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular worktree file: {name}")
        # A symlinked parent can otherwise make a regular file escape the tree.
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError(f"Symlinked worktree directory: {name}")

    if staged:
        # These flags can hide modified worktree files from git diff. Refuse
        # them rather than silently trusting an incomplete content comparison.
        for entry in git(root, "ls-files", "-v", "-z").split(b"\0"):
            if entry and entry[:1] != b"H":
                raise ValueError("Unsupported index visibility flag: " + os.fsdecode(entry[2:]))
        # The exact index inventory above is already checked, so no enormous
        # per-file pathspec or one-process-per-file loop is necessary.
        changed = names(git(root, "diff", "--no-ext-diff", "--no-textconv",
                            "--ignore-submodules=none", "--name-only", "-z"))
        if changed:
            raise ValueError("Index differs from verified worktree: " + ", ".join(sorted(changed)))
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Audit the current Git index and its contents")
    args = parser.parse_args()
    try:
        from release_files import ROOT, release_files
        from verify_release import verify

        verify()
        expected = {path.relative_to(ROOT).as_posix() for path in release_files()}
        expected.add("SHA256SUMS")
        count = check_repository(ROOT, expected, args.staged)
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"Git release preflight failed: {error}")
    if args.staged:
        print(f"Verified current index: {count} files match the verified worktree.")
    else:
        print(f"Verified candidate worktree: {count} files; this is not a staged or committed release.")
    print("Scope/content check only: Git history, secrets and hosting anonymity are not audited.")


if __name__ == "__main__":
    main()
