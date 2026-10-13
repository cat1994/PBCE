#!/usr/bin/env python3
"""Exercise publication preflight in isolated, disposable Git repositories."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "check_git_release", Path(__file__).resolve().parents[1] / "scripts/check_git_release.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class GitReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.command("init", "--quiet")
        self.command("config", "core.autocrlf", "false")
        self.command("config", "core.filemode", "false")

    def command(self, *args, input=None):
        return subprocess.run(["git", "-C", str(self.root), *args], input=input,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout

    def write(self, name, content="content\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def check(self, expected=("file.txt",), staged=False):
        return PREFLIGHT.check_repository(self.root, expected, staged)

    def test_empty_repository_reports_missing(self):
        with self.assertRaisesRegex(ValueError, "missing files: file.txt"):
            self.check()

    def test_untracked_candidate_is_not_staged(self):
        self.write("file.txt")
        self.assertEqual(self.check(), 1)
        with self.assertRaisesRegex(ValueError, "missing files"):
            self.check(staged=True)

    def test_extra_untracked_file_is_rejected(self):
        self.write("file.txt")
        self.write("private-notes.txt")
        with self.assertRaisesRegex(ValueError, "unexpected files: private-notes.txt"):
            self.check()

    def test_ignored_build_artifact_is_excluded(self):
        self.write("file.txt")
        self.write(".gitignore", "build/\n")
        self.write("build/local.bin")
        self.assertEqual(self.check(("file.txt", ".gitignore")), 2)

    def test_tracked_ignored_artifact_is_rejected(self):
        self.write("file.txt")
        self.write("build/local.bin")
        self.command("add", "file.txt", "build/local.bin")
        self.write(".gitignore", "build/\n")
        with self.assertRaisesRegex(ValueError, "unexpected files: build/local.bin"):
            self.check(("file.txt", ".gitignore"))

    def test_tracked_missing_required_file_is_rejected(self):
        self.write("file.txt")
        self.command("add", "file.txt")
        (self.root / "file.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing files: file.txt"):
            self.check()
        with self.assertRaisesRegex(ValueError, "Missing or non-regular"):
            self.check(staged=True)

    def test_deleted_extra_tracked_file_needs_index_removal(self):
        self.write("file.txt")
        self.write("old.txt")
        self.command("add", ".")
        (self.root / "old.txt").unlink()
        self.assertEqual(self.check(), 1)
        with self.assertRaisesRegex(ValueError, "unexpected files: old.txt"):
            self.check(staged=True)
        self.command("add", "-u")
        self.assertEqual(self.check(staged=True), 1)

    def test_matching_index_passes_without_commit(self):
        self.write("file.txt")
        self.command("add", "file.txt")
        before = (self.root / ".git/index").read_bytes()
        self.assertEqual(self.check(staged=True), 1)
        self.assertEqual((self.root / ".git/index").read_bytes(), before)

    def test_stale_staged_content_is_rejected(self):
        self.write("file.txt", "old\n")
        self.command("add", "file.txt")
        self.write("file.txt", "new content\n")
        self.assertEqual(self.check(), 1)
        with self.assertRaisesRegex(ValueError, "Index differs.*file.txt"):
            self.check(staged=True)

    def test_assume_unchanged_cannot_hide_stale_content(self):
        self.write("file.txt")
        self.command("add", "file.txt")
        self.command("update-index", "--assume-unchanged", "file.txt")
        self.write("file.txt", "new content\n")
        with self.assertRaisesRegex(ValueError, "index visibility flag"):
            self.check(staged=True)

    def test_skip_worktree_cannot_hide_stale_content(self):
        self.write("file.txt")
        self.command("add", "file.txt")
        self.command("update-index", "--skip-worktree", "file.txt")
        self.write("file.txt", "new content\n")
        with self.assertRaisesRegex(ValueError, "index visibility flag"):
            self.check(staged=True)

    def test_untracked_file_is_not_an_index_member(self):
        self.write("file.txt")
        self.command("add", "file.txt")
        self.write("private-notes.txt")
        self.assertEqual(self.check(staged=True), 1)

    def test_git_line_ending_normalization_is_respected(self):
        self.write(".gitattributes", "*.txt text eol=lf\n")
        self.write("file.txt", b"one\ntwo\n")
        self.command("add", ".")
        self.write("file.txt", b"one\r\ntwo\r\n")
        self.assertEqual(self.check(("file.txt", ".gitattributes"), staged=True), 2)

    def test_index_symlink_is_rejected(self):
        self.write("file.txt")
        blob = self.command("hash-object", "-w", "--stdin", input=b"destination").strip().decode()
        self.command("update-index", "--add", "--cacheinfo", f"120000,{blob},file.txt")
        with self.assertRaisesRegex(ValueError, "Symlinks and gitlinks"):
            self.check(staged=True)

    def test_worktree_symlink_is_rejected(self):
        self.write("target.txt")
        try:
            (self.root / "file.txt").symlink_to("target.txt")
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable")
        with self.assertRaisesRegex(ValueError, "non-regular"):
            self.check(("file.txt", "target.txt"))

    def test_gitlink_is_rejected(self):
        self.write("file.txt")
        self.command("update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",module")
        with self.assertRaisesRegex(ValueError, "Symlinks and gitlinks"):
            self.check(("file.txt", "module"), staged=True)

    def test_index_conflict_is_rejected(self):
        self.write("file.txt")
        blob = self.command("hash-object", "-w", "--stdin", input=b"content").strip()
        self.command("update-index", "--index-info", input=b"100644 " + blob + b" 1\tfile.txt\n")
        with self.assertRaisesRegex(ValueError, "Unresolved index conflict"):
            self.check()

    def test_subdirectory_is_rejected(self):
        self.write("nested/file.txt")
        with self.assertRaisesRegex(ValueError, "repository root"):
            PREFLIGHT.check_repository(self.root / "nested", {"file.txt"})

    def test_special_path_names_are_not_pathspecs(self):
        expected = ("a b.txt", "[wild].txt", "line\nbreak.txt")
        for name in expected:
            self.write(name)
        self.command("add", ".")
        self.assertEqual(self.check(expected, staged=True), len(expected))

    def test_empty_expected_inventory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected inventory"):
            self.check(())


if __name__ == "__main__":
    unittest.main()
