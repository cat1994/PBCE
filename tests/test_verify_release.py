"""Publication validation must not require a source checksum manifest."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_release


class ReleaseValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/raw").mkdir(parents=True)
        self.readme = self.root / "README.md"
        self.readme.write_text("Original documentation\n")
        self.files = [self.readme]
        self.entries = []
        for number in range(35):
            raw = b"example data\n"
            packed = gzip.compress(raw, mtime=0)
            path = self.root / f"data/raw/{number}.ssv.gz"
            path.write_bytes(packed)
            self.files.append(path)
            self.entries.append({"path": f"raw/{number}.ssv.gz",
                                 "gzip_bytes": len(packed), "source_bytes": len(raw)})
        self.manifest = self.root / "data/manifest.json"
        self.files.append(self.manifest)
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({"files": self.entries}))

    def verify(self):
        with patch.object(verify_release, "ROOT", self.root), \
             patch.object(verify_release, "release_files", return_value=self.files), \
             contextlib.redirect_stdout(io.StringIO()):
            verify_release.verify()

    def test_readme_edits_need_no_hash_refresh(self):
        self.verify()
        self.readme.write_text("Shortened documentation\n")
        self.verify()
        self.assertFalse((self.root / "SHA256SUMS").exists())

    def test_missing_data_inventory_is_rejected(self):
        self.files.pop(1)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.verify()

    def test_uncompressed_size_mismatch_is_rejected(self):
        self.entries[0]["source_bytes"] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Raw data mismatch"):
            self.verify()

    def test_invalid_gzip_still_fails(self):
        path = self.files[1]
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaises(gzip.BadGzipFile):
            self.verify()


if __name__ == "__main__":
    unittest.main()
