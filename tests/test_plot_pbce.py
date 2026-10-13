"""Regression checks for the release's metric conversion and plotting inputs."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_pbce", ROOT / "bin/plot_PBCE.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PlotInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "result.ssv"

    def fixed(self, count=3):
        lines = [plot.HEADER]
        for t in range(count):
            lines.append(f"t = {t}")
            lines.extend(f"{a}  (0, 1, {6*(t+1)}, {3*(t+1)}, 2, 999)" for a in plot.ALGORITHMS)
        return "\n".join(lines) + "\n"

    def test_fixed_cumulative_normalization_leaves_csr_unchanged(self):
        self.path.write_text(self.fixed())
        records, meta = plot.fixed_records(self.path, stride=1)
        self.assertEqual(meta["rounds"], 3)
        for values in records.values():
            self.assertEqual([v["x"] for v in values], [1, 2, 3])
            self.assertTrue(all((v["posr"], v["osr"], v["csr"]) == (6, 3, 2) for v in values))

    def test_paper_sampling_order_and_compression(self):
        self.path = self.path.with_suffix(".ssv.gz")
        with gzip.open(self.path, "wt") as stream:
            stream.write(self.fixed(1000))
        records, meta = plot.fixed_records(self.path, max_points=3, stride=100)
        self.assertEqual([r["x"] for r in records["CFR"]], [1, 401, 901])
        self.assertEqual(meta["last_plotted_round"], 901)

    def test_fixed_truncation_and_duplicate_algorithm_are_errors(self):
        content = self.fixed()
        for bad in (content.rsplit("CFR_IN", 1)[0], content.replace("CFR_IN", "CFR")):
            self.path.write_text(bad)
            with self.assertRaises(ValueError):
                plot.fixed_records(self.path)

    def test_fixed_rejects_nonfinite_even_on_unsampled_round(self):
        self.path.write_text(self.fixed().replace("(0, 1, 12,", "(0, 1, nan,"))
        with self.assertRaises(ValueError):
            plot.fixed_records(self.path)

    def adaptive(self, missing_end=False):
        fields = ["record_type", "algorithm", "global_t", "epsilon_stage", "epsilon", "posr", "osr", "csr", "status"]
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for algorithm in plot.ALGORITHMS:
                writer.writerow(["CHECKPOINT", algorithm, 50, 0, .1, .2, .3, .4, ""])
                writer.writerow(["STAGE_END", algorithm, 50, 0, .1, .2, .3, .4, "CERTIFIED"])
                writer.writerow(["CHECKPOINT", algorithm, 100, 1, .05, .1, .2, .3, ""])
                if not (missing_end and algorithm == "CFR"):
                    writer.writerow(["RUN_END", algorithm, 100, 1, .05, .1, .2, .3, "MAX_GLOBAL_ITER"])

    def test_adaptive_only_checkpoints_no_extra_normalization(self):
        self.adaptive()
        records, meta = plot.adaptive_records(self.path)
        self.assertEqual(len(records["CFR"]), 2)
        self.assertEqual(records["CFR"][-1]["csr"], .3)
        self.assertEqual(meta["endings"]["CFR"]["status"], "MAX_GLOBAL_ITER")

    def test_adaptive_missing_terminal_row_is_error(self):
        self.adaptive(missing_end=True)
        with self.assertRaises(ValueError):
            plot.adaptive_records(self.path)

    def test_adaptive_nan_and_numerical_error_are_rejected(self):
        self.adaptive()
        content = self.path.read_text()
        for bad in (content.replace(",0.4,", ",nan,"), content.replace("MAX_GLOBAL_ITER", "NUMERICAL_ERROR")):
            self.path.write_text(bad)
            with self.assertRaises(ValueError):
                plot.adaptive_records(self.path)

    def test_adaptive_rejects_invalid_terminal_and_trailing_rows(self):
        self.adaptive()
        content = self.path.read_text()
        bad_inputs = [content.replace("RUN_END,CFR,100,", "RUN_END,CFR,40,"),
                      content.replace("MAX_GLOBAL_ITER", "BOGUS"),
                      content + "IGNORED,A-EFR_IN\n",
                      content + "CHECKPOINT,CFR,150,1,0.05,0.1,0.2,0.3,\n"]
        for bad in bad_inputs:
            self.path.write_text(bad)
            with self.assertRaises(ValueError):
                plot.adaptive_records(self.path)

    def test_terminal_epsilon_can_differ_from_last_checkpoint(self):
        self.adaptive()
        self.path.write_text(self.path.read_text().replace("RUN_END,CFR,100,1,0.05", "RUN_END,CFR,100,2,0.025"))
        records, _ = plot.adaptive_records(self.path)
        self.assertEqual(records["CFR"][-1]["epsilon"], .05)
        self.assertEqual(records["CFR"][-1]["terminal_epsilon"], .025)

    def test_aliases_and_invalid_sampling(self):
        self.assertEqual(plot.parse_args(["--algorithms", "CF"]).algorithms, ["CF"])
        self.assertEqual(plot.parse_args([]).label_mode, "auto")
        self.assertEqual(plot.parse_args(["--label-mode", "historical"]).label_mode, "historical")
        with self.assertRaises(SystemExit):
            plot.parse_args(["--fixed-stride", "0"])


class LabelProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "result.ssv"
        self.path.write_text("test numeric records\n", encoding="utf-8")
        self.manifest = self.directory / "manifest.json"

    def provenance(self, mode="auto"):
        return plot.label_provenance(self.path, mode, self.manifest)

    def archive(self):
        self.manifest.write_text(json.dumps({"files": [{"source_sha256": plot.sha256(self.path)}]}))

    def sidecar(self, legacy):
        self.path.with_suffix(".metadata.json").write_text(json.dumps({"legacy_algorithm_mapping": legacy}))

    def test_unknown_raw_cfr_is_not_silently_declared_paper_cf(self):
        context = self.provenance()
        self.assertEqual(context["label_mode"], "unspecified")
        cfr = next(item for item in context["algorithm_mapping"] if item["raw_label"] == "CFR")
        self.assertEqual(cfr["display_label"], "CF")
        self.assertIsNone(cfr["class"])
        self.assertIsNone(cfr["paper_family_equivalence"])
        _, note = plot.figure_provenance("CFR", [context])
        self.assertIn("Unverified", note)

    def test_verified_archive_mapping_explicitly_retains_different_families(self):
        self.archive()
        context = self.provenance()
        self.assertEqual(context["label_mode"], "historical")
        self.assertEqual(context["evidence"], ["bundled_archive_sha256"])
        for mapping in context["algorithm_mapping"]:
            self.assertEqual(mapping["class"], plot.HISTORICAL_CLASSES[mapping["raw_label"]])
            self.assertEqual(mapping["paper_family_equivalence"], mapping["raw_label"] not in ("CFR", "CFR_IN"))
        _, note = plot.figure_provenance("CFR", [context])
        self.assertIn("not the paper's CF family", note)

    def test_archive_file_modification_loses_automatic_historical_identity(self):
        self.archive()
        self.path.write_text("different numeric records\n", encoding="utf-8")
        self.assertEqual(self.provenance()["label_mode"], "unspecified")

    def test_launch_metadata_controls_family_mode(self):
        for legacy, expected in ((False, "paper"), (True, "historical")):
            self.sidecar(legacy)
            context = self.provenance()
            self.assertEqual(context["label_mode"], expected)
            self.assertIn("launcher_metadata_mode", context["evidence"])
        self.sidecar("false")
        self.assertEqual(self.provenance()["label_mode"], "unspecified")

    def test_compressed_input_reads_original_sidecar_name(self):
        self.sidecar(False)
        self.path = self.path.with_suffix(".ssv.gz")
        with gzip.open(self.path, "wt") as stream:
            stream.write("test numeric records\n")
        self.assertEqual(self.provenance()["label_mode"], "paper")

    def test_explicit_mode_is_recorded_but_cannot_override_verified_history(self):
        context = self.provenance("paper")
        self.assertEqual(context["label_mode"], "paper")
        self.assertEqual(context["evidence"], ["explicit_label_mode"])
        self.archive()
        with self.assertRaisesRegex(ValueError, "contradicts historical"):
            self.provenance("paper")
        self.sidecar(False)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.provenance()

    def test_cfr_paper_and_historical_inputs_cannot_be_mixed(self):
        for raw in ("CFR", "CFR_IN"):
            with self.assertRaisesRegex(ValueError, "plot historical and paper-family runs separately"):
                plot.figure_provenance(raw, [{"label_mode": "paper"}, {"label_mode": "historical"}])
        self.assertEqual(plot.figure_provenance("A-EFR_IN", [{"label_mode": "paper"}, {"label_mode": "historical"}])[0], "mixed")

    def test_validate_manifest_uses_raw_keys_display_aliases_and_anonymous_paths(self):
        adaptive = self.directory / "adaptive"
        adaptive.mkdir()
        source = adaptive / "pefr.kuhn_3p.null.sim.gen.adaptive.ssv"
        fields = ["record_type", "algorithm", "global_t", "epsilon_stage", "epsilon", "posr", "osr", "csr", "status"]
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for algorithm in plot.ALGORITHMS:
                writer.writerow(["CHECKPOINT", algorithm, 50, 0, .1, .2, .3, .4, ""])
                writer.writerow(["RUN_END", algorithm, 50, 0, .1, .2, .3, .4, "MAX_GLOBAL_ITER"])
        before = source.read_bytes()
        output = self.directory / "figures"
        self.assertEqual(plot.main(["--mode", "adaptive", "--adaptive-data", str(adaptive),
                                    "--games", "kuhn_3p", "--algorithms", "CF", "CF-IN", "--label-mode", "historical",
                                    "--output-dir", str(output), "--validate-only"]), 0)
        report = json.loads((output / "plot_manifest.json").read_text())
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual([item["raw_label"] for item in report["figures"]], ["CFR", "CFR_IN"])
        self.assertEqual([item["display_label"] for item in report["figures"]], ["CF", "CF-IN"])
        self.assertEqual(report["inputs"][0]["path"], source.name)
        self.assertNotIn(str(self.directory), json.dumps(report))


if __name__ == "__main__":
    unittest.main()
