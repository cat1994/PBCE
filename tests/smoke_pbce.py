#!/usr/bin/env python3
"""Exercise a built PBCE executable on a small deterministic game.

The relaxed adaptive thresholds below exercise stage transitions; they are not
recommended convergence criteria for scientific experiments. Every output is
written to a temporary directory. An optional reference executable checks that
extraction preserves numerical results while ignoring machine-dependent timing.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbce.experiments import GAME_MAP

ALGORITHMS = {"A-EFR_IN", "CSPS-EFR", "CFPS-EFR", "CFR", "CFR_IN"}
METRICS = (
    "posr", "osr", "csr", "acsr", "posr_norm", "osr_norm", "csr_norm",
    "w_min", "w_min_positive", "w_mean", "w_max", "w_at_max_csr",
)
ADAPTIVE_FLAGS = [
    "--adaptive_epsilon=true", "--t=24", "--epsilon_init=0.1",
    "--epsilon_decay=0.5", "--epsilon_min=0.025", "--checkpoint_freq=1",
    "--min_stage_iterations=4", "--stability_window=2",
    "--stable_checkpoints=1", "--stability_tolerance=1000000",
    "--quality_tolerance=1000000", "--save_stage_policies=true",
]
EXE = None
REFERENCE_EXE = None


class PbceSmokeTests(unittest.TestCase):
    def test_all_configured_games_load(self):
        # Loading checks registration and parameter compatibility without expanding
        # large game trees or running the full experiment suite.
        for name, game in GAME_MAP.items():
            with self.subTest(game=name):
                result = subprocess.run(
                    [str(EXE), f"--game={game}", "--show_num=true"],
                    cwd=self.directory, capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "5")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pbce-smoke-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.run_count = 0

    def run_experiment(self, flags=(), *, executable=None, success=True):
        self.run_count += 1
        output = self.directory / f"run_{self.run_count}.ssv"
        command = [
            str(executable or EXE), "--game=kuhn_poker(players=2)",
            "--sampler=null", "--alg_group=3", "--threads=1",
            "--random_seed=0", "--t=12", "--epsilon=0.01",
            "--alt=true", "--cfr_plus=true", f"--file_dir={output}",
            *flags,
        ]
        result = subprocess.run(
            command, cwd=self.directory, capture_output=True, text=True,
            timeout=30, check=False,
        )
        details = f"Command: {command}\n{result.stdout}\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, details)
            self.assertTrue(output.is_file(), details)
            self.assertFalse(Path(str(output) + ".tmp").exists(), details)
        else:
            self.assertNotEqual(result.returncode, 0, details)
            self.assertFalse(output.exists(), details)
        return output, result

    def fixed_records(self, output):
        lines = output.read_text().splitlines()
        self.assertTrue(lines[0].startswith("algorithm"))
        records = {}
        iteration = None
        for line in lines[1:]:
            if not line:
                continue
            if line.startswith("t = "):
                iteration = int(line[4:])
                self.assertNotIn(iteration, records)
                records[iteration] = {}
                continue
            match = re.fullmatch(r"(\S+)\s+\(([^)]+)\)", line)
            self.assertIsNotNone(match, line)
            self.assertIsNotNone(iteration)
            algorithm = match.group(1)
            values = tuple(float(part) for part in match.group(2).split(","))
            self.assertEqual(len(values), 6, line)
            self.assertTrue(all(math.isfinite(value) for value in values), line)
            self.assertNotIn(algorithm, records[iteration])
            records[iteration][algorithm] = values
        self.assertEqual(set(records), set(range(12)))
        for values in records.values():
            self.assertEqual(set(values), ALGORITHMS)
        return records

    def adaptive_records(self, output):
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertEqual({row["algorithm"] for row in rows}, ALGORITHMS)
        for row in rows:
            for field in METRICS:
                self.assertTrue(math.isfinite(float(row[field])), (field, row))
        ends = [row for row in rows if row["record_type"] == "RUN_END"]
        self.assertEqual(len(ends), 5)
        self.assertEqual({row["algorithm"] for row in ends}, ALGORITHMS)
        return rows

    def assert_numerically_equal(self, actual, expected):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
            (actual, expected),
        )

    def test_fixed_modes_and_algorithm_parallelism(self):
        for flags in (
            [], ["--alt=false", "--cfr_plus=false"], ["--threads=3"],
            ["--epsilon=0"],
        ):
            with self.subTest(flags=flags):
                output, _ = self.run_experiment(flags)
                self.fixed_records(output)

    def test_adaptive_transitions_and_policy_history(self):
        output, result = self.run_experiment(ADAPTIVE_FLAGS)
        rows = self.adaptive_records(output)
        for algorithm in ALGORITHMS:
            own_rows = [row for row in rows if row["algorithm"] == algorithm]
            updates = [row for row in own_rows if row["record_type"] == "EPSILON_UPDATE"]
            self.assertEqual(len(updates), 2, algorithm)
            for row, old, new in zip(updates, (0.1, 0.05), (0.05, 0.025)):
                self.assert_numerically_equal(float(row["old_epsilon"]), old)
                self.assert_numerically_equal(float(row["new_epsilon"]), new)
            checkpoints = [row for row in own_rows if row["record_type"] == "CHECKPOINT"]
            self.assertEqual({int(row["stage_k"]) for row in checkpoints}, {0, 1, 2})
            for stage in range(3):
                stage_rows = [row for row in checkpoints if int(row["stage_k"]) == stage]
                self.assertEqual(int(stage_rows[0]["stage_t"]), 1)
            end = own_rows[-1]
            self.assertEqual(end["record_type"], "RUN_END")
            self.assertEqual(end["status"], "CERTIFIED_FINAL")
            self.assert_numerically_equal(float(end["epsilon"]), 0.025)

        reset_lines = [line for line in result.stdout.splitlines() if line.startswith("STAGE_RESET ")]
        self.assertEqual(len(reset_lines), 10)
        for line in reset_lines:
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            self.assert_numerically_equal(
                float(fields["solver_regret_l1_before"]), float(fields["solver_regret_l1_after"])
            )
            self.assertEqual(float(fields["stage_original_l1_after"]), 0)
            self.assertEqual(float(fields["stage_W_sum_after"]), 0)

        policy_directory = Path(str(output) + ".policies")
        policy_files = sorted(policy_directory.iterdir())
        self.assertEqual(len(policy_files), 15)
        seen_stages = set()
        for policy_file in policy_files:
            self.assertEqual(policy_file.suffix, ".jsonl")
            history = [json.loads(line) for line in policy_file.read_text().splitlines()]
            metadata, *profiles = history
            self.assertEqual(metadata["type"], "metadata")
            key = (metadata["algorithm"], metadata["stage_k"])
            self.assertNotIn(key, seen_stages)
            seen_stages.add(key)
            self.assertEqual(len(profiles), 4)
            self.assertEqual([profile["stage_t"] for profile in profiles], [1, 2, 3, 4])
            for profile in profiles:
                self.assertEqual(profile["type"], "PROFILE")
                self.assertTrue(profile["policies"])
                info_states = set()
                for policy in profile["policies"]:
                    info_state = (policy["player"], policy["info_state"])
                    self.assertNotIn(info_state, info_states)
                    info_states.add(info_state)
                    probabilities = [action[1] for action in policy["actions"]]
                    self.assertTrue(probabilities)
                    self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
                    self.assert_numerically_equal(sum(probabilities), 1)
        self.assertEqual(seen_stages, {(algorithm, stage) for algorithm in ALGORITHMS for stage in range(3)})

    def test_adaptive_budget_limit_keeps_partial_policy_history(self):
        flags = [*ADAPTIVE_FLAGS, "--t=2"]
        output, _ = self.run_experiment(flags)
        rows = self.adaptive_records(output)
        ends = [row for row in rows if row["record_type"] == "RUN_END"]
        self.assertEqual({row["status"] for row in ends}, {"MAX_GLOBAL_ITER"})
        partial_files = list(Path(str(output) + ".policies").glob("*.jsonl.partial"))
        self.assertEqual(len(partial_files), 5)
        for policy_file in partial_files:
            history = [json.loads(line) for line in policy_file.read_text().splitlines()]
            self.assertEqual(len(history), 3)

    def test_invalid_configuration_is_rejected(self):
        for flags in (
            ["--epsilon=0.5"], ["--epsilon=-0.01"], ["--epsilon=nan"],
            ["--t=0"], ["--threads=0"], ["--alg_group=0"], ["--alg_group=999"],
            ["--adaptive_epsilon=true", "--epsilon_decay=1"],
        ):
            with self.subTest(flags=flags):
                self.run_experiment(flags, success=False)

    def test_optional_reference_numerical_equivalence(self):
        if REFERENCE_EXE is None:
            self.skipTest("No reference executable supplied")
        for flags in ([], ["--alt=false", "--cfr_plus=false"]):
            with self.subTest(flags=flags):
                actual_output, _ = self.run_experiment(flags)
                reference_output, _ = self.run_experiment(flags, executable=REFERENCE_EXE)
                actual = self.fixed_records(actual_output)
                expected = self.fixed_records(reference_output)
                for iteration in actual:
                    for algorithm in ALGORITHMS:
                        for index in (0, 2, 3, 4, 5):
                            self.assert_numerically_equal(
                                actual[iteration][algorithm][index], expected[iteration][algorithm][index]
                            )
        actual_output, _ = self.run_experiment(ADAPTIVE_FLAGS)
        reference_output, _ = self.run_experiment(ADAPTIVE_FLAGS, executable=REFERENCE_EXE)
        actual = self.adaptive_records(actual_output)
        expected = self.adaptive_records(reference_output)
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(set(row), set(reference))
            for key, value in row.items():
                if key == "milliseconds" or value == reference[key]:
                    continue
                try:
                    self.assert_numerically_equal(float(value), float(reference[key]))
                except ValueError:
                    self.assertEqual(value, reference[key], key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--reference-exe", type=Path)
    args = parser.parse_args()
    EXE = args.exe.resolve()
    REFERENCE_EXE = args.reference_exe.resolve() if args.reference_exe else None
    for executable in (EXE, REFERENCE_EXE):
        if executable is not None and not executable.is_file():
            parser.error(f"executable does not exist: {executable}")
    unittest.main(argv=[__file__], verbosity=2)
