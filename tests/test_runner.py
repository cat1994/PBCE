"""Bounded launcher checks using only the Python standard library."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("pbce_runner", ROOT / "bin/run_all_pbce.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def fixed_result(iterations=2, algorithms=runner.PBCE_ALGORITHMS):
    lines = ["algorithm  (ev, milliseconds, max_full_phi_regrets_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret)"]
    for iteration in range(iterations):
        lines.append(f"t = {iteration}")
        lines.extend(f"{algorithm}  (0, 1, 2, 3, 4, 5)" for algorithm in sorted(algorithms))
    return "\n".join(lines) + "\n"


def adaptive_result(algorithms=runner.PBCE_ALGORITHMS, status="MAX_GLOBAL_ITER", iteration=2):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["record_type", "algorithm", "global_t", "epsilon", "epsilon_stage", "stage_t", "status", "posr", "osr", "csr"])
    for algorithm in sorted(algorithms):
        writer.writerow(["RUN_END", algorithm, iteration, "0.1", 0, iteration, status, 0.01, 0.02, 0.03])
    return output.getvalue()


class RunnerTests(unittest.TestCase):
    def test_documented_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.games, ["kuhn_3p", "leduc", "random_goofspiel", "sheriff", "tiny_bridge"])
        self.assertEqual(args.data_dir, "outputs/runs")
        self.assertFalse(args.overwrite)
        self.assertFalse(args.legacy_algorithm_mapping)
        self.assertEqual(args.epsilon_decay, 0.5)
        runs = runner.build_runs(args)
        self.assertEqual(len(runs), 35)
        self.assertEqual(sum(run.adaptive for run in runs), 5)
        self.assertTrue(all(run.iterations == 100_000 for run in runs))
        self.assertEqual(
            runner.GAME_MAP["random_goofspiel"],
            "goofspiel(imp_info=True,num_cards=4,points_order=random)",
        )

    def test_modes_and_repeated_games(self):
        for mode, count in (("fixed", 6), ("adaptive", 1), ("both", 7)):
            args = runner.parse_args(["--mode", mode, "--games", "kuhn_2p", "kuhn_2p", "--iterations", "2"])
            runs = runner.build_runs(args)
            self.assertEqual(len(runs), count)
            self.assertTrue(all("--t=2" in run.command for run in runs))

    def test_historical_mapping_is_explicit_in_every_command(self):
        for flags, expected in (([], "false"), (["--legacy-algorithm-mapping"], "true")):
            args = runner.parse_args(flags)
            self.assertTrue(all(
                f"--legacy_algorithm_mapping={expected}" in run.command
                for run in runner.build_runs(args)
            ))

    def test_reject_invalid_options(self):
        for flags in (
            ["--iterations", "0"], ["--iterations", "-1"], ["--iterations", str(2**31)],
            ["--max-parallel-jobs", "0"], ["--iteration-log-interval", "-1"],
            ["--epsilon-decay", "nan"], ["--epsilon-init", "inf"],
            ["--epsilon-min", "0.2"], ["--checkpoint-freq", "0"],
            ["--stability-window", "1"], ["--response-ratio-tol", "2"],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.parse_args(flags)
                self.assertEqual(error.exception.code, 2)

    def test_paths_are_absolute_and_independent_of_current_directory(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                runs = runner.build_runs(runner.parse_args(["--mode", "adaptive"]))
                for run in runs:
                    self.assertEqual(run.output.parent, ROOT / "outputs/runs")
                    self.assertEqual(Path(run.command[0]), ROOT / "build/bin/run_simultaneous_ltbr_pbce")
        finally:
            os.chdir(original)

    def test_bundled_data_is_protected_even_with_overwrite(self):
        for output_directory in ("data", "data/raw", "outputs/../data/new"):
            with self.subTest(directory=output_directory), contextlib.redirect_stderr(io.StringIO()):
                result = runner.main(["--dir", output_directory, "--overwrite", "--dry-run"])
                self.assertEqual(result, 1)

    @unittest.skipUnless(os.name == "posix", "Symbolic links require POSIX")
    def test_bundled_data_symlink_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "data-alias"
            alias.symlink_to(ROOT / "data", target_is_directory=True)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["--dir", str(alias), "--dry-run"]), 1)

    def test_fixed_complete(self):
        self.assertEqual(runner.validate_fixed(io.StringIO(fixed_result()), 2), (True, "ok"))

    def test_fixed_truncated_or_missing_algorithm(self):
        invalid = (
            fixed_result(1),
            fixed_result(algorithms=runner.PBCE_ALGORITHMS - {"CFR"}),
            fixed_result().rsplit("\n", 2)[0] + "\n",
            fixed_result().replace("t = 1", "t = 2"),
            fixed_result().replace("(0, 1, 2, 3, 4, 5)", "(nan, 1, 2, 3, 4, 5)", 1),
        )
        for result in invalid:
            self.assertFalse(runner.validate_fixed(io.StringIO(result), 2)[0])

    def test_adaptive_complete_and_early_certification(self):
        for status, iteration in (("MAX_GLOBAL_ITER", 2), ("CERTIFIED_FINAL", 1), ("STAGE_TIMEOUT", 1)):
            self.assertEqual(
                runner.validate_adaptive(io.StringIO(adaptive_result(status=status, iteration=iteration)), 2),
                (True, "ok"),
            )

    def test_adaptive_missing_algorithm_or_invalid_end(self):
        invalid = (
            adaptive_result(algorithms=runner.PBCE_ALGORITHMS - {"CFR"}),
            adaptive_result(status="NUMERICAL_ERROR"),
            adaptive_result(iteration=1),
            adaptive_result(iteration=3),
            adaptive_result() + "RUN_END,CFR,2,0.1,0,2,MAX_GLOBAL_ITER,0.01,0.02,0.03\n",
            adaptive_result().rsplit(",", 1)[0] + "\n",
            adaptive_result().replace("0.01,0.02,0.03", "nan,0.02,0.03", 1),
            adaptive_result().replace("0.01,0.02,0.03", "0.01,inf,0.03", 1),
            adaptive_result().replace("0.01,0.02,0.03", "0.01,0.02,nan", 1),
            adaptive_result().replace("stage_t", "status", 1),
            adaptive_result().replace("0,2,MAX_GLOBAL_ITER", "0,3,MAX_GLOBAL_ITER", 1),
        )
        for result in invalid:
            self.assertFalse(runner.validate_adaptive(io.StringIO(result), 2)[0])

    def test_dry_run_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                result = runner.main(["--dry-run", "--mode", "adaptive", "--dir", str(output)])
            self.assertEqual(result, 0)
            self.assertIn("--adaptive_epsilon=true", printed.getvalue())
            self.assertFalse(output.exists())

    def test_malformed_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.ssv"
            self.assertFalse(runner.validate_output(output, False, 2)[0])
            output.write_text(fixed_result().replace("t = 0", "t = nope"), encoding="utf-8")
            self.assertFalse(runner.validate_output(output, False, 2)[0])

    @unittest.skipUnless(os.name == "posix", "Executable scripts require POSIX")
    def test_previous_result_is_not_accepted_as_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "run_simultaneous_ltbr_pbce"
            executable.write_text(f"#!{sys.executable}\n", encoding="utf-8")
            executable.chmod(0o755)
            output = directory / "pefr.kuhn_2p.null.sim.gen.adaptive.ssv"
            output.write_text(adaptive_result(), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
                result = runner.main([
                    "--mode", "adaptive", "--games", "kuhn_2p", "--iterations", "2",
                    "--dir", str(directory), "--exe-dir", str(directory), "--overwrite",
                ])
            self.assertEqual(result, 1)
            self.assertIn("unreadable or malformed result", printed.getvalue())
            self.assertFalse(output.exists())

    def test_existing_later_output_blocks_entire_matrix_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(["--mode", "both", "--games", "kuhn_2p", "--dir", directory])
            runs = runner.build_runs(args)
            preserved = runs[-1].output
            preserved.write_text("existing result", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as printed:
                self.assertEqual(runner.main([
                    "--mode", "both", "--games", "kuhn_2p", "--dir", directory,
                ]), 1)
            self.assertIn("Output already exists", printed.getvalue())
            self.assertEqual(preserved.read_text(), "existing result")
            self.assertEqual(list(Path(directory).iterdir()), [preserved])

    def test_orphaned_artifacts_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = runner.build_runs(runner.parse_args([
                "--mode", "adaptive", "--games", "kuhn_2p", "--dir", directory,
            ]))
            for path in runner.run_artifacts(runs[0])[1:]:
                with self.subTest(path=path.name):
                    path.write_text("preserve", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runner.check_output_paths(runs, overwrite=False)
                    self.assertEqual(path.read_text(), "preserve")
                    path.unlink()

    @unittest.skipUnless(os.name == "posix", "Executable scripts require POSIX")
    def test_explicit_overwrite_clears_only_requested_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "run_simultaneous_ltbr_pbce"
            executable.write_text(
                f"#!{sys.executable}\nimport pathlib, sys\n"
                "output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--file_dir='))\n"
                f"pathlib.Path(output).write_text({adaptive_result()!r})\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            args = [
                "--mode", "adaptive", "--games", "kuhn_2p", "--iterations", "2",
                "--dir", str(directory), "--exe-dir", str(directory), "--overwrite",
            ]
            run = runner.build_runs(runner.parse_args(args))[0]
            for path in runner.run_artifacts(run)[:-1]:
                path.write_text("stale", encoding="utf-8")
            policies = runner.run_artifacts(run)[-1]
            policies.mkdir()
            (policies / "old.jsonl").write_text("stale", encoding="utf-8")
            unrelated = directory / "unrelated.ssv"
            unrelated.write_text("preserve", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(args), 0)
            self.assertEqual(run.output.read_text(), adaptive_result())
            metadata = json.loads(run.output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["command"], run.command)
            self.assertFalse(metadata["legacy_algorithm_mapping"])
            self.assertEqual(metadata["executable_sha256"], runner.executable_sha256(executable))
            self.assertFalse(policies.exists())
            self.assertFalse(Path(str(run.output) + ".tmp").exists())
            self.assertEqual(unrelated.read_text(), "preserve")

    @unittest.skipUnless(os.name == "posix", "Symbolic links require POSIX")
    def test_overwrite_refuses_output_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = runner.build_runs(runner.parse_args([
                "--mode", "adaptive", "--games", "kuhn_2p", "--dir", directory,
            ]))
            target = Path(directory) / "important.txt"
            target.write_text("preserve", encoding="utf-8")
            runs[0].output.symlink_to(target)
            with self.assertRaises(ValueError):
                runner.check_output_paths(runs, overwrite=True)
            self.assertEqual(target.read_text(), "preserve")

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup requires POSIX")
    def test_sigterm_reaps_running_child(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pid_file = directory / "child.pid"
            executable = directory / "run_simultaneous_ltbr_pbce"
            executable.write_text(
                f"#!{sys.executable}\nimport os, pathlib, time\n"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            process = subprocess.Popen([
                sys.executable, str(ROOT / "bin/run_all_pbce.py"), "--mode", "adaptive",
                "--games", "kuhn_2p", "--iterations", "2", "--dir", str(directory),
                "--exe-dir", str(directory),
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline and process.poll() is None:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists(), "Child did not start within five seconds")
                child_pid = int(pid_file.read_text())
                process.send_signal(signal.SIGTERM)
                output, _ = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 128 + signal.SIGTERM, output)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=2)
                if child_pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(child_pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
