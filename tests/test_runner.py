"""Bounded launcher checks using only the Python standard library."""

import contextlib
import csv
import importlib.util
import io
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
    writer = csv.writer(output)
    writer.writerow(["record_type", "algorithm", "global_t", "epsilon", "epsilon_stage", "stage_t", "status"])
    for algorithm in sorted(algorithms):
        writer.writerow(["RUN_END", algorithm, iteration, "0.1", 0, iteration, status])
    return output.getvalue()


class RunnerTests(unittest.TestCase):
    def test_modes_and_repeated_games(self):
        for mode, count in (("fixed", 6), ("adaptive", 1), ("both", 7)):
            args = runner.parse_args(["--mode", mode, "--games", "kuhn_2p", "kuhn_2p", "--iterations", "2"])
            runs = runner.build_runs(args)
            self.assertEqual(len(runs), count)
            self.assertTrue(all("--t=2" in run.command for run in runs))

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

    def test_game_assets_are_portable(self):
        for game in ("shapley", "test_game"):
            path = Path(runner.GAME_MAP[game])
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(ROOT))

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
            adaptive_result() + "RUN_END,CFR,2,0.1,0,2,MAX_GLOBAL_ITER\n",
            adaptive_result().rsplit(",", 1)[0] + "\n",
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
                    "--dir", str(directory), "--exe-dir", str(directory),
                ])
            self.assertEqual(result, 1)
            self.assertIn("result was not replaced", printed.getvalue())

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
