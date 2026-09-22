#!/usr/bin/env python3
"""Run fixed-epsilon PBCE baselines and/or stagewise adaptive PEFR (Python 3.9+)."""

import argparse
import csv
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pbce.experiments import (
    DEFAULT_GAMES,
    FIXED_EPSILONS,
    GAME_MAP,
    NUM_ITERATIONS_MAP,
    PBCE_ALGORITHMS,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", dest="data_dir", default="data", help="Output directory.")
    parser.add_argument("--exe-dir", default="build/bin", help="Directory with the C++ executable.")
    parser.add_argument("--mode", choices=("fixed", "adaptive", "both"), default="both")
    parser.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES), choices=sorted(GAME_MAP))
    parser.add_argument("--max-parallel-jobs", type=int, default=6)
    parser.add_argument("--threads-per-run", type=int, default=1)
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Fixed: T records (T-1 updates); adaptive: at most T updates. Default: 100000.",
    )
    parser.add_argument(
        "--iteration-log-interval", type=int, default=0,
        help="Print child progress every N completed iterations; 0 disables it.",
    )
    parser.add_argument("--alt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfr-plus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epsilon-init", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.75)
    parser.add_argument("--epsilon-min", type=float, default=1e-10)
    parser.add_argument("--stability-window", type=int, default=5)
    parser.add_argument("--stability-tolerance", type=float, default=0.05)
    parser.add_argument("--min-response", type=float, default=0.01)
    parser.add_argument("--response-ratio-tol", type=float, default=0.01)
    parser.add_argument("--quality-tolerance", type=float, default=0.05)
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    parser.add_argument("--min-stage-iterations", type=int, default=500)
    parser.add_argument("--stable-checkpoints", type=int, default=3)
    parser.add_argument("--max-stage-iterations", type=int, default=0)
    parser.add_argument("--w-numerical-eps", type=float, default=1e-12)
    parser.add_argument(
        "--save-stage-policies", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args(argv)
    for name in ("max_parallel_jobs", "threads_per_run", "iterations"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.iterations is not None and args.iterations > 2**31 - 1:
        parser.error("--iterations exceeds the C++ executable's 32-bit limit")
    if args.iteration_log_interval < 0:
        parser.error("--iteration-log-interval must be non-negative")
    if args.mode in ("adaptive", "both"):
        for name in (
            "epsilon_init", "epsilon_min", "stability_tolerance", "min_response",
            "response_ratio_tol", "quality_tolerance", "w_numerical_eps",
        ):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
        if not 0 < args.epsilon_decay < 1:
            parser.error("--epsilon-decay must be in (0, 1)")
        if args.epsilon_min > args.epsilon_init:
            parser.error("--epsilon-min must be <= --epsilon-init")
        if args.response_ratio_tol > 1:
            parser.error("--response-ratio-tol must be <= 1")
        for name, minimum in (
            ("stability_window", 2), ("checkpoint_freq", 1), ("stable_checkpoints", 1),
            ("min_stage_iterations", 0), ("max_stage_iterations", 0),
        ):
            if getattr(args, name) < minimum:
                parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    args.games = list(dict.fromkeys(args.games))
    return args


def bool_flag(value):
    return "true" if value else "false"


@dataclass
class Run:
    adaptive: bool
    game: str
    parameter: str
    output: Path
    command: list
    iterations: int


def build_runs(args):
    """Construct commands; relative paths are resolved against this repository."""
    executable = PROJECT_ROOT / args.exe_dir / "run_simultaneous_ltbr_pbce"
    runs = []
    modes = (False, True) if args.mode == "both" else (args.mode == "adaptive",)
    for adaptive in modes:
        for game in args.games:
            for parameter in (("adaptive",) if adaptive else FIXED_EPSILONS):
                output = PROJECT_ROOT / args.data_dir / f"pefr.{game}.null.sim.gen.{parameter}.ssv"
                iterations = args.iterations if args.iterations is not None else NUM_ITERATIONS_MAP[game]
                command = [
                    str(executable), f"--game={GAME_MAP[game]}", "--sampler=null", "--alg_group=3",
                    f"--t={iterations}", f"--threads={args.threads_per_run}", f"--file_dir={output}",
                    f"--alt={bool_flag(args.alt)}", f"--cfr_plus={bool_flag(args.cfr_plus)}",
                    "--verbose_iteration_log=false", f"--iteration_log_interval={args.iteration_log_interval}",
                    f"--adaptive_epsilon={bool_flag(adaptive)}",
                ]
                if adaptive:
                    for name in (
                        "epsilon_init", "epsilon_decay", "epsilon_min", "stability_window",
                        "stability_tolerance", "min_response", "response_ratio_tol", "quality_tolerance",
                        "checkpoint_freq", "min_stage_iterations", "stable_checkpoints",
                        "max_stage_iterations", "w_numerical_eps",
                    ):
                        command.append(f"--{name}={getattr(args, name)}")
                    command.append(f"--save_stage_policies={bool_flag(args.save_stage_policies)}")
                else:
                    command.append(f"--epsilon={parameter}")
                runs.append(Run(adaptive, game, parameter, output, command, iterations))
    return runs


def validate_fixed(file, iterations):
    if not file.readline().startswith("algorithm  (ev, milliseconds,"):
        return False, "invalid fixed header"
    next_iteration = 0
    algorithms = set()
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith("t = "):
            if next_iteration and algorithms != PBCE_ALGORITHMS:
                return False, f"incomplete fixed iteration {next_iteration - 1}"
            if int(line[4:]) != next_iteration:
                return False, "non-sequential fixed iterations"
            next_iteration += 1
            algorithms = set()
            continue
        match = re.fullmatch(r"(\S+)\s+\(([^()]*)\)", line)
        if not match or not next_iteration:
            return False, "malformed fixed result row"
        algorithm, values = match.groups()
        metrics = [float(value) for value in values.split(",")]
        if algorithm not in PBCE_ALGORITHMS or algorithm in algorithms:
            return False, "unexpected or duplicate fixed algorithm"
        if len(metrics) != 6 or not all(math.isfinite(value) for value in metrics):
            return False, "invalid fixed metrics"
        algorithms.add(algorithm)
    if next_iteration != iterations or algorithms != PBCE_ALGORITHMS:
        return False, f"incomplete fixed output: expected {iterations} full iterations"
    return True, "ok"


def validate_adaptive(file, iterations):
    reader = csv.DictReader(file)
    required = {"record_type", "algorithm", "global_t", "epsilon", "epsilon_stage", "stage_t", "status"}
    if not required.issubset(reader.fieldnames or []):
        return False, "missing adaptive columns"
    ended = set()
    last_iteration = dict.fromkeys(PBCE_ALGORITHMS, 0)
    final_statuses = {"CERTIFIED_FINAL", "FINAL_NOT_CERTIFIED", "MAX_GLOBAL_ITER", "STAGE_TIMEOUT"}
    record_types = {"CHECKPOINT", "STAGE_END", "EPSILON_UPDATE", "RUN_END"}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            return False, "malformed adaptive row"
        algorithm, record_type = row["algorithm"], row["record_type"]
        if algorithm not in PBCE_ALGORITHMS or algorithm in ended:
            return False, "unexpected algorithm or data after RUN_END"
        if record_type not in record_types:
            return False, "unexpected adaptive record type"
        iteration, epsilon = int(row["global_t"]), float(row["epsilon"])
        if not last_iteration[algorithm] <= iteration <= iterations:
            return False, "invalid adaptive iteration"
        if int(row["stage_t"]) < 0 or int(row["epsilon_stage"]) < 0:
            return False, "negative adaptive stage or iteration"
        if not math.isfinite(epsilon) or epsilon < 0:
            return False, "invalid adaptive epsilon"
        last_iteration[algorithm] = iteration
        if record_type == "RUN_END":
            if row["status"] not in final_statuses or iteration == 0:
                return False, f"unsuccessful RUN_END for {algorithm}: {row['status']}"
            if row["status"] == "MAX_GLOBAL_ITER" and iteration != iterations:
                return False, "RUN_END precedes the requested iteration limit"
            ended.add(algorithm)
    if ended != PBCE_ALGORITHMS:
        return False, f"missing RUN_END for {sorted(PBCE_ALGORITHMS - ended)}"
    return True, "ok"


def validate_output(output, adaptive, iterations):
    """Stream results to check every expected algorithm without loading large files."""
    try:
        with output.open(newline="", encoding="utf-8") as file:
            return (validate_adaptive if adaptive else validate_fixed)(file, iterations)
    except (OSError, ValueError, csv.Error) as error:
        return False, f"unreadable or malformed result: {error}"


def file_signature(path):
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def stop_children(children):
    """Reap all owned subprocesses, including when launching a later job fails."""
    active = [(process, log) for process, log in children if process.poll() is None]
    for process, _ in active:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process, _ in active:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
    for _, log in children:
        log.close()


def main(argv=None):
    args = parse_args(argv)
    pending = deque(build_runs(args))
    if args.dry_run:
        for run in pending:
            print(shlex.join(run.command))
        return 0
    executable = Path(pending[0].command[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        print(f"Executable not found or not executable: {executable}. Build the C++ target first.", file=sys.stderr)
        return 1
    print("start time:", datetime.now(timezone.utc).isoformat(timespec="seconds"), flush=True)
    running, children, failures = [], [], 0
    saved_signals = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            saved_signals[signum] = signal.signal(signum, interrupted)
        while pending or running:
            while pending and len(running) < args.max_parallel_jobs:
                run = pending.popleft()
                run.output.parent.mkdir(parents=True, exist_ok=True)
                previous = file_signature(run.output)
                log = run.output.with_suffix(".log").open("w", encoding="utf-8")
                print(f"[RUN] game={run.game} parameter={run.parameter}", flush=True)
                try:
                    process = subprocess.Popen(
                        run.command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=(os.name == "posix"),
                    )
                except BaseException:
                    log.close()
                    raise
                children.append((process, log))
                running.append((process, log, run, previous))
            still_running = []
            for process, log, run, previous in running:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((process, log, run, previous))
                    continue
                log.close()
                valid, reason = validate_output(run.output, run.adaptive, run.iterations)
                if previous is not None and file_signature(run.output) == previous:
                    valid, reason = False, "result was not replaced by this run"
                if return_code != 0 or not valid:
                    failures += 1
                    print(f"[FAIL] game={run.game} parameter={run.parameter} exit={return_code} {reason}", flush=True)
                else:
                    print(f"[DONE] game={run.game} parameter={run.parameter}", flush=True)
            running = still_running
            if running:
                time.sleep(0.1)
    except OSError as error:
        print(f"Cannot run PBCE experiment: {error}", file=sys.stderr)
        return 1
    finally:
        for signum in saved_signals:
            signal.signal(signum, signal.SIG_IGN)
        stop_children(children)
        for signum, handler in saved_signals.items():
            signal.signal(signum, handler)
    print("end time:", datetime.now(timezone.utc).isoformat(timespec="seconds"), flush=True)
    if failures:
        print(f"{failures} PBCE run(s) failed completeness checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
