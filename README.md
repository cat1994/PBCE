# PBCE experiments

This repository contains the code needed to run the fixed-epsilon PBCE experiments and the stagewise adaptive PEFR controller extracted from the original experimental workspace. It includes the C++ runner, its shared learner and evaluation code, the required game implementations, a Python batch launcher, and regression tests.

Intended GitHub publication target: [cat1994/PBCE](https://github.com/cat1994/PBCE).

## Build and test

Requirements:

- A C++17 compiler and a C++ standard library with filesystem support.
- CMake 3.16 or newer and a supported build tool, such as Make or Ninja.
- Python 3.9 or newer for the experiment launcher and its tests.
- A POSIX environment such as Linux for the documented process-management workflow.

The required OpenSpiel source subset, Abseil, and Eigen headers are included. Configuration and compilation do not download dependencies, and the Python launcher uses only the standard library. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licenses.

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
(cd build && ctest --output-on-failure)
```

The experiment executable is `build/bin/run_simultaneous_ltbr_pbce`. CTest runs the C++ regression tests and Python launcher checks. To run the Python checks independently:

```sh
python3 -B -m unittest discover -s tests -v
```

## Run experiments

The batch launcher supports `fixed`, `adaptive`, and `both` modes. All examples below write results and per-run logs under `data/`. Paths passed to `--dir` and `--exe-dir` are resolved relative to the repository root; absolute paths are also accepted.

Start with a small two-player Kuhn poker run covering both modes:

```sh
python3 bin/run_all_pbce.py \
  --mode both --games kuhn_2p --iterations 20 \
  --checkpoint-freq 5 --max-parallel-jobs 2 --threads-per-run 1 \
  --dir data/smoke
```

This starts six fixed-epsilon jobs and one adaptive job. Each job runs the five algorithms listed below. The small iteration budget checks that the executable and output pipeline work; it does not establish convergence.

Run either experiment family separately:

```sh
python3 bin/run_all_pbce.py \
  --mode fixed --games kuhn_2p --iterations 1000 \
  --max-parallel-jobs 2 --threads-per-run 1 --dir data/fixed

python3 bin/run_all_pbce.py \
  --mode adaptive --games kuhn_2p --iterations 1000 \
  --checkpoint-freq 50 --max-parallel-jobs 1 --threads-per-run 1 \
  --dir data/adaptive
```

For the larger experiment suite:

```sh
python3 bin/run_all_pbce.py \
  --mode both \
  --games kuhn_3p leduc random_goofspiel goofspiel_3p \
          goofspiel_ascending_3p sheriff tiny_bridge battleship \
  --iterations 100000 --max-parallel-jobs 2 --threads-per-run 5 \
  --iteration-log-interval 1000 --dir data/experiments
```

Game trees vary substantially in size. Set `--max-parallel-jobs` to limit concurrent game/epsilon jobs and `--threads-per-run` to limit concurrent algorithms within each job. Run `--dry-run` to inspect commands without starting jobs or creating output directories:

```sh
python3 bin/run_all_pbce.py --mode both --games kuhn_2p --dry-run
python3 bin/run_all_pbce.py --help
```

With no options, the launcher selects `random_goofspiel`, both modes, 100,000 iterations, at most six concurrent jobs, one algorithm thread per job, and `build/bin` as the executable directory. Alternate updates and CFR+ truncation are enabled; `--no-alt` and `--no-cfr-plus` disable them.

The fixed sweep uses epsilon values `1e-1`, `5e-2`, `1e-2`, `5e-3`, `1e-3`, and `0`. Adaptive defaults include initial epsilon `0.1`, decay `0.75`, minimum epsilon `1e-10`, and checkpoint frequency `50`; `--help` lists the controller options. The C++ runner checks that the chosen epsilon is feasible for the game's action counts.

The historical iteration conventions are preserved: fixed mode with `--iterations T` writes `T` records per algorithm, indexed `t = 0, ..., T-1`, with `T-1` learner updates and a final evaluation. Adaptive mode performs at most `T` updates and may terminate earlier. Equal iteration arguments therefore use these respective conventions.

## Games and algorithm labels

[pbce/experiments.py](pbce/experiments.py) defines the available game tags and exact parameters:

- `kuhn_2p`, `kuhn_3p`, `kuhn_4p`, `leduc`, and `leduc_3p`.
- `goofspiel`, `goofspiel_ascending`, `random_goofspiel`, `goofspiel_3p`, and `goofspiel_ascending_3p`.
- `sheriff`, `tiny_bridge`, `tiny_hanabi`, and `battleship`.
- `shapley` and `test_game`, loaded from the bundled files in `games/efg/`.

`leduc_3p` uses the bundled Leduc game's standard raise sizes of 2 and 4.
The original configuration's `raise_sizes=[1]` argument was unsupported by this
OpenSpiel version and prevented the game from loading; it has been removed.

The executable supports the original PBCE algorithm group `3`. Historical output labels and their exact predecessor classes are retained in [src/bin/ltbr.h](src/bin/ltbr.h):

| Output label | Predecessor class |
| --- | --- |
| `A-EFR_IN` | `InformedActionSequencePredecessors` |
| `CSPS-EFR` | `CausalPartialSequencePredecessors` |
| `CFPS-EFR` | `CounterfactualPartialSequencePredecessors` |
| `CFR` | `CounterfactualPartialSequenceExInPredecessors` |
| `CFR_IN` | `TwiceInformedPartialSequencePredecessors` |

Each is instantiated through `BehavioralDeviationTabularCfvLearner` and wrapped by `PerturbedCfTreeLearnerProfile`. In particular, interpret `CFR` and `CFR_IN` using the explicit mappings above when comparing results or documenting methods.

## Results and completion checks

Result filenames retain the original convention:

```text
pefr.<game>.null.sim.gen.<epsilon>.ssv
pefr.<game>.null.sim.gen.adaptive.ssv
```

Fixed results use the historical space-separated text format: a metric header, a `t = ...` line for each recorded iteration, and one tuple for each algorithm. Adaptive results contain comma-separated CSV despite the `.ssv` suffix. Read adaptive files with a CSV reader and use fields such as `record_type`, `algorithm`, `global_t`, `epsilon_stage`, `stage_t`, `epsilon`, and `status`.

Each job has a matching `.log` file containing the executable's stdout and stderr. Adding `--save-stage-policies` to an adaptive run writes JSONL files under `<result>.ssv.policies/`; these contain metadata and per-iteration policy profiles. Files from an uncertified stage receive an additional `.partial` suffix. Serializing policies can produce large outputs.

The launcher checks the process exit status, all five algorithms, every requested fixed record, and one valid adaptive `RUN_END` per algorithm. It rejects malformed or incomplete results and previously existing result files that the new run did not replace. A failed batch exits with a nonzero status. On interruption, the launcher terminates and waits for its running subprocesses.

A completed adaptive job can end with `MAX_GLOBAL_ITER`, `STAGE_TIMEOUT`, or `FINAL_NOT_CERTIFIED`; these indicate a completed run without final controller certification. `CERTIFIED_FINAL` means the implemented controller's final-stage conditions were satisfied. Successful process completion and valid output alone do not establish certified convergence. `NUMERICAL_ERROR` and `PROCESS_ERROR` are failures.

Use separate output directories for configurations you want to compare: filenames encode the game and epsilon mode, while options such as update order, CFR+, and adaptive thresholds are supplied on the command line. Repeating a successful run with the same filename replaces its result and log.

## Repository scope

The extraction excludes historical result data, notebooks, plotting environments, build artifacts, and unrelated general EFR/QBCE executable entry points. Shared learner, regret, and evaluation implementations are retained where needed by the PBCE dependency chain, including shared definitions whose names also appear in other experiment families. This preserves the original implementation semantics while limiting the exposed experiment interface to PBCE.

Project comments and documentation are in English. Third-party source and license notices are retained; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
