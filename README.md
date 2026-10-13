# PBCE / PEFR: code and experimental data

This standalone package contains the PBCE experiment runner, the five-game
dataset used in the manuscript figures, and a script that redraws all ten
fixed/adaptive figures. Dependencies for the C++ build are vendored; no source
checkout, submodule fetch, notebook or pickled NumPy dataset is required.

**Implementation status.** The bundled data reproduce the historical numerical
experiments. The current manuscript's intended-device and full-continuation
definitions are not fully implemented by the inherited learner/evaluator.
There are also two documented historical algorithm-label mappings. These are
material scientific differences, recorded in
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md); successful tests
and figure reproduction do not resolve them.

## Reproduce the figures from bundled data

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-plot.txt
python3 scripts/verify_release.py
python3 bin/plot_PBCE.py
```

The final command writes five fixed and five adaptive PDFs to
`outputs/figures/{fixed,adaptive}/`, with the filenames referenced by the paper.
It also writes `plot_manifest.json` with input hashes, sampling settings and
adaptive terminal statuses. No C++ compilation is needed to redraw the data.

```sh
# PDF and PNG; accept either original output labels or paper labels.
python3 bin/plot_PBCE.py --formats pdf png
python3 bin/plot_PBCE.py --mode adaptive --algorithms CSPS CF
# Exact raw records with no fixed-curve subsampling:
python3 bin/plot_PBCE.py --mode fixed --fixed-stride 1 --max-points 0
# Inspect a new experiment directory (plain .ssv and compressed .ssv.gz work):
python3 bin/plot_PBCE.py --fixed-data outputs/runs --adaptive-data outputs/runs
```

Defaults match the paper's fixed-curve sampling: first retain rounds with
`t=0,100,...,99900`, then select 100 equally spaced indices and plot `t+1`.
The last plotted round is 99,901; the archive still includes all 100,000
records. Adaptive plots use all `CHECKPOINT` rows, mark stage transitions and
annotate the terminal epsilon. `--disconnect-stages` breaks curves at stage
boundaries. Nonpositive values are masked on logarithmic axes, never replaced
in the dataset. ACSR and utility-range-normalized columns are not plotted.

The identical script in the parent `correlated_eq` project automatically uses
`data_pbce/` for fixed records and `data_pbce/data_pbce_gamma0d5/` for adaptive
records. Explicit `--fixed-data`, `--adaptive-data` or `--data-dir` paths override
these defaults. Relative paths are resolved against the script's repository.

## Build and test

Requirements: a C++17 compiler, CMake 3.16+, a POSIX environment and Python
3.10+. Matplotlib is optional for running simulations and unit tests.

```sh
make JOBS=4
make test
python3 scripts/verify_release.py --full
```

The executable is `build/bin/run_simultaneous_ltbr_pbce`. OpenSpiel, Abseil and
the required Eigen headers/licenses are included; configuration does not
download dependencies. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Run experiments

```sh
# Fast pipeline check: six fixed epsilon values and one adaptive run.
python3 bin/run_all_pbce.py --games kuhn_2p --iterations 20 \
  --checkpoint-freq 5 --max-parallel-jobs 2 --dir outputs/smoke

# Inspect the paper's 35-run matrix before running it.
python3 bin/run_all_pbce.py --legacy-algorithm-mapping --dry-run

# Re-run the historical five-game experiment configuration.
python3 bin/run_all_pbce.py --legacy-algorithm-mapping \
  --max-parallel-jobs 2 --threads-per-run 1 --dir outputs/historical
```

Defaults select Kuhn (3 players), Leduc (2), random-prize Goofspiel (2 players,
4 cards), Sheriff and Tiny Bridge; alternate CFR+ updates; 100,000 iterations;
fixed epsilon `0.1, 0.05, 0.01, 0.005, 0.001, 0`; adaptive initial epsilon `0.1`,
decay `0.5` and minimum `1e-10`. `kuhn_2p` is retained for regression checks.
Use `--help` for all controller and scheduling parameters.

Without `--legacy-algorithm-mapping`, `CFR` and `CFR_IN` use the standalone
repository's corrected immediate external/internal predecessor classes.
The historical switch restores the two predecessor classes used to produce
the bundled data; it does not fix the separate manuscript/implementation
differences. See the mapping table in the implementation status document.

New runs default to `outputs/runs/`. Each run writes results, a log and a JSON
metadata file containing the command, options and executable SHA-256. Existing
outputs are protected; use `--overwrite` only to replace that run's outputs.
The launcher refuses to write inside the bundled `data/` directory.

Fixed mode preserves the historical convention: `T` records, indexed
`t=0,...,T-1`, include `T-1` learner updates and a final evaluation. Adaptive
mode performs at most `T` updates. Process completion is separate from final
stage certification; `MAX_GLOBAL_ITER`, `STAGE_TIMEOUT` and
`FINAL_NOT_CERTIFIED` remain valid completed runs without final certification.

## Dataset and release

[data/README.md](data/README.md) describes the 35 losslessly compressed raw
files (about 167 MiB; 948 MiB decompressed), their schemas, sources and the
verification against the paper's intermediate data and ten PDF figures.
[data/manifest.json](data/manifest.json) records compressed and raw hashes,
record counts, settings and provenance. No experiment rerun is needed to use it.

```sh
make figures       # Redraw the ten paper figures.
make verify        # Check every distributed file and decompress/hash raw data.
make release       # Export the verified source+data ZIP to dist/.
```

The export excludes Git/IDE metadata, builds, local outputs, logs and unrelated
experiments. `SHA256SUMS` covers its complete source/data inventory.
[VALIDATION.md](VALIDATION.md) records what was tested and the remaining scope.
The original MIT license is preserved in [LICENSE.txt](LICENSE.txt); third-party
code retains its own notices and licenses.
