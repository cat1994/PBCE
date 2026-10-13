# PBCE / PEFR: code and data

This standalone repository contains the PBCE experiment runner, the five-game
historical dataset, and the plotting pipeline. C++ dependencies are vendored;
no parent-project checkout, submodule fetch or notebook is required.

The submitted manuscript is unchanged. The current implementation contains
memory-weight and signed-evaluation corrections; the bundled measurements
remain historical results, not reruns of this corrected implementation.
See [implementation status](docs/IMPLEMENTATION_STATUS.md).

## Labels: preserve the data, map the figures

Generation and processing retain the original five labels:
`A-EFR_IN`, `CFR`, `CFR_IN`, `CSPS-EFR`, and `CFPS-EFR`.
Figure titles and legends use **ACT-IN, CF, CF-IN, CSPS, and CFPS**.
Record keys, metric columns, data files and generated figure filenames are
not renamed. The names CFR/CFR_IN in stored data must not be confused with
the choice of the CFR+ local regret update.

A display mapping is not a claim of algorithm equivalence. Historical
`CFR/CFR_IN` records used different predecessor classes from the default
CF/CF-IN implementation. Figures explicitly annotate this distinction.
[The label table](docs/ALGORITHM_LABELS.md) gives the exact correspondence.

## Plot the bundled data

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-plot.txt
python3 scripts/verify_release.py
python3 bin/plot_PBCE.py
```

The last command writes five fixed and five adaptive figures to
`outputs/figures/{fixed,adaptive}/`. The original filenames remain compatible
with downstream processing. A plot manifest records input hashes, raw/display
labels, mapping provenance, sampling and adaptive terminal status.

```sh
# CLI selectors accept either raw labels or paper display names.
python3 bin/plot_PBCE.py --mode adaptive --algorithms CSPS CF
python3 bin/plot_PBCE.py --formats pdf png
# Inspect every raw point.
python3 bin/plot_PBCE.py --mode fixed --fixed-stride 1 --max-points 0
# Plot newly generated runs.
python3 bin/plot_PBCE.py --fixed-data outputs/runs --adaptive-data outputs/runs
```

`--label-mode auto` is the default. It identifies bundled historical data by
verified manifest hashes, or reads a run's metadata. For external data, select
`--label-mode historical` or `--label-mode paper` only when its factory mapping
is known. Unidentified mappings are visibly marked as unspecified; an explicit
choice cannot silently override conflicting verified provenance.

Historical fixed sampling retains rounds `t=0,100,...,99900`, then selects
100 evenly spaced points and plots `t+1`. The last plotted round is 99,901.
Adaptive figures use all checkpoints and annotate stage changes and terminal
epsilon. Nonpositive values are masked on logarithmic axes, not replaced in
the data. ACSR and utility-normalized columns are retained in raw records but
are not plotted.

## Build, test and run

Requirements: C++17, CMake 3.16+, a POSIX environment and Python 3.10+.
Matplotlib is needed only for figures; Git is needed for publication preflight.

```sh
make JOBS=4
make test
python3 scripts/verify_release.py --full
# Small seven-job pipeline check.
python3 bin/run_all_pbce.py --games kuhn_2p --iterations 20 \
  --checkpoint-freq 5 --max-parallel-jobs 2 --dir outputs/smoke
# Inspect the default five-game run matrix.
python3 bin/run_all_pbce.py --dry-run
```

Defaults use Kuhn (3 players), Leduc, four-card random-prize Goofspiel, Sheriff
and Tiny Bridge; 100,000 iterations; fixed epsilon
`0.1, 0.05, 0.01, 0.005, 0.001, 0`; adaptive decay `0.5`.
The executable is `build/bin/run_simultaneous_ltbr_pbce`.

`--legacy-algorithm-mapping` selects the historical predecessor classes while
keeping the same raw labels. It does not restore the old weighting or clipped
evaluation behavior, so it is not an exact historical-rerun switch. Per-run
metadata records this choice. See [the label table](docs/ALGORITHM_LABELS.md).

New fixed records with index `t` contain `t+1` evaluated rounds, including the
final record. Fixed and adaptive outputs use signed evaluation accumulators
separate from CFR+ optimization state. A fixed run has one evaluation stage.
Budget completion is distinct from final-stage certification.

New runs default to `outputs/runs/`. Existing outputs are protected, and the
launcher refuses to write into bundled `data/`. Runtime logs and run metadata
can contain the executing machine's paths; they are ignored by Git and should
be reviewed before sharing separately.

## Git publication

```sh
make test
make preflight
```

The publication scope is source, licenses, tests, documentation and the 35
curated compressed data files. Build products, runtime outputs, caches, local
configuration and old archives stay outside Git. No ZIP is required.

The preflight verifies file hashes and the candidate Git file set without
staging, committing or pushing. After staging the intended changes, run
`python3 scripts/check_git_release.py --staged` to check the actual index.
See [the Git release checklist](docs/GIT_RELEASE.md) for checksum maintenance
and the separate review of Git history and hosting metadata before anonymizing.
The optional ZIP export tool remains available for local use.

[data/README.md](data/README.md) describes the unchanged data and conversions.
[VALIDATION.md](VALIDATION.md) records verification scope.
Source-content author identifiers are omitted; this does not anonymize Git
history or hosting metadata. Required third-party copyright and
license notices remain in [LICENSE.txt](LICENSE.txt) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
