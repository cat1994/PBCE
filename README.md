# PBCE / PEFR

Code, data, and plotting scripts for the PBCE experiments. The required C++
dependencies are included.

## Reproduce the paper figures

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m pip install -r requirements-plot.txt
python3 bin/plot_PBCE.py
```

This produces five fixed and five adaptive figures from the included data,
saved in `outputs/figures/{fixed,adaptive}/`. For PNG previews, use
`python3 bin/plot_PBCE.py --formats pdf png`.

Data labels remain `A-EFR_IN`, `CFR`, `CFR_IN`, `CSPS-EFR`, and
`CFPS-EFR`; plot headings are ACT-IN, CF, CF-IN, CSPS, and CFPS.
[data/README.md](data/README.md) gives the file and metric mappings.

## Build and run experiments

Requirements: C++17, CMake 3.16+, Python 3.10+, and a POSIX environment.

```sh
make JOBS=4
python3 bin/run_all_pbce.py --dir outputs/runs
```

Defaults use five games: Kuhn (3 players), Leduc, random-order Goofspiel
(4 cards), Sheriff, and Tiny Bridge. Each algorithm has a 100,000-iteration
budget. Fixed epsilon values are `0.1, 0.05, 0.01, 0.005, 0.001, 0`.
Adaptive runs use initial epsilon `0.1`, decay `0.5`, and minimum epsilon
`1e-10`.

For a short pipeline check:

```sh
python3 bin/run_all_pbce.py --games kuhn_2p --iterations 20 \
  --checkpoint-freq 5 --max-parallel-jobs 2 --dir outputs/smoke
```

Results are written under `outputs/`. Run `python3 bin/run_all_pbce.py --help`
for options, or add `--dry-run` to inspect the run commands.

## Tests

```sh
make test
```
