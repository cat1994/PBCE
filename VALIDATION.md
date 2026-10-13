# Release validation — 2026-10-13

Environment: Ubuntu, GCC 11.4.0, CMake 3.22.1, Python 3.10.12,
NumPy 1.26.4, Matplotlib 3.7.5.

## Completed checks

- Release build and all seven CTest entries passed: four inherited C++
  regression tests, the new PBCE invariants test, Python tests, and integration
  smoke tests.
- The Python suite includes 19 launcher checks and 10 plotting/parser checks.
  Checks exercise output protection, path independence, child cleanup,
  complete/finite results, metric conversion, raw/gzip inputs, sampling order,
  terminal epsilon, malformed rows and missing/invalid terminal records.
- PBCE invariants cover all five algorithms in default and historical modes
  on three-player Kuhn: feasible action probabilities, preservation of realized
  policies when epsilon shrinks, stage resets and subsequent learner updates.
- Default-mode fixed and adaptive small-game outputs agree numerically with
  the pre-cleanup standalone executable for all five algorithms. Historical
  mode agrees with the parent-project executable. Each integration suite
  passed six tests; timing columns are excluded from numerical comparison.
- The actual launcher ran six fixed jobs plus one adaptive job in each mapping
  mode from an unrelated working directory. Output completeness and binary
  metadata were checked; reusing the output directory was rejected as expected.
- All 35 compressed data files match their manifest and decompress to the
  exact original raw SHA-256 values. The source files remained unchanged.
- The paper's 150,000 retained fixed data rows and 44,194 adaptive rows were
  checked against the selected raw files. All ten original paper PDFs matched
  their server counterparts by SHA-256. See `data/manifest.json` for details.
- All 60,000 plotted fixed-data scalars (five games, six epsilons, five
  algorithms, 100 samples, x plus three metrics) equal the original notebook's
  sampled values exactly. The two-step sampling indices also agree exactly.
- The new script generated all ten PDFs and PNG previews directly from bundled
  gzip inputs. The five-game fixed and adaptive CSPS grids were visually
  inspected for clipping, labels, stages, scales and legends.
- The identical plotting script and its ten parser tests passed in
  `correlated_eq`, including a five-game adaptive figure using its default data
  paths. Script hashes match across the two repositories.
- `verify_release.py --full` passed for all 1,779 release files and 35 data
  archives, including full parsing of fixed records and adaptive checkpoints.
- The source/data ZIP was extracted into a separate clean directory, verified,
  configured and compiled without downloading dependencies; all seven CTest
  entries passed again. The final documentation/checksum update does not change
  any of that tested source code.

## Scope

The full 100,000-iteration experiments were not rerun. Numerical regressions
verify small-game preservation; archive verification and figure reproduction
use complete existing raw data. Neither proves equivalence of the inherited
metrics and memory propagation to the current manuscript definitions. The
remaining scientific differences are in `docs/IMPLEMENTATION_STATUS.md`.

No GitHub push or public release upload is part of this local preparation.
