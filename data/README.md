# Historical manuscript data

This directory includes the complete raw metric records selected for the PBCE manuscript figures, compressed without changing their contents. It covers three-player Kuhn poker, two-player Leduc poker, four-card two-player random-order Goofspiel, Sheriff, and Tiny Bridge. Each run contains the historical labels `A-EFR_IN`, `CSPS-EFR`, `CFPS-EFR`, `CFR`, and `CFR_IN`.

- `raw/fixed/`: 30 runs, covering epsilon = 0.1, 0.05, 0.01, 0.005, 0.001, and 0 for each game.
- `raw/adaptive/`: 5 runs with initial epsilon 0.1, decay factor 0.5, minimum epsilon 1e-10, and a maximum budget of 100,000 updates per algorithm.
- `manifest.json`: relative source paths, raw and compressed SHA-256 digests, sizes, schemas, counts, adaptive terminal records, and checks against the original plotting arrays.

The source project is `correlated_eq`. Fixed runs come from its `data_pbce/` root. Adaptive runs come specifically from `data_pbce/data_pbce_gamma0d5/`. The adaptive files in the source `data_pbce/` root use decay 0.75 and are excluded. Runs for other games, alternate solvers, duplicate arrays, logs, and backups are outside this manuscript dataset.

Each `.ssv.gz` can be read directly with Python `gzip.open(path, "rt")` or decompressed with `gzip -dk path.ssv.gz`. Gzip headers use zero modification time and no original filename. The manifest records both the compressed digest and the digest of the exact original `.ssv` bytes.

## Stored record formats

Fixed files are text blocks. Each block starts with `t = N`, followed by one row per algorithm:

```text
algorithm  (ev, milliseconds, max_full_phi_regrets_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret)
```

Every fixed file has 100,000 blocks indexed from 0 through 99,999 and 500,000 algorithm records. The historical runner uses this as 99,999 learner updates plus its initial/final evaluation convention. The original data converter uses:

| Plot field | Raw conversion |
| --- | --- |
| `posr` | `max_full_phi_regrets_fus / (t + 1)` |
| `osr` | `max_full_phi_regret / (t + 1)` |
| `csr` | `max_phi_regret` |

Adaptive files are ordinary CSV despite their historical `.ssv` suffix. The `posr`, `osr`, and `csr` columns already contain the stage-local diagnostics used by the plotting notebook. Do not divide them by `global_t` or `stage_t` again. Plot `CHECKPOINT` rows against `global_t`; use `epsilon_stage` for stage boundaries. `STAGE_END` and `RUN_END` are separate records. Some Tiny Bridge runs terminate before 100,000 updates when their stopping condition is met. Final epsilon and termination status are recorded separately in the manifest; reaching a budget is not itself stage certification.

Fields such as `acsr`, normalized variants, and controller diagnostics are retained because these are unchanged raw records. The manuscript plots use only `posr`, `osr`, and `csr`. ACSR is not a current manuscript evaluation metric.

## Relation to the original figures

The source notebook is `notebooks/pefr_data_simple.ipynb`. Its fixed and adaptive input arrays are `results/results_pbce/pefr_data_fixed.npy` and `results/results_pbce/pefr_data_adaptive.npy` in the source project. Those arrays also contain games not selected by the paper; they are not needed to read this release's raw records.

For the five manuscript games, all 150,000 fixed array rows were checked against the raw files. The six fields `value`, `milliseconds`, `posr`, `osr`, `csr`, and `acsr` agree exactly. All 44,194 adaptive array records match the selected decay-0.5 files by algorithm, record type, global iteration, and stage, with zero discrepancies in epsilon and the three plotted diagnostics at relative tolerance 1e-13 and absolute tolerance 1e-15.

All ten current manuscript figure PDFs also match the server outputs under `results/results_pbce/fixed/` and `results/results_pbce/adaptive/` byte-for-byte by SHA-256. Their file digests are recorded in the manifest. The `adaptive/` figure-output directory is distinct from the excluded root raw-data directory; its figures use the verified decay-0.5 plotting array.

The historical fixed array keeps `t = 0, 100, ..., 99900`. The notebook then selects 100 evenly spaced indices from each resulting 1,000-point curve and plots against `t + 1`. Thus its plotted final fixed x value is 99,901, although the raw files extend to 100,000. To reproduce the historical plot sampling, apply both steps in this order. Plotting all raw points, or selecting 100 points directly from the raw files, gives a different sampling of the same run. The adaptive notebook plots all checkpoint rows and connects adjacent stages by default, with dashed stage-boundary markers. Nonpositive y values are hidden on logarithmic axes rather than replaced.

The adaptive CSPS terminal epsilons match those reported in the manuscript: Kuhn 6.103515625e-6, Leduc 4.8828125e-5, random-order Goofspiel 0.0015625, Sheriff 1.9073486328125e-7, and Tiny Bridge 1e-10.

## Historical implementation scope

The files preserve measurements produced by the historical implementation. They do not contain full intended or realized empirical devices, and the presence of three named columns does not independently establish equality with the current manuscript's full-continuation definitions.

The source PBCE factory maps historical `CFR` to `CounterfactualPartialSequenceExInPredecessors` and `CFR_IN` to `TwiceInformedPartialSequencePredecessors`. By default, the standalone release factory uses `ImmediateExternalSequencePredecessors` and `ImmediateInternalSequencePredecessors` instead. The archived `CFR` and `CFR_IN` curves therefore must not be presented as reruns of those corrected factory entries. Their original labels remain unchanged for traceability.

In the inspected historical learner, reported regret maxima are formed from local regret tables; the fixed perturbed column uses the solver regret table. Stage metrics use stage-local tables. Successor memory probabilities are propagated using the stored execution-policy probabilities. The current manuscript distinguishes intended-memory weights, full continuation regret, and intended/realized empirical devices. Resolving these scientific differences requires a separately validated implementation and new experiments; this archive only preserves and reproduces the historical recorded diagnostics.

The release launcher now provides `--legacy-algorithm-mapping` to select those historical classes explicitly; see [implementation status](../docs/IMPLEMENTATION_STATUS.md).
