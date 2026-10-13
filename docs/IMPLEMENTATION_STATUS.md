# Implementation status and manuscript correspondence

This anonymous release distinguishes the bundled historical measurements from
new runs of the corrected implementation. The submitted manuscript and raw
data are unchanged. The corrections below do not claim to complete the
manuscript's intended-device/full-continuation implementation.

## Corrected implementation paths

- `BehavioralDeviationTabularCfvLearner::Update` now propagates two memory
  probability lists, each with current and next-round weights. Solver local
  regret updates and EFR aggregation use the intended policy
  `xi = (x - epsilon) / (1 - n * epsilon)`; original-game regret and exposure
  statistics use realized `x`. The CF/CF-IN all-star memory remains one.
- Fixed output now reads `StageMetrics`, the same signed accumulators used by
  adaptive output. In a fixed run this single stage is the whole run. CFR+
  remains enabled for optimization; the
  legacy `PhiRegrets` API is retained but is not used for PBCE fixed output.
- Fixed and adaptive output both use `w_numerical_eps` as the exposure floor.
  Negative instantaneous gains remain in the evaluation sums, and their
  positive parts are taken only at reporting. Stage resets clear these sums
  and exposure, not the warm-started optimizer state.
- The fixed runner now evaluates all requested rounds, including the final
  record. Record index `t` therefore consistently represents `t+1` evaluated
  rounds. Previously, the final record did not add a new round.
- Fixed records retain 17 significant digits, matching adaptive output, so
  saved fixed/stage statistics can be compared without six-digit rounding loss.

These changes require new experiments; old curves cannot be relabeled as
corrected results. The bundled data remain historical reproduction
artifacts. The legacy-algorithm-mapping flag changes families only, not these
implementation corrections.

## What the archive reproduces

The five paper games, six fixed epsilon values, adaptive decay 0.5, historical
sampling and all three stored diagnostic columns are included. The selected
raw records were compared with the notebook's processed data, and the ten
existing PDF figures were hash-matched to the local manuscript's PDFs.
The plotting script reads raw data directly and uses the same numerical
conversions and sample positions. Its visual styling has been cleaned up;
new PDFs are not expected to be byte-identical to the archived PDFs.

## Differences that remain scientific work

1. **Execution reach.** The intended/realized memory-weight split is now
   implemented. The existing evaluator still supplies a realized-action CFV
   tree. Equivalence of its path/memory enumeration and channel reach factors
   to an explicit perturbed game remains a separate verification task; changing
   the memory coordinate alone is not a proof of that equivalence.
2. **Empirical devices.** The historical runner does not separately accumulate
   and output the intended empirical device `mu_xi` and realized diagnostic law
   `nu_x`. Optional policy sidecars store realized profiles. Stage resets and
   warm starts are implemented, but they are not a substitute for two empirical
   devices.
3. **Full continuation regret.** `StageMetrics` aggregates signed local
   regret-table entries. This is not an independently verified evaluation of
   the manuscript's full continuation deviation objective. In particular, the
   manuscript explicitly says full continuation regret need not equal a single
   local table entry. The names `posr`, `osr` and `csr` are preserved as data
   column names; three populated columns alone do not establish equivalence to
   the three current mathematical definitions. At a CF/CF-IN target, the
   specified continuation resumes obedience, so the local comparison can be
   the relevant full gain. The correspondence must be checked for each
   evaluated deviation and target, rather than rejected solely because local
   tables are used.

These findings are based on `src/hr_edl/tabular_learner.h`,
`src/hr_edl/adaptive_profile.h` and `src/bin/run_simultaneous_ltbr_pbce.cc`,
checked against the manuscript's intended/induced strategy, memory weighting,
empirical-device and full-regret definitions. Resolving them would change the
scientific implementation and requires re-evaluation of the experiments; the
bundled historical data remain available without numerical changes.

## Output labels and historical reproduction

`src/bin/ltbr.h` defines the exact predecessor classes; the single mapping
table is in [algorithm labels](ALGORITHM_LABELS.md). The standalone repository
had already corrected the default CFR/CFR_IN bindings before this cleanup.
The bundled data were generated using the historical
classes. Use launcher option `--legacy-algorithm-mapping` (C++ option
`--legacy_algorithm_mapping=true`) to select them explicitly. Default behavior
has not been silently changed. Per-run metadata records the selection.

Before these corrections, small fixed/adaptive numerical regressions compared all five default algorithms
to the pre-cleanup standalone executable, and all five historical algorithms to
the historical reference executable, excluding elapsed-time columns. Both agreed. This
tested preservation and compatibility on small instances, not theoretical
correctness or a repeat of the full 100,000-iteration experiments.

Generation and processing retain the stored labels. Only the plotting display
uses paper names, with explicit annotations when historical classes differ.
