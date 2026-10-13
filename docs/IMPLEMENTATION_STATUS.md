# Implementation status and manuscript correspondence

Audit date: 2026-10-13. The authoritative manuscript is `PBCE.tex` in the
author's paper project. This code/data release preserves historical numerical
behavior while removing unused interfaces. It does not claim to complete the
current manuscript's intended-device/full-continuation implementation.

## What the archive reproduces

The five paper games, six fixed epsilon values, adaptive decay 0.5, historical
sampling and all three stored diagnostic columns are included. The selected
raw records were compared with the notebook's processed data, and the ten
existing PDF figures were hash-matched to the local manuscript's PDFs.
The plotting script reads raw data directly and uses the same numerical
conversions and sample positions. Its visual styling has been cleaned up;
new PDFs are not expected to be byte-identical to the archived PDFs.

## Differences that remain scientific work

1. **Intended and realized strategies.** `_tl::ImmediateDecisionInfo::Update`
   reconstructs an intended coordinate temporarily, updates it and stores the
   realized response. Memory propagation in
   `BehavioralDeviationTabularCfvLearner::Update` (`SuccessorReachProbs`) uses the
   stored realized policies. The current manuscript instead specifies intended
   memory/time-selection weights while the channel changes execution reach.
2. **Empirical devices.** The historical runner does not separately accumulate
   and output the intended empirical device `mu_xi` and realized diagnostic law
   `nu_x`. Optional policy sidecars store realized profiles. Stage resets and
   warm starts are implemented, but they are not a substitute for two empirical
   devices.
3. **Full continuation regret.** `PhiRegrets` and `StageMetrics` aggregate local
   regret-table entries. This is not an independently verified evaluation of
   the manuscript's full continuation deviation objective. In particular, the
   manuscript explicitly says full continuation regret need not equal a single
   local table entry. The names `posr`, `osr` and `csr` are preserved as data
   column names; three populated columns alone do not establish equivalence to
   the three current mathematical definitions.

These findings are based on `src/hr_edl/tabular_learner.h`,
`src/hr_edl/adaptive_profile.h` and `src/bin/run_simultaneous_ltbr_pbce.cc`,
checked against the manuscript's intended/induced strategy, memory weighting,
empirical-device and full-regret definitions. Resolving them would change the
scientific implementation and requires re-evaluation of the experiments; this
release cleanup deliberately keeps the measured numerical behavior available.

## Output labels and historical reproduction

`src/bin/ltbr.h` defines the exact predecessor classes. The paper names below
are label associations, not a claim that every historical class implements
the current paper's named primitive deviation family.

| Stored label | Paper label | Default class | Historical class |
| --- | --- | --- | --- |
| A-EFR_IN | ACT-IN | InformedActionSequencePredecessors | same |
| CSPS-EFR | CSPS | CausalPartialSequencePredecessors | same |
| CFPS-EFR | CFPS | CounterfactualPartialSequencePredecessors | same |
| CFR | CF | ImmediateExternalSequencePredecessors | CounterfactualPartialSequenceExInPredecessors |
| CFR_IN | CF-IN | ImmediateInternalSequencePredecessors | TwiceInformedPartialSequencePredecessors |

The standalone repository had already corrected the last two default mappings
before this cleanup. The bundled data were generated using the historical
classes. Use launcher option `--legacy-algorithm-mapping` (C++ option
`--legacy_algorithm_mapping=true`) to select them explicitly. Default behavior
has not been silently changed. Per-run metadata records the selection.

Small fixed/adaptive numerical regressions compare all five default algorithms
to the pre-cleanup standalone executable, and all five historical algorithms to
the parent-project executable, excluding elapsed-time columns. Both agree. This
tests preservation and compatibility on small instances, not theoretical
correctness or a repeat of the full 100,000-iteration experiments.
