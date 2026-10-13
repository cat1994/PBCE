# Algorithm labels and figure display names

The original identifiers remain unchanged in generated results, parser keys,
JSONL policy files, data manifests and figure filenames. Only human-facing
figure titles and legends map to the terminology of the manuscript.

| Raw identifier | Figure display name | Default predecessor class | Historical predecessor class |
| --- | --- | --- | --- |
| A-EFR_IN | ACT-IN | InformedActionSequencePredecessors | same |
| CFR | CF | ImmediateExternalSequencePredecessors | CounterfactualPartialSequenceExInPredecessors |
| CFR_IN | CF-IN | ImmediateInternalSequencePredecessors | TwiceInformedPartialSequencePredecessors |
| CSPS-EFR | CSPS | CausalPartialSequencePredecessors | same |
| CFPS-EFR | CFPS | CounterfactualPartialSequencePredecessors | same |

The default CF and CF-IN classes correspond to a sequence of external
transformations before the target, followed by an external or internal
transformation at the target; their predecessor memory is all-star.
The two historical classes in those rows retain additional memory structure.
Thus a historical curve displayed under the CF or CF-IN heading is explicitly
annotated as a historical-class result, not an evaluation of the default class.

## Factory choice versus update rule

- Normal runs use the default predecessor column.
- `--legacy-algorithm-mapping` selects the historical column without renaming
  any raw identifiers. C++ uses `--legacy_algorithm_mapping=true`.
- `--cfr-plus` / C++ `--cfr_plus` controls regret-state truncation; CFR+ is an
  optimization rule, not a sixth deviation family or a replacement label.
- The bundled historical measurements used the historical column. The same
  raw text labels alone cannot identify a factory mapping for other files.

## Plotting provenance

The plotting command accepts raw identifiers or paper names as selectors,
but keeps parsed records under their original keys and leaves numbers intact.
`--label-mode auto` checks the bundled data manifest hashes or a matching
launcher metadata file. Unidentified inputs are marked as unspecified.

An explicit `--label-mode historical` or `--label-mode paper` may describe
external files whose generating factory is known; contradictory verified
metadata is rejected. Neither flag alters the numeric data or makes the two
factory classes equivalent.

Each figure and its plot manifest record the applicable mapping. Original
filenames, such as `pefr.CFR.adaptive.gaps.pdf`, are retained for compatibility;
their displayed heading is CF, with the applicable provenance annotation.
