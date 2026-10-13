# Validation scope

The submitted manuscript and bundled historical measurements are unchanged.
This release separates corrected new runs from archived results; see
[implementation status](docs/IMPLEMENTATION_STATUS.md).

## Current verification results

- Release build and all seven CTest entries passed.
- All 61 Python checks passed: 19 runner tests, 18 plotting/parser tests,
  20 Git-preflight tests and four no-checksum publication-validation tests.
- All ten five-game figures rendered successfully in PDF, PNG and SVG.
  Their 35 inputs were recognized by hash as the historical archive. Fixed CF
  and adaptive CF-IN previews were visually checked for readable headings,
  provenance notes and unclipped panels.
- All 30 rendered files passed checks for removal of author, generator and
  timestamp metadata. Plot manifests contain no absolute machine paths.
- This Git-publication cleanup changes no solver or plot numeric logic.
  Ten text files were normalized to LF for consistent Git checkouts;
  executable permissions were removed from 338 source files.
  Data-generation and processing labels remain unchanged.

## Solver and runner checks

The C++ tests cover all five deviation families, including intended/realized
memory weights, signed evaluation versus CFR+ state, exposure, stage resets,
next-round aggregation, feasible policies and warm starts. Integration tests
compare fixed and single-stage adaptive statistics at every checkpoint,
including the final round.

These targeted checks do not establish complete equivalence to an explicitly
expanded perturbed game or every full-continuation OSR objective. Full
100,000-iteration experiments have not been rerun. Selecting historical
predecessor classes does not revert the corrected weight/statistic semantics.

## Git publication and labels

Generation labels, record keys, numeric conversions, sampling positions and
figure filenames are preserved. Figure display names follow the paper, with
provenance annotations for historical or unidentified mappings. Plotting tests
cover raw and gzip input, malformed records, metric conversion, sampling,
completion, display aliases and conflicting provenance.

Author-repository links, identifying source revisions and private provenance
are omitted from source content. Required upstream copyright and licenses
are preserved. Git ignores build products, caches, logs, archives and runtime
outputs. The preflight audits the candidate file set or staged index against
the release inventory. It does not anonymize Git history, remotes or hosting
metadata; those require a separate publication review. The 35 gzip headers have no
source filename, timestamp, comment or extra fields, and the decompressed
records were scanned for author identifiers and personal paths.

Publication no longer requires a manually maintained source hash manifest.
The preflight checks the file inventory, data sizes and gzip readability.
Historical data digests remain only as provenance for plot-label identification.
Run `python3 scripts/verify_release.py --full` to additionally parse every
fixed record and adaptive checkpoint.

## Historical data selection checks

Previously completed checks of the selected archive found:

- 150,000 retained fixed array rows and 44,194 adaptive rows matched the raw
  files; adaptive comparisons used relative tolerance 1e-13 and absolute
  tolerance 1e-15.
- All 60,000 plotted fixed-data scalars and the two-step sampling indices
  matched the original notebook exactly.
- All ten original manuscript PDFs matched their archived plotting outputs
  by SHA-256.

These are checks of the historical measurements, not corrected solver reruns.
New figure headings and provenance annotations intentionally change rendered
PDFs without changing their plotted numbers. See [data documentation](data/README.md)
and [label mappings](docs/ALGORITHM_LABELS.md).

The formal publication branch is `main`.
