# Extraction review

This snapshot was extracted on 2026-09-22. See `source_manifest.json` for the
original source revision, dependency versions, and original-file digests.

## Scope and preservation

- The original checkout was read only; its tracked file contents and existing
  changes were compared before and after extraction.
- The standalone build has no absolute source-workspace paths, symlinks to the
  source checkout, submodules, or dependency downloads.
- The PBCE group-3 factory retains the same five labels, predecessor classes,
  epsilon handling, updates, and evaluation logic. General EFR, RT-CFR, and QBCE
  experiment factories and executables are excluded. Shared definitions remain
  in the included learner headers to avoid changing their implementation.
- Chinese comments in the extracted project source were translated into
  English. Translations were checked against noncomment source text.
- Experimental results, logs, notebooks, backup files, IDE settings, compiled
  objects, and the original Git history are excluded from the distribution.
- Third-party licenses and source notices are retained.

## Corrections in the standalone copy

1. The C++ executable defaults to PBCE group 3 and rejects unrelated groups.
   The original non-PBCE factory and its unused flags were removed.
2. A non-finite adaptive metric now sets the process failure flag; a
   `NUMERICAL_ERROR` result cannot produce a successful process exit.
3. The launcher validates positive counts and controller options before
   launching, checks every expected algorithm and requested fixed record,
   rejects failed adaptive statuses and unchanged stale outputs, and reaps
   children on interruption or launch failure. `--dry-run` exposes commands.
4. The Python launcher uses only the standard library. Repository-relative
   executable and EFG paths replace dependencies on the old package and checkout.
5. The unsupported `raise_sizes=[1]` option is removed from `leduc_3p`.
   The bundled Leduc implementation uses raise sizes 2 and 4. This fixes a
   configuration that previously failed to load; it does not reproduce a
   hypothetical raise-size-1 variant.
6. The inherited policy-evaluation test failed in both the original and the
   extracted tree: it equated positive-reach traversal (450 histories) with
   PBCE's complete traversal (3,780 histories). The test now checks full
   information-set coverage and verifies that additional zero-reach nodes have
   zero weights and values. Existing payoff and regret comparisons remain.

## Validation

Validated with GCC 11.4.0, CMake 3.22.1, and Python 3.10.12 on Linux:

- Release build of the standalone source, including a fresh Git checkout.
- Four C++ regression executables: action transformation, decision points,
  policy evaluation, and best responses.
- Eleven Python launcher tests, including subprocess cleanup on SIGTERM,
  malformed/incomplete results, stale outputs, and portable EFG paths.
- Integration tests covering all 16 configured games loading successfully,
  fixed update modes and algorithm threads, adaptive epsilon transitions and
  stage resets, full/partial policy JSONL, and invalid configuration rejection.
- A real seven-job batch: six fixed epsilons plus adaptive mode on two-player
  Kuhn poker, 20 requested iterations, two concurrent jobs and two threads.
- Numerical comparison with the original executable on two-player Kuhn poker
  in fixed and adaptive modes, ignoring timing columns; the results agree.
- No Chinese characters remain in distributed project source, and no source
  checkout paths are required at runtime.

The tests use small games and budgets. The full 100,000-iteration experiment
suite was not rerun. Fixed mode's historical `T` records / `T-1` updates and
the historical algorithm labels remain documented in the README. Adaptive
process completion is distinct from final-stage certification.

To repeat the optional numerical reference comparison when another compatible
executable is available:

```sh
python3 tests/smoke_pbce.py --exe build/bin/run_simultaneous_ltbr_pbce \
  --reference-exe /path/to/original/run_simultaneous_ltbr_pbce
```

The reference executable is optional and is not used by the standalone build
or default test suite. Test outputs are written to temporary directories.
