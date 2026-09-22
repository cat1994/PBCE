# Source and license notices

The original MIT license and copyright notice are preserved in [LICENSE.txt](LICENSE.txt).
This repository extracts the PBCE experiment from the working tree of
`cat1994/correlated_eq` at commit `5caf894221c7569ff3cf99b24d5769cfa7004717`,
including the then-current PBCE launcher changes. It is a source snapshot with
its own Git history; it does not depend on the original checkout.

The exact source-file SHA-256 values before extraction edits and dependency
provenance are recorded in [docs/source_manifest.json](docs/source_manifest.json).
These are provenance hashes, not checksums of the translated distribution files.

## OpenSpiel

`src/open_spiel`, excluding `abseil-cpp`, contains the necessary C++ runtime,
game implementations, and test dependencies from the original working tree.
The original build recipe identifies upstream base commit
`53eeb6127578f91e88a6c0451983d10e3509446d` of
<https://github.com/google-deepmind/open_spiel>. The original checkout contains
project changes, so this subset is not claimed to be byte-identical to upstream.

OpenSpiel is licensed under Apache License 2.0; see
[src/open_spiel/LICENSE](src/open_spiel/LICENSE). Copyright notices are retained
in the individual source files. The upstream build system is replaced by the
repository's root CMake configuration. Liar's Dice and the tabular best-response
reference code are included for regression tests; Tic-Tac-Toe is required by
the minimax implementation used by Tiny Bridge.

## Abseil

`src/open_spiel/abseil-cpp` vendors Abseil release `20200923.3`, commit
`6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c`, from
<https://github.com/abseil/abseil-cpp> under Apache License 2.0.
See [its license](src/open_spiel/abseil-cpp/LICENSE).

The existing local portability patch in `absl/debugging/failure_signal_handler.cc`
is retained: both arguments to the stack-size `std::max` expression use `size_t`,
allowing compilation with modern libc definitions of `SIGSTKSZ`. The rest of
the tracked dependency snapshot is preserved; its tests are disabled and only
libraries needed by PBCE are built. Nested Git metadata is not included.

## Eigen

`src/eigen/Eigen` contains the header-only Eigen library at commit
`39ec31c0adbdde6b8cda36b3415e9cc2af20dab6` from
<https://gitlab.com/libeigen/eigen>. The source headers retain their individual
license notices. The accompanying `COPYING.*` files are included in
[src/eigen](src/eigen), including MPL 2.0 and the licenses for individual
components. Eigen examples, benchmarks, tests, and build products are omitted.

No third-party dependency is downloaded by the PBCE build.
