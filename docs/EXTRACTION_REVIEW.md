# Standalone distribution scope

The standalone repository contains the PBCE runner, five benchmark games,
required libraries, tests, plotting code and historical metric data.
It is independent of any source checkout and downloads no build dependencies.

Original labels are preserved for generation and processing. Display labels
and the distinction between current and historical factory mappings are
specified in [ALGORITHM_LABELS.md](ALGORITHM_LABELS.md).
Implementation corrections and remaining verification scope are in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Author-repository URLs and private revision identifiers are omitted from
source content. This does not strip identity from Git history or hosting
metadata; see [GIT_RELEASE.md](GIT_RELEASE.md). Dependency versions and public upstream references remain in
[source_manifest.json](source_manifest.json), and all mandatory third-party
notices are retained. `SHA256SUMS` describes the distributed files, not an
earlier source tree.

See [the validation record](../VALIDATION.md) for completed checks.
