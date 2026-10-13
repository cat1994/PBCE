# Git release checklist

Publication uses the repository source tree; no ZIP submission is required.
Keep the original data-generation and processing identifiers. Only figure
display names use the mapping in [ALGORITHM_LABELS.md](ALGORITHM_LABELS.md).

## Files to publish

The inventory is defined by `scripts/release_files.py`, with `SHA256SUMS`
added to the Git tree. It includes source, build recipes, tests, documentation,
required licenses and all 35 curated data files. Generated figures, new runs,
logs, build products, virtual environments and local configuration are not
part of the release. Keep them in ignored locations; do not force-add them.
Ignored files can remain locally without being published.

Text files use LF line endings; compressed raw data are binary and unchanged.
The checksum inventory describes file contents, not executable permissions or
Git authorship. Preserve required third-party notices and dependency versions.

## Check the candidate and staged content

From the repository root:

```sh
make test
make preflight
git status --short
git diff --check
```

`make preflight` verifies the complete checksum/data inventory and compares it
with tracked and non-ignored untracked working files. It makes no Git changes.
This checks the candidate tree, not what has already been staged or committed.
Review and stage the intended changes, including new documentation and tests,
then check the actual index:

```sh
python3 scripts/check_git_release.py --staged
git diff --cached --stat
git diff --cached --check
```

The staged check rejects missing/extra files, unresolved conflicts, unsupported
file types and staged content that differs from the verified working files.
It does not certify previous commits or publish anything. Commit and push are
separate, deliberate steps after review.

If a published source/documentation file is intentionally edited, review the
changes first and regenerate `SHA256SUMS`:

```sh
python3 -c 'import sys; sys.path.insert(0, "scripts"); from release_files import ROOT, release_files; from verify_release import digest; (ROOT / "SHA256SUMS").write_text("".join(digest(p) + "  " + p.relative_to(ROOT).as_posix() + "\n" for p in release_files()))'
make preflight
```

Do not alter the historical data hashes to bypass a failed data check.
For a deeper data check, run `python3 scripts/verify_release.py --full`.

## Separate anonymity review

The intended workflow is to publish the source Git repository, then expose it
through an anonymous-repository mapping service. Keep the source repository's
history and remote configuration intact; this workflow does not require
inventing anonymous commit authors or rewriting existing commits.

Source-content cleanup does not anonymize Git metadata. Before sharing the
mapped view, check it as an unauthenticated reviewer would see it:

- Commit authors/committers, messages, tags and previously committed files.
- Repository/account URLs, branches, issues, releases, badges and profile links.
- What the mapping service actually exposes, including source-repository
  redirects, file links, downloads, commit history and any cached earlier view.
- Any manually shared logs, run sidecars or generated artifacts, which may
  include local paths even though Git ignores their directories.

Do not assume that renaming a repository or removing a URL from the README
removes identity from its history. Configure and verify the service's anonymous
view separately; do not rewrite or delete history, change remotes, or force-push
as part of routine code cleanup. Required third-party
copyright and licenses must remain in the anonymous source.

The optional `make release` command still creates a local ZIP when needed;
it is not part of this Git publication workflow.
