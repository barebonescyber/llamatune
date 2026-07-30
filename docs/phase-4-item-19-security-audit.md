# Phase 4 Item 19 security audit

Status date: 2026-07-27.

Status: **local pre-public audit complete; public-repository CodeQL and GitHub
security evidence remain required before release**.

This document records the local Item 19 release audit and the state observed
in the private development repository. That historical GitHub state is not
evidence for the clean public repository.

## Local results

### Dependency vulnerability audit

The exact `uv.lock` runtime and development dependency set was exported with
`uv export --locked --all-groups --no-emit-project` and audited with
`pip-audit 2.10.1`.

- Packages audited: 34
- Result: no known vulnerabilities
- Note: `pip-audit --locked .` does not recognize `uv.lock`, so the audit used
  the exact fully pinned `uv export` result instead.

### Static security analysis

`bandit 1.9.4` recursively scanned `src`.

- High severity: 0
- Medium severity: 0
- Low severity: 9

The low findings are accepted, already-reviewed process-control sites:
`subprocess` imports and argv-only calls in `executor`, `qualserver`, and
`sandbox`; the fixed Windows `taskkill` fallback; and one invariant assertion
in `search`. Ruff's configured `S` rules independently enforce these sites and
their narrow suppressions. No Bandit finding requires a source change.

### Secret scan

`detect-secrets 1.5.0` scanned every Git-tracked file.

- Initial heuristic findings: 6
- Confirmed secrets: 0
- Final findings after narrow inline false-positive annotations: 0

The false positives were the Windows CPU registry path plus deliberately fake
credential names and values used by tests that prove environment filtering and
session-path non-disclosure.

## Private-development GitHub state observed

The GitHub repository remains private. GitHub API checks on 2026-07-27
confirmed:

- dependency graph and SBOM: enabled and available, with 39 package entries;
- Dependabot alerts: enabled, with zero open alerts;
- Dependabot security updates: enabled; and
- code scanning: unavailable for this private personal repository.

An administrative attempt to enable Code Security returned HTTP 422 with
`Advanced security has not been purchased`. The code-scanning alert API
separately returned HTTP 403 because code scanning is not enabled. These are
plan-availability results, not unresolved repository configuration.

The private development repository's manual `Security audit` workflow ran
against exact `main` commit
`9911036a2731394b35f1bf0c0a8adba8d9d3bb75`. Its dependency audit, Bandit
source scan, and tracked-file secret scan all passed. The CodeQL job was
intentionally skipped with `run_codeql=false` under the approved private-repo
deferral. The private workflow URL is intentionally not published as public
release evidence.

## Certification disposition

The local portion of Item 19 is complete:

- the exact locked dependencies have no known vulnerability;
- the source and tracked-file scans have no unaccepted finding;
- dependency and Dependabot surfaces are enabled and reviewed; and
- the same checks pass in the published GitHub Actions workflow.

The new public repository must rerun all GitHub-hosted checks against its exact
candidate commit. CodeQL remains a mandatory public-release gate, not completed
evidence. Run `Security audit` with `run_codeql=true`, review the resulting
code-scanning alerts, and record the public workflow URL before tagging or
announcing the public beta. Do not enable CodeQL default setup alongside the
committed advanced workflow; the advanced workflow is the evidence path of
record.
