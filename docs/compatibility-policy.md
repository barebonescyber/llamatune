# Compatibility policy

## Scope

This policy defines when a llamatune result may be reused during the public
beta. [`DESIGN.md`](../DESIGN.md) remains authoritative for implementation
behavior, and [`beta-contract.md`](beta-contract.md) defines the narrower beta
support claim.

## Runtime support

The beta-supported runtime is CPython 3.11 or newer on the exact operating
systems and architectures listed in the beta contract, limited to Python
versions exercised by the release CI. A wheel is platform-independent Python
code, but that does not promote an unvalidated operating system or backend into
the support claim.

llama.cpp compatibility is capability- and identity-based, not version-range
based. The supplied binary must expose the required flags, and its recorded
help hash, build commit/number, and backend identity distinguish compatible
results. llamatune does not install, update, or build llama.cpp.

## Recommendation identity

A recommendation is reusable only when all material identity dimensions remain
compatible:

- model fingerprint and shard layout;
- hardware group and operating-system environment;
- llama.cpp binary/build/backend identity;
- tuning target and prompt/generation workload;
- KV depth; and
- required validated context.

A change in any dimension requires a fresh lookup and normally revalidation or
retuning. A larger requested context cannot silently reuse evidence validated
only at a smaller context. Experimental multi-GPU, lossy-KV, calibration,
orchestration, and quality evidence remains labeled and does not become
beta-supported merely because the other identity dimensions match.

## Evidence and schemas

Session and result schemas are versioned. Existing evidence is never silently
rewritten to resemble a newer schema. Readers may perform documented,
conservative in-memory migration of older schemas; unsupported newer, corrupt,
or incomplete evidence is rejected or reported as such.

Patch releases may add tolerant reader fields without changing established
meaning. A release that intentionally changes identity, scoring, confirmation,
or evidence semantics must document the change and require revalidation of
affected recommendations.

## Dependency compatibility

The authoritative Python and dependency constraints are in `pyproject.toml`
and the release lockfile. Supported installation uses a released wheel or
source distribution; development checkouts use the locked development
environment. Upstream dependency or llama.cpp behavior outside those recorded
constraints is not implied to be compatible.
