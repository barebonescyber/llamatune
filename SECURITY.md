# Security policy

## Supported versions

Security fixes are provided for the latest published beta only. Older betas,
development snapshots, experimental surfaces, and environments outside the
[public beta contract](docs/beta-contract.md) do not receive backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/barebonescyber/llamatune/security/advisories/new)
and include:

- the affected llamatune version or commit;
- the operating system and Python version;
- the smallest safe reproduction;
- the expected and observed security impact; and
- whether the issue affects a beta-supported or experimental surface.

Do not attach credentials, environment-variable values, private model files,
unsanitized session evidence, or other sensitive workstation data. Redact local
paths and identifiers when they are not material to the report.

The maintainer will acknowledge a complete report when practical, assess its
scope, and coordinate remediation and disclosure through the private advisory.
The project is maintained on a best-effort basis and does not promise a response
or resolution service level.

## Security boundaries

llamatune treats models and benchmark output as data, uses argument-vector child
execution, bounds child runtime and captured output, passes an allowlisted child
environment, confines session writes, and performs no runtime network access.
It does not install models, llama.cpp, drivers, or system tuning.

The optional experimental quality `--exec` path is an accident barrier, not a
security sandbox. Never use it with adversarial generated code. See
[`docs/beta-contract.md`](docs/beta-contract.md) for the complete beta boundary.
