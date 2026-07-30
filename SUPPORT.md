# Support policy

llamatune is beta software maintained on a best-effort basis. The
[public beta contract](docs/beta-contract.md) defines the supported,
experimental, and excluded surfaces. There is no response-time, resolution, or
compatibility service-level agreement.

## Getting help

Search existing [GitHub issues](https://github.com/barebonescyber/llamatune/issues)
before opening a new bug report or feature request. For a bug, include:

- the llamatune version and commit, when known;
- operating system, architecture, and Python version;
- the llama.cpp build identity and backend;
- the command with model paths and sensitive values redacted;
- the exit code and the smallest relevant sanitized evidence; and
- whether the behavior is reproducible.

Do not post credentials, environment-variable values, private model files, or
unsanitized session directories. Security concerns belong in
[private vulnerability reporting](SECURITY.md), not public issues.

## Scope

Beta-supported defects are triaged against the current beta. Experimental
features may change and do not carry the beta acceptance claim. Excluded
environments may still produce useful diagnostic reports, but the project does
not promise fixes or compatibility for them.

llamatune does not provide llama.cpp builds, models, drivers, CUDA installation,
hardware tuning, benchmark interpretation for unrelated tools, or production
capacity planning.
