# llamatune Night Shift report (experimental)

## Shift summary

- Started: 2026-01-01T22:00:00+00:00
- Ended: 2026-01-02T06:00:00+00:00
- Deadline: 2026-01-02T06:00:00+00:00
- Deadline outcome: completed
- Total benchmark invocations: 42
- Machine: Test CPU / Test GPU
- llama.cpp build: abc1234 (help SHA-256 `ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff`)
- Counts by phase:
  - tune:succeeded: 1
  - calibrate:consistent: 2
  - calibrate:error: 1
  - tune:deferred: 1
- Content groups:
  - qwen-family: representative /models/alpha.gguf
- Warning: llama.cpp build changed since recorded tuning evidence.

## Per-model results

| Model | Fingerprint | Action(s) | Verdict / result |
|---|---|---|---|
| /models/alpha.gguf | `aaaaaaaaaaaaaaaa` | tune | succeeded |
| /models/beta.gguf | `bbbbbbbbbbbbbbbb` | calibrate | consistent (pp -1.2%, tg +0.4%) |
| /models/gamma.gguf | `cccccccccccccccc` | calibrate | transfer-consistent vs aaaaaaaa (pp drift +0.3%, tg drift +0.9%) |
| /models/delta.gguf | `dddddddddddddddd` | calibrate | failed: llama-bench failed |

## Deferred and skipped

| Item | Outcome | Reason | Estimate |
|---|---|---|---|
| /models/epsilon.gguf | deferred | insufficient time for tune-class item | 45.0 min |

## Warnings

- model epsilon skipped: deadline reached
