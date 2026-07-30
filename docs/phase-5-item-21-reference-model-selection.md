# Phase 5 Item 21 — Linux reference-model selection

Status date: 2026-07-25.

Status: **complete; dense, MoE, memory-bound, CPU-feasible, and sharded
reference roles qualified**.

## Scope and acceptance

This record defines the native-Linux model and workload inputs that will be
used with the pinned llama.cpp binaries from
[Item 20](phase-5-item-20-linux-build-selection.md). Model acquisition and
splitting were explicit operator actions; LlamaTune runtime paths retain their
no-network and model-data-only contracts.

The public-beta Linux checklist requires coverage of:

- dense and MoE architectures;
- memory-bound and CPU-feasible models; and
- a genuine multi-file sharded model.

A model may cover more than one role, but a role is credited only when its
local artifacts and header metadata prove that property. Merely loading on the
CPU does not make a 20+ GB model the deliberately CPU-feasible acceptance
case, and a filename containing `merged` does not count as sharded evidence.

## Qualification method

LlamaTune's header-only `inspect_model` path produced the metadata and fast
fingerprints below. Per `DESIGN.md` section 5, the fingerprint hashes the
first and last MiB, file size, and identity metadata; a whole-file hash is not
required for the default model identity.

Each existing model also completed this bounded smoke workload with the CPU
binaries from both pinned llama.cpp releases:

```bash
timeout 180s BUILD_BIN/llama-bench \
  -m MODEL.gguf -p 1 -n 0 -r 1 --no-warmup \
  -ngl 0 -t 8 -o json
```

The smoke run proves that both builds can parse, load, and execute the model.
Its throughput is not campaign performance evidence.

## Existing local candidates

| Model | Size | Architecture | Layers / experts | Fast fingerprint | b9637 CPU | b10107 CPU |
|---|---:|---|---:|---|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,134,528,992 B | `qwen35moe` | 40 / 256 | `12f6fcd83a6df5928afad8adabee6b255b003ad6c33107b4324963a930a86a4b` | Pass | Pass |
| `Qwen3.6-35B-A3B-UD-Q6_K.gguf` | 29,308,320,736 B | `qwen35moe` | 40 / 256 | `d6fce677855346801d3c8f71c8ef8401736f914ab5b2aa05a2ea57ae0e886fbd` | Pass | Pass |
| `Qwen3-Coder-Next-UD-Q5_K_XL-merged.gguf` | 59,541,180,512 B | `qwen3next` | 48 / 512 | `113e8de0a3a19e925078b6fd104e386f250b3eb9525debc27f0f31e76e174121` | Pass | Pass |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 B | `qwen2` | 28 / 0 | `54dbd5eb4d61d99fc131d1374d3f65ad6fd9becd0573beafb9c9b37ece26b2bf` | Pass | Pass |
| Qwen2.5 Q4_K_M two-shard set | 1,117,320,928 B | `qwen2` | 28 / 0 | `922400bcaaa41f7323cc566412c1f88638d80c3d2a334f9721fe33ff73e2b39d` | Pass | Pass |

Additional inspected properties:

| Model | Expert tensor bytes | Dense tensor bytes | f16 KV bytes/token | KV-bearing layers |
|---|---:|---:|---:|---:|
| Qwen3.6 Q4_K_M | 19,568,525,312 | 2,555,013,632 | 81,920 | 40 |
| Qwen3.6 Q6_K | 26,619,150,336 | 2,678,180,352 | 81,920 | 40 |
| Qwen3-Coder-Next Q5_K_XL | 56,845,402,112 | 2,689,788,928 | 98,304 | 48 |
| Qwen2.5 1.5B Q4_K_M | 0 | 1,111,370,240 | 28,672 | 28 |

The three original files are single-file MoE artifacts and exceed the
reference GPU's 15,877 MiB reported VRAM. They therefore exercise memory
pressure and joint GPU/MoE placement. The added 1.1 GB Qwen2.5 file is a
single-file dense, CPU-feasible reference.

## Selected roles

### Primary memory-bound MoE reference

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

- Smallest current local model, reducing campaign runtime while retaining
  real partial-offload and MoE-placement pressure.
- Existing sessions demonstrate successful CPU fallback, partial placement,
  full layer offload with CPU-resident experts, and confirmed improvements.
- Primary model for the stable-versus-recent build comparison.

### Large MoE stress reference

`Qwen3-Coder-Next-UD-Q5_K_XL-merged.gguf`

- Distinct `qwen3next` architecture with 512 experts and a 59.5 GB artifact.
- Expert tensors dominate its storage, making it the stronger CPU-MoE
  placement and host-memory stress case.
- The file is explicitly merged and single-file; it does not satisfy the
  sharded role.

### Quantization control

`Qwen3.6-35B-A3B-UD-Q6_K.gguf`

- Same architecture, layer count, expert count, and KV geometry as the Q4
  primary with a larger weight representation.
- Retained as an optional memory-pressure/quantization comparison.
- It does not add an independent architecture role and is not required in
  every build-comparison run.

### Dense and CPU-feasible reference

`qwen2.5-1.5b-instruct-q4_k_m.gguf`

- Official Qwen Q4_K_M artifact pinned to model-repository revision
  `91cad51170dc346986eccefdc2dd33a9da36ead9`.
- Downloaded size: 1,117,320,736 bytes.
- Published and verified full-file SHA-256:
  `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
- Header inspection reports `qwen2`, 28 layers, zero experts, and 1,111,370,240
  dense tensor bytes.
- Both pinned CPU builds completed the bounded smoke workload against the
  exact file.

## Coverage disposition

| Required role | Current disposition | Evidence / gap |
|---|---|---|
| MoE | Covered | Both selected references have non-zero expert metadata |
| Memory-bound | Covered | Both selected references exceed available GPU VRAM |
| Dense | Covered | Qwen2.5 reports `moe: false` and zero expert tensors |
| CPU-feasible | Covered | The 1.1 GB Qwen2.5 original passes both CPU build probes |
| Sharded | Covered | Both builds load the set; LlamaTune validates and fingerprints both shards |

## Selection criteria applied

The gap-closing selection used the fewest additional downloads:

1. A single-file dense GGUF small enough to complete the full CPU-only
   `scan`/baseline/tune/report/export/matrix workflow routinely. Prefer an
   artifact no larger than 4 GiB so CPU acceptance remains practical.
2. A genuine multi-file GGUF shard set supported by both pinned llama.cpp
   builds. The group must remain split during qualification; a merged copy
   does not exercise shard discovery.

The dense model satisfies CPU-feasible coverage, and its separately generated
shard set exercises llama.cpp and LlamaTune shard discovery without duplicating
an existing 20–60 GB model. The repository and LlamaTune runtime do not
download models automatically.

## Gap-closing asset evidence

The selected additional asset is the official
[`Qwen/Qwen2.5-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
`qwen2.5-1.5b-instruct-q4_k_m.gguf` file:

- published by the Qwen organization under Apache-2.0;
- dense 1.54B-parameter transformer with 28 layers;
- 32,768-token context support stated by its model card;
- approximately 1.12 GB for the Q4_K_M artifact; and
- supported through llama.cpp's ordinary GGUF path.

Its header and bounded CPU probes pass, so it fills both the dense and
CPU-feasible roles. The b9637 and b10107 build systems both expose the
official `llama-gguf-split` target. The b9637 tool created a separate
two-file shard set while preserving the verified original:

```bash
llama-gguf-split --split-max-size 600M \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  qwen2.5-1.5b-instruct-q4_k_m-split
```

The resulting artifacts are:

| Shard | Size | SHA-256 |
|---|---:|---|
| `qwen2.5-1.5b-instruct-q4_k_m-00001-of-00002.gguf` | 599,458,880 B | `16e585b421871a28c7a99a20409b116ce54616751873743296c2673807d5d903` |
| `qwen2.5-1.5b-instruct-q4_k_m-00002-of-00002.gguf` | 517,862,048 B | `d1a18dd500d195eb1d72cca36ffa0e82d52d9e999cf3aa09c3151f456de92ebf` |

llama.cpp automatically discovered shard 2 when given shard 1, and both b9637
and b10107 completed the bounded smoke workload. The original plus shards use
about 2.24 GB, avoiding duplication of a 20–60 GB model.

## Sharded-identity remediation

Qualification exposed that the previous `inspect_model` implementation treated
the supplied path as the entire model. On shard 1 it recorded:

- `size_bytes: 599458880`, not the 1,117,320,928-byte shard-set total;
- tensor-byte totals from shard 1 only; and
- a fast fingerprint derived from shard 1 only.

The GGUF metadata exposes `split.no=0`, `split.count=2`, and
`split.tensors.count=339`. `inspect_model` now resolves the complete ordered
set, validates every member, aggregates size and tensor accounting, and
includes every shard in deterministic fast and full hashes. It rejects a
non-first entry path and missing, inconsistent, or out-of-directory members
before benchmarking.

The real two-shard set now reports:

- aggregate size: 1,117,320,928 bytes;
- aggregate dense tensor bytes: 1,111,370,240; and
- aggregate fast fingerprint:
  `922400bcaaa41f7323cc566412c1f88638d80c3d2a334f9721fe33ff73e2b39d`.

`DESIGN.md`, `model.py`, and model/discovery regression tests carry the same
contract. Single-file fingerprint and full-file SHA-256 behavior remains
unchanged.

## Campaign workload identity

The first performance campaign uses the workload already established by the
project's RTX 5080/Qwen remediation plan:

- prompt processing: 512 tokens;
- generation: 128 tokens;
- required context validation: 8,192 tokens;
- stable and recent llama.cpp builds tested separately;
- CPU and CUDA backends recorded separately;
- default behavior attempted before any safe CPU fallback; and
- exact model fingerprint, llama.cpp commit/build, and binary SHA-256 retained
  in evidence.

Stretch context begins at 16,384 and 32,768 tokens only after the required
8,192-token validation succeeds. These rungs are context evidence, not
substitutes for the 512/128 comparison workload. Performance runs, winner
confirmation, and cross-build conclusions belong to Phase 5 Items 22–25.

## Existing evidence disposition

The July 17 sessions for the Q4 and Coder models are useful historical
feasibility evidence, but their llama.cpp identity was recorded as unknown.
They must not be relabeled as results from either Item 20 build. New campaign
sessions must use the exact pinned binaries and hashes.

## Ownership deviation

The original implementation ownership did not define a Phase 5 campaign
owner. Item 21 owns this selection record plus the narrowly required
shard-identity changes in `DESIGN.md`, `model.py`, and `test_model.py`. It
changes no benchmark, executor, session-path, workflow, or dependency
contract.
