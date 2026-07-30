# Phase 5 Item 20 — Linux llama.cpp build selection

Status date: 2026-07-25.

Status: **complete; all four binaries built, capability-probed, and
model-backed runtime identities verified**.

## Scope and acceptance

This record selects the two immutable upstream source revisions for the native
Linux reference campaign and records their initial build and capability
validation on the reference host. Item 20 is complete only when the CPU-only
and CUDA `llama-bench` binaries built from both pins have recorded:

- upstream tag, full commit, and llama.cpp build number;
- exact build commands and compiler/toolkit versions;
- backend configuration;
- `llama-bench` SHA-256;
- a successful bounded capability probe; and
- for CUDA builds, successful device discovery on the reference RTX 5080.

The checklist is closed by the evidence below. A release tag or commit is
reproducible provenance, not proof of runtime stability; each recorded binary
was therefore probed independently.

## Selected upstream pins

| Role | Release/build | Full commit | Published | Selection state |
|---|---|---|---|---|
| Stable reference | [`b9637`](https://github.com/ggml-org/llama.cpp/releases/tag/b9637) | [`aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`](https://github.com/ggml-org/llama.cpp/commit/aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3) | 2026-06-14 | Build and provenance qualified |
| Recent reference | [`b10107`](https://github.com/ggml-org/llama.cpp/releases/tag/b10107) | [`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`](https://github.com/ggml-org/llama.cpp/commit/c0bc8591e8815c63cb01dd3f051a8b0df02501c9) | 2026-07-24 | Build and provenance qualified |

Selection rationale:

- Both are official, non-draft, non-prerelease llama.cpp releases whose tags
  resolve directly to the recorded commits.
- `b9637` provides a time-separated comparison point rather than selecting two
  adjacent rolling builds. Its completed build and provenance qualification
  establish it as the stable reference for this campaign.
- `b10107` was the latest official non-prerelease release when this record was
  created. A moving `master` commit was rejected because it would make the
  campaign less reproducible without adding a required test surface.
- Both pins are newer than upstream's `b7824` patched-version floor for
  GHSA-96jg-mvhq-q7q7. This fact does not replace dependency or source
  security checks.

## Required binary matrix

Every row was built from a clean detached source worktree at the exact commit.

| Source pin | Backend | Build number | Compiler/toolkit | `llama-bench` SHA-256 | Probe |
|---|---|---:|---|---|---|
| `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` | CPU, native x86_64 | 9637 | GCC 15.2.1 | `d1c988eaeb9a2105c6cc8847f10f986ce78a7bad1a3e74fb7c2a78ab82aa6b87` | Pass: `aedb2a5e9`, build 9637, backend `CPU` |
| `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` | CUDA, RTX 5080 compute capability 12.0 | 9637 | GCC 15.2.1 / CUDA 13.3.73 | `fbf82593f6c470fa0c53e5cc0ca7587f526d8e04396cae4509debb7f999965e0` | Pass: `aedb2a5e9`, build 9637, backend `CUDA`; CUDA0 detected |
| `c0bc8591e8815c63cb01dd3f051a8b0df02501c9` | CPU, native x86_64 | 10107 | GCC 15.2.1 | `a59ca6c30efbcbc5275402f85e1f17c68749ff74d80789dbb6a08d3232628ed2` | Pass: `c0bc8591e`, build 10107, backend `CPU` |
| `c0bc8591e8815c63cb01dd3f051a8b0df02501c9` | CUDA, RTX 5080 compute capability 12.0 | 10107 | GCC 15.2.1 / CUDA 13.3.73 | `00ae37cdf3e0769f41265007f87bb5040002d2d34a64bd8bcd86945047f2e695` | Pass: `c0bc8591e`, build 10107, backend `CUDA`; CUDA0 detected |

The CUDA configuration explicitly records
`CMAKE_CUDA_ARCHITECTURES=120`, matching the RTX 5080's reported compute
capability 12.0. CMake 4.3 normalizes this to `120a` for CUDA compilation.
The same native CPU and CUDA architecture configuration was applied to both
source pins.

CUDA 13.3 must also receive
`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`. Pinning only the C and C++
project compilers still allowed `nvcc` to select system GCC 16, which CUDA
13.3 rejects. No `-allow-unsupported-compiler` override is permitted.

## Reference host snapshot

- OS: Fedora Linux 44 KDE, native x86_64.
- Kernel: `7.1.4-202.fc44.x86_64`.
- CPU: AMD Ryzen 7 9850X3D, 8 physical cores / 16 logical CPUs.
- GPU on PCI bus: NVIDIA GB203 GeForce RTX 5080 (`10de:2c02`).
- Native-shell CUDA probe: compute capability 12.0, VMM enabled, 15,877 MiB
  device memory reported by llama.cpp.
- NVIDIA kernel module: `610.43.03`; the NVIDIA modules are loaded.
- CUDA toolkit: `13.3.73` at `/usr/local/cuda`.
- C/C++ compiler selected for both builds: GCC/G++ `15.2.1`.
- CMake: `4.3.0`.

Native workstation verification supplied by the operator:

```text
NVIDIA-SMI 610.43.03
KMD Version: 610.43.03
CUDA UMD Version: 13.3
GPU 0: NVIDIA GeForce RTX 5080, 16303 MiB

llama-bench --list-devices
CUDA0: NVIDIA GeForce RTX 5080
compute capability 12.0, VMM: yes, VRAM: 15877 MiB
```

The automated command sandbox used during evidence capture could not
communicate with the NVIDIA device,
which caused the earlier false workstation-blocker assessment. The operator's
native shell and `uv run llamatune scan` both detect the GPU. Campaign device
probes must therefore run in the native shell or another execution context
with device access. No driver, kernel module, clock, governor, cache, or other
system state was changed during this inventory.

## Existing binary disposition

The pre-existing binary at
`$HOME/llama.cpp/build-cuda/bin/llama-bench` is useful only as historical
input:

- SHA-256:
  `77b0fdd401bc04dd6070445efdbd33dd09d7662c04d719aeef8065fa092bf073`;
- ELF x86-64 release build with CUDA enabled;
- build cache records GCC/G++ 15, CUDA 13.3, `GGML_CUDA=ON`, and
  `GGML_NATIVE=ON`;
- `llamatune scan` detects the RTX 5080 with this binary, but reports
  `build_commit: null`, `build_number: null`, and `backends: null`;
- its source directory is not a Git checkout, and repository evidence records
  its embedded build identity as unknown.

It cannot satisfy Item 20 because its exact upstream commit and build number
cannot be proven. It must not be relabeled as either selected pin.

## Reproducible build procedure

The source checkout and build directories belong outside the LlamaTune
repository. The following procedure produced the recorded binaries.

```bash
git clone https://github.com/ggml-org/llama.cpp $HOME/llama.cpp-reference
git -C $HOME/llama.cpp-reference worktree add --detach \
  $HOME/llama.cpp-reference-b9637 \
  aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3
git -C $HOME/llama.cpp-reference worktree add --detach \
  $HOME/llama.cpp-reference-b10107 \
  c0bc8591e8815c63cb01dd3f051a8b0df02501c9
```

For each source worktree, configure a CPU build with:

```bash
cmake -S SOURCE -B BUILD_CPU \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DGGML_NATIVE=ON
cmake --build BUILD_CPU --config Release -j 8 --target llama-bench
```

Configure the corresponding CUDA build with:

```bash
cmake -S SOURCE -B BUILD_CUDA \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON
cmake --build BUILD_CUDA --config Release -j 8 --target llama-bench
```

`SOURCE`, `BUILD_CPU`, and `BUILD_CUDA` are placeholders that must be replaced
with explicit validated paths before execution. Do not use unresolved shell
variables or overwrite the existing `$HOME/llama.cpp/build-cuda`
directory.

## Evidence capture and qualification

For each completed binary:

1. Record `git rev-parse HEAD` from its source worktree.
2. Preserve the complete CMake configure invocation and final configure
   summary.
3. Record the C, C++, and CUDA compiler versions actually selected by CMake.
4. Record `sha256sum` of the exact `llama-bench`.
5. Run `llama-bench --help` with a 10-second wall timeout and bounded capture.
6. Run `llama-bench --list-devices` with the same bounds.
7. Run `llamatune scan --json --llama-bin BUILD_BIN_DIR` as the static
   capability probe. Its help-only probe does not emit llama.cpp build identity.
8. Run a bounded, one-repetition, one-prompt-token model probe and capture its
   JSON identity:

   ```bash
   timeout 180s BUILD_BIN/llama-bench \
     -m REFERENCE_MODEL.gguf -p 1 -n 0 -r 1 --no-warmup \
     -ngl 0 -t 8 -o json
   ```

9. Confirm that the reported build commit/build number agrees with the source
   pin. Any `unknown` identity is a failed provenance gate even when the binary
   hash is present.

All four model-backed identity probes used the local
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and passed. CUDA model probes used
`-ngl 0` so identity measurement did not depend on device access; their
separate bounded `--list-devices` probes detected CUDA0 on the RTX 5080.
Performance and winner validation remain Phase 5 Items 22–25.

## Ownership deviation

The original implementation ownership did not define a Phase 5 campaign
owner. This docs-only record is the conservative minimum needed to complete
roadmap Item 20; it changes no product contract, runtime code, workflow,
dependency, or acceptance result.
