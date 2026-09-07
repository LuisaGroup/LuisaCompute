# MPP participation at matched threadgroup geometry

Predeclared exploratory screen, after commit `5be8d24b7`. The question is
whether whole-threadgroup MPP participation supplies a useful missing TIRx
realization. This is not permission to replace the single-subgroup default.

Compare three matched pairs, each with the same output rectangle and total
threads. Independent operations compute 32x32 rectangles. A collective
operation computes the entire rectangle:

| Group rectangle | Threads | Independent subgroup grid | Collective participants |
|---|---:|---|---:|
| 128x64 | 256 | 4x2 | 8 |
| 128x32 | 128 | 4x1 | 4 |
| 64x64 | 128 | 2x2 | 4 |

Shapes: 512x512x512, 4096x4096x4096, 8192x8192x8192,
256x11008x4096, and 2049x4097x1025. Every operation processes physical K
in one MPP invocation. FP32, row-major compact buffers, alpha=1, beta=0,
cooperative output, dynamic K, inline tensors, no fast math or relaxed
precision. The direct MPS matrix API is a seventh arm. This standalone
screen does not execute TIRx and does not claim a same-policy Torch result.

Two rounds: rotate arm/shape order in the first and reverse both orders in
the second. Each shape sees every arm once per round, so every matched pair
has reversed precedence. Five samples per timing phase, requested 20 ms
windows, 100 ms warmup. Save all GPU/E2E batch/single-call samples. GPU time
is a no-counter command-buffer interval per dispatch, not isolated kernel
time. Warm JIT/setup/upload/download are outside timing.

Validate every complete output against the existing FP64 oracle with
atol=rtol=1e-4. Capture each generated MSL source and before/after executable
and driver hashes. Run a full build before the screen, never concurrently
with it. Record failures rather than substituting a different configuration.
Desktop activity is not controlled; two rounds cannot establish accepted
speedup, model calibration or hardware/cache causality. No default changes
follow from a noisy minimum. An admitted candidate needs fresh numerical
tests and an independently frozen replay before performance acceptance.
