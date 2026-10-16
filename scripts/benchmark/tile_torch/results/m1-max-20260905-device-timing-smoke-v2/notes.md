# GPU compute-pass versus end-to-end timing: integration smoke

September 5, 2026; M1 Max; macOS 26.6.2; PyTorch 2.14.0.
Assessment: **integration smoke only, not stable comparative performance**.
The complete inputs/outputs passed the same FP64 oracle for native and Torch.

Follow-up qualification: this ABI-1 capture did not include a no-counter
control. The subsequent [observer audit](../m1-max-20260905-device-timing-counter-control/notes.md)
found substantial probe perturbation on other operators. These historical
counter samples remain unchanged and must not be treated as uninstrumented
kernel times. Current ABI-2 reports expose the separate command-buffer control.

| Case | Native GPU single µs | Torch GPU single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|
| add 17×257 | 4.334 | 6.917 | 194.333 | 212.125 |
| GELU(A+B) 17×257 | 7.667 | 13.792 | 439.167 | 351.958 |

These are three-sample p50s after 50 ms warmup, not balanced repeated trials.
They show that the two metrics are different; they do not establish a stable
speed ranking. GELU has one native kernel versus two eager Torch operations.

GPU time comes from actual Metal compute-pass start/end counter samples,
converted using paired CPU/GPU clock calibration as described by
[Apple](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).
The M1 Max reports stage sampling supported and dispatch/blit-boundary sampling
unsupported. Command-buffer GPU times are also retained as a broader check.
Host samples run first without hooks; device samples are a separate phase.
Device batch counts are capped at 64 independently of host batch counts.

The helper temporarily intercepts public command-buffer encoder factories and
commit, preserving dispatch type and counter attachments, and restores them
after measurement. It does not replace generated code or framework kernels.
The independent device test checks numerical transparency, all three encoder
factories, multiple command buffers, capacity/nesting/empty failures, required
completion and method restoration. A 4096-pass allocation attempt exceeded
this device's counter-buffer limit and failed closed; the caller now requests
1024 passes. That failed attempt is not a performance result.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/device-timing-smoke \
  --backends metal --operations add,gelu_add --row-shapes 17x257 --input-views \
  --samples 3 --sample-ms 10 --warmup-ms 50 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources
```

[Raw samples and hashes](results.json) and [generated report](results.md) are
the evidence. The helper is diagnostic instrumentation: no cross-device,
Metal 4 command-buffer, concurrent-profiler or instruction-only timing claim.
Larger reduction/GEMM cohorts and balanced GPU replays remain follow-up work.
