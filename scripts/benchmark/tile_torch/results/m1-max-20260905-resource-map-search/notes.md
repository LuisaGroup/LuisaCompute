# Joint reduction resource and execution search

The search completes all 12 FP32 M1 Max cases: softmax, RMSNorm and LayerNorm
at 23×769, 128×2048, 1024×4096 and 128×8193. It enumerates five cooperating
widths {32,128,256,512,1024} × {reload,cache}, holding V=4/U=1/P=1 and the
64-scalar private budget fixed. There are 101 valid trials, 19 resource/mapping
rejections and 12 freshly JITed winners; all 226 executed native/Torch outputs
pass full validation. Rejected trials are retained, not hidden or timed as
successful kernels.

The [technical report and independent audit](../m1-max-20260905-access-demand-validation/notes.md)
define the features, reference family, frozen selection and acceptance plan.
[Raw results](results.json) contain every trial, source hash, timing sample,
correctness record and realized plan. [Generated tables](results.md) include
fresh post-selection measurements; neither they nor search minima constitute
independent speedup evidence. Sources are archived by SHA256 in `sources/`.

Selection uses no-counter command-buffer GPU throughput, not isolated-kernel
timestamps or host-wall throughput. E2E batched/single-call phases are separate;
instrumented compute-pass samples diagnose probe perturbation. Device inputs
are resident, outputs preallocated natively and for Torch softmax, while Torch
functional norms allocate their outputs inside warm timing. Torch is eager
2.14.0. This is the TIRx/TVM runtime route, not MPP, MPS or direct XIR.

The executable and bridge were built before timing at implementation commit
`d579211f9`. The existing unowned `cooperative.cpp` change and unrelated edits
remain excluded from the commit; the recorded worktree is therefore dirty.
The six W=512/1024×4096 anchor kernels have identical source hashes to the
preceding input-cache checkpoint despite the additional cost facts.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-resource-map-search \
  --backends metal --operations softmax,rmsnorm,layernorm \
  --row-shapes 23x769,128x2048,1024x4096,128x8193 \
  --metal-subgroup-reductions --reduction-programs-per-group 1 \
  --reduction-lane-elements 4 --tune-group-threads 32,128,256,512,1024 \
  --tune-reduction-input-caches reload,cache --tuning-metric gpu-control \
  --samples 5 --sample-ms 10 --warmup-ms 100 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources
uv run --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/analyze.py select
```

Output directories and frozen catalog paths must be new when reproducing.
