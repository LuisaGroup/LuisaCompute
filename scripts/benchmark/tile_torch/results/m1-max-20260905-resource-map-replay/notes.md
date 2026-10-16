# Frozen joint resource and mapping replay

Four balanced rounds replay all 12 preselected softmax/RMSNorm/LayerNorm
configurations on Apple M1 Max. All 96 native/Torch measurements and 192
complete outputs pass executed validation; the independent raw-timing,
source/plan, coverage and artifact audit passes. No parameter search or
minimum-of-rounds selection happens during acceptance.

The [technical report](../m1-max-20260905-access-demand-validation/notes.md)
contains results, uncertainty, controls, cost-feature definitions and next
decisions. [Raw samples](results.json), [generated GPU/E2E tables](results.md)
and SHA256-addressed `sources/` are the supporting evidence. The
[audit receipt](../m1-max-20260905-access-demand-validation/audit.txt) includes
paired GPU/E2E gain ranges and single-call comparisons for every case.

The reference is the best valid reload width in the same five-width search;
the candidate is the joint reload/cache × width winner. Both are frozen
catalogs, not the old automatic planner or the prior fixed-width baseline.
Seven changed-source cases improve in every GPU pair; four have mixed GPU
pairs, though their medians improve. All 11 improve in every E2E throughput
pair. The unchanged smallest LayerNorm is retained as a control.

GPU means no-counter command-buffer execution, **not isolated-kernel time**.
Separate E2E batched and synchronized single-call phases include dispatch.
Their medians are not subtracted to infer overhead. Torch is eager 2.14.0
with the recorded operator/output-allocation policy. All inputs are resident;
JIT, transfer and validation time are outside warm samples. This is TIRx/TVM
runtime evidence, not native MPP/MPS or XIR evidence.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/reference-plan.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/candidate-plan.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-resource-map-replay \
  --operations softmax,rmsnorm,layernorm --rounds 4 \
  --samples 9 --sample-ms 30 --warmup-ms 200 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact cmake-build-tirx/bin/libluisa-tile-bridge-tirx.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime_metal.dylib \
  --capture-sources
uv run --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/analyze.py audit
```

The output directory must be new. Replace local TVMx artifact paths only with
the compiler/runtime actually used by the rebuilt benchmark; do not reuse
this report's hashes for a different installation.
