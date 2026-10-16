# CPU Accelerate add/sum/softmax candidate

This is the candidate half of the CPU array-math policy experiment. It differs
from the reference report only by `--cpu-math-backend accelerate`. The TileIR
and execution hierarchy are unchanged. Structural contracts may realize exact
contiguous FP32 add/max/min reductions with vDSP and one compiler-materialized
shared FP32 exp map with vForce.

## Command

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-v2 \
  --backends cpu --operations add,sum,softmax \
  --auto-vectorize --cpu-vector-lanes 64 --cpu-stack-bytes 32768 \
  --cpu-input-views --cpu-model native --cpu-math-backend accelerate \
  --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8 --capture-sources
```

## Structural and numerical checks

- All 12 complete outputs pass. Add is bit-exact and reports zero provider
  calls. Sum uses `atol=rtol=1e-5`; all recorded maximum absolute errors are
  zero for these deterministic inputs.
- Softmax reports exactly three semantic static call sites (`max`, `exp`,
  `sum`) per generated kernel. Its maximum absolute error ranges from
  `9.26e-11` to `1.93e-9` under `atol=2e-6`, `rtol=2e-5`.
- The 17-row sum has 17 static call sites because its cheap small root is
  serialized/unrolled. This diagnostic is not a dynamic-call counter.
- The binary and adjacent Tile libraries are fingerprinted; the executable
  SHA256 is
  `a4a902d79d5df2325bcedec49c6c1274bb7da1fdd20f9dd3453d17f45ca4ec3d`.

The candidate medians beat eager Torch for every recorded case, but this
single framework-order sample is not the causal policy comparison. Use the
[six-round replay](../m1-max-20260905-cpu-accelerate-ops-replay/notes.md).
Generated tables and raw evidence are [results.md](results.md) and
[results.json](results.json).

`accelerate` is an explicit relaxed provider policy: vDSP may choose a
different FP32 reduction order and vForce has denormal/floating-exception
differences from scalar libm. It is not silently inferred from the DSL.
