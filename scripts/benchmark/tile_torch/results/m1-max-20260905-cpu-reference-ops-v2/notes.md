# CPU reference add/sum/softmax control

This report is the reference half of the CPU array-math policy experiment. It
uses the current TileIR→TIRx→LLVM implementation with native M1 codegen,
read-only input forwarding, a 64-scalar Cartesian pack budget and a 32768-byte
compiler-temporary stack budget. `CpuMathBackend` remains `reference`.

## Command

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-reference-ops-v2 \
  --backends cpu --operations add,sum,softmax \
  --auto-vectorize --cpu-vector-lanes 64 --cpu-stack-bytes 32768 \
  --cpu-input-views --cpu-model native --cpu-math-backend reference \
  --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8 --capture-sources
```

## Result and scope

All 12 complete outputs pass. The generated LLVM reports no external vector-
math provider calls. The binary SHA256 is
`a4a902d79d5df2325bcedec49c6c1274bb7da1fdd20f9dd3453d17f45ca4ec3d`.
Raw samples, cold/setup phases, source hashes, exact policies and errors are in
[results.json](results.json); the generated table is [results.md](results.md).

This single-session report is not used by itself to claim a speedup. Its frozen
plans are paired against the Accelerate policy in the six-round
[counterbalanced replay](../m1-max-20260905-cpu-accelerate-ops-replay/notes.md).
The reference path remains the semantic control even where the provider is
faster.
