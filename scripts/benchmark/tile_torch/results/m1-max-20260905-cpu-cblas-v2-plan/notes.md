# Proved whole-GEMM CBLAS plan

This report exercises the TIRx CPU whole-GEMM realization. TileIR still
contains an MMA; structural lowering proves the complete compact FP32
`C=A*B` contract, and the target pass revalidates three static rank-two noalias
buffers before emitting one registered `tvm.contrib.cblas.matmul` call.

## Command

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --system-baseline /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_system \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-plan \
  --backends cpu --operations gemm --cpu-model native \
  --cpu-matrix-backend cblas --samples 9 --sample-ms 40 \
  --warmup-ms 200 --threads 8 --capture-sources
```

All eight square, rectangular and ragged FP32 GEMMs pass the complete FP64
oracle. Every generated LLVM module reports exactly one semantic external
matrix call. The Tile executable SHA256 is
`a4a902d79d5df2325bcedec49c6c1274bb7da1fdd20f9dd3453d17f45ca4ec3d`.

In this single session Tile beats eager Torch on seven shapes and trails at
1024³. It remains 0.2--13.3% above the separately timed direct CBLAS median on
seven shapes; wrapper overhead is most visible for 32³/128³. These are initial
observations, not the final comparison. The frozen schedules are replayed in
all six implementation orders in the
[balanced report](../m1-max-20260905-cpu-cblas-v2-replay/notes.md).

See [results.md](results.md) for the generated table and [results.json](results.json)
for samples, cold phases, source hashes, exact ABI/provider metadata and errors.
