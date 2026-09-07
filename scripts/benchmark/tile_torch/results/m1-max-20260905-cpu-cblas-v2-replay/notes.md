# Balanced Tile CBLAS / PyTorch / direct CBLAS replay

This report separates two questions: whether a proved TileIR contract can
reach a BLAS-class realization, and what wrapper overhead remains versus a
standalone direct Accelerate call. It freezes the eight schedules from the
current CBLAS plan and does not reuse their timing scores.

## Protocol

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_system.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --system-baseline /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_system \
  --plan scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-plan/results.json \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay \
  --rounds 6 --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8 \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_extra.dylib
```

Each shape receives all six permutations of Tile, eager PyTorch and direct
Accelerate CBLAS order; shape order rotates. Warm host-wall timing includes API
dispatch and synchronization but excludes JIT, allocation and upload. All 48
full outputs pass the common FP64 oracle. No row is discarded, and every
fingerprinted executable/library is unchanged.

## Result

Tile medians are lower than PyTorch on seven of eight shapes. At 1024³ Tile is
984.515 µs, PyTorch 930.311 µs and direct CBLAS 965.743 µs. Paired Tile/direct
CBLAS medians range from 0.995× at 512³ to 1.105× at 128³ for the seven
non-tiny cohorts; 32³ is 1.254× and exposes fixed wrapper cost. The 1024³ paired
range is 0.893--1.234×, so this noisy case is reported by its paired median,
not by its best round.

The full eight-shape table is [results.md](results.md). [results.json](results.json)
contains all timing and latency samples, implementation orders, numerical
errors, cold phases, thread requests, API/storage contracts and hashes.

This demonstrates a reachable provider atom, not a new hand-written CPU
microkernel and not direct-XIR parity. CBLAS is eligible only for the exact
proved whole-kernel subset; other layouts, epilogues, dtypes and partial
contractions still need legal realization families and a break-even model.
