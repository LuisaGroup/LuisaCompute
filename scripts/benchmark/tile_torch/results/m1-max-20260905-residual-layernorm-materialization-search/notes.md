# Residual LayerNorm shared-Tile materialization search

Date: 2026-09-05 Asia/Shanghai (`2026-09-05T00:39:21.476557Z` in the
raw report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.

## Question and result

This run asks whether a multi-consumer pure Tile should be retained as one
logical SSA definition or cloned into its consumers. Each policy is an
ordinary separately captured and JIT-compiled candidate:

- `preserve` materializes every multi-consumer pure Tile in structural TIRx;
  the Metal mapper may compact that logical object to a worker-private stripe;
- `expensive-only` preserves shared transcendental Tiles but recomputes shared
  cheap arithmetic, matching the earlier lowering policy.

All eight search candidates and all four fresh winner measurements pass a
complete FP64 oracle. Measurement selected `preserve` for every shape.

| Rows×width | Preserve trial µs | Expensive-only trial µs | Fresh winner µs | Eager Torch MPS µs | Winner/Torch | Worker stripe scalars |
|---|---:|---:|---:|---:|---:|---:|
| 1×127 | 3.511 | 3.751 | 3.426 | 10.671 | 0.321× | 4 |
| 17×257 | 3.646 | 3.712 | 3.655 | 11.705 | 0.312× | 6 |
| 128×1024 | 6.074 | 8.352 | 6.321 | 18.592 | 0.340× | 8 |
| 64×4096 | 9.614 | 13.811 | 8.324 | 27.046 | 0.308× | 32 |

Times are p50 warm synchronized host-wall throughput. Inputs remain on the
device; compilation, upload and download are excluded. Native writes a
preallocated output. PyTorch evaluates eager `layer_norm(X + residual)` and
returns an allocated output, so Winner/Torch is an API-level comparison, not
an isolated-kernel claim. Raw samples and output policies are in
`results.json`.

Maximum native absolute error is `3.47e-7`. The four final rows contain
397,712 checked output elements in total; search candidates are independently
checked before they may win.

## Structural finding

The kernel is expressed without a memory or hardware special case:

```cpp
auto combined = X[origin, shape(one, columns)] +
                residual[origin, shape(one, columns)];
auto mean = reduce(combined, columns, add) / width;
auto centered = combined - mean;
auto variance = reduce(centered * centered, columns, add) / width;
Y(origin, shape(one, columns))
    .store(centered / sqrt(variance + 1e-5f));
```

Under `expensive-only`, generated Metal expands `combined` once in the mean,
twice in the squared-deviation expression and once in the output: each input
element is loaded four times. Under `preserve`, `combined` and `centered` are
each defined once and become two compact worker stripes. At width 4096 and
256 workers those are two `float[16]` arrays; each input element is loaded
once. No source-level manual `Memory` was introduced.

The existing analytic reduction score does not yet price duplicated global
loads or expression depth. It therefore selected `expensive-only` as its
diagnostic model choice, with measured regret 6.82%, 1.80%, 37.51% and 43.66%
across the four shapes. The measured staged/JIT authority correctly selected
`preserve`. This is direct evidence for adding memory-traffic and recomputation
features to the cost model; it is not hidden by relabeling the measured result
as a model success.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search \
  --backends metal --operations residual_layernorm \
  --metal-subgroup-reductions \
  --tune-shared-tile-materializations preserve,expensive-only \
  --samples 9 --sample-ms 60 --warmup-ms 100 --capture-sources
```

Artifact identity:

- `results.json`: `c3b248ab564c18062e6d92e3d5564ae8be0a7e2ddd0fd182f4b73e1e93437dbf`
- benchmark executable: `f420ebf571dbb348e813ad5cc040dbe33cd068f16f0adce32801b3d85684a0c7`
- loaded TIRx bridge: `2bab5db74c520268c568643562c8e433b9ad47b1b200fb69ee81ad41463d315c`

The report records the dirty working tree and exact generated-source hashes.
These measurements establish this FP32 M1 Max cohort only; they do not cover
backward normalization, affine parameters, low precision or other GPUs.
