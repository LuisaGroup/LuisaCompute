# Residual LayerNorm bounded Metal thread search

Date: 2026-09-05 Asia/Shanghai (`2026-09-05T00:41:14.181109Z` in the raw
report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.

## Result

This run searches exact 32/64/128/256-thread subgroup realizations with shared
Tile SSA preservation enabled. Each exact width is a separate capture/JIT and
full correctness check; the selected candidate is then captured and measured
again.

| Rows×width | Valid widths | Rejected widths | Measured winner | Fresh native µs | Torch MPS µs | Native/Torch |
|---|---|---|---:|---:|---:|---:|
| 1×127 | 32, 64, 128, 256 | — | 128 | 3.109 | 10.950 | 0.284× |
| 17×257 | 32, 64, 128, 256 | — | 256 | 3.465 | 11.717 | 0.296× |
| 128×1024 | 32, 64, 128, 256 | — | 128 | 5.719 | 18.585 | 0.308× |
| 64×4096 | 128, 256 | 32, 64 | 256 | 9.119 | 26.981 | 0.338× |

At width 4096 the two logical shared Tiles would require 256 stripe scalars
per worker at 32 threads and 128 at 64 threads. Both exceed the explicit
`max_reduction_striped_scalars_per_worker=64` software-state budget and are
rejected before Metal code generation. The valid 128/256-thread candidates
require 64/32 scalars. The bound is not presented as a physical register
count: the final compiler may scalarize or spill it, but the planner must not
silently create unbounded per-thread arrays.

All 14 valid search candidates and four fresh winners pass. The two rejected
candidates remain in `results.json` with their errors and cannot win. The
analytic model chose the measured winner for 64×4096; its diagnostic regret
on the other shapes is retained rather than used to override measurement.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-bounded-thread-search \
  --backends metal --operations residual_layernorm \
  --metal-subgroup-reductions --shared-tile-materialization preserve \
  --tune-group-threads '32,64,128,256' \
  --samples 7 --sample-ms 40 --warmup-ms 80 --capture-sources
```

Artifact identity:

- `results.json`: `e343d47d52f13d8eb9c898b1f24073ab0361d988b8e687de7cd599cf010374a7`
- benchmark executable: `f420ebf571dbb348e813ad5cc040dbe33cd068f16f0adce32801b3d85684a0c7`
- loaded TIRx bridge: `2bab5db74c520268c568643562c8e433b9ad47b1b200fb69ee81ad41463d315c`

The comparison has the same eager-PyTorch/output-allocation qualification as
the materialization report and is not a GPU-event or all-device claim.
