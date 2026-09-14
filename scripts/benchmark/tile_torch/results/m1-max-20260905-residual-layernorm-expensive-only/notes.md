# Residual LayerNorm `expensive-only` source report

Date: 2026-09-05 Asia/Shanghai (`2026-09-05T00:33:46.602041Z` in the raw
report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.

This is the frozen reference-policy input to the four-round
[materialization A/B replay](../m1-max-20260905-residual-layernorm-materialization-replay/notes.md).
Each shape was captured and JIT-compiled with
`shared_tile_materialization=expensive-only`: shared transcendental Tiles are
preserved, while the cheap shared `combined` and `centered` arithmetic in
residual LayerNorm is recomputed at its consumers.

| Rows×width | Native µs | eager Torch MPS µs | Native/Torch |
|---|---:|---:|---:|
| 1×127 | 3.699 | 10.663 | 0.347× |
| 17×257 | 3.674 | 11.711 | 0.314× |
| 128×1024 | 8.141 | 18.421 | 0.442× |
| 64×4096 | 13.627 | 26.717 | 0.510× |

Times are p50 warm synchronized host-wall throughput. Native output is
preallocated; eager PyTorch evaluates `layer_norm(X + residual)` and allocates
its returned output, so the external ratio is API-level. The balanced replay,
not this single-session report, is the causal materialization comparison.

Exact command:

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-expensive-only \
  --backends metal --operations residual_layernorm \
  --metal-subgroup-reductions \
  --shared-tile-materialization expensive-only \
  --samples 11 --sample-ms 100 --warmup-ms 100 --capture-sources
```

Artifact identity:

- `results.json`: `8e7ed95d11d244675858c7d1548a1861887e9658d7aecad777d782bcbb9830da`
- benchmark executable: `513af14b6c40163bd93aa3c3eb2189784ba14082c86fa29c8e8c7ab61587a266`
- loaded TIRx bridge: `d032d4c687daa52471df167877f2ffa42d5a74fe81aa26a9cd28cdb0ff1ecfe3`

The report predates only the later JSON field for the already-enforced
64-scalar worker-stripe budget. Raw samples, complete-output checks, execution
plans and generated Metal hashes remain in `results.json` and `sources/`.
