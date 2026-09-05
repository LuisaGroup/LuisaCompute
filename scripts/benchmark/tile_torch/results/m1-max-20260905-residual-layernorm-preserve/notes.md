# Residual LayerNorm `preserve` source report

Date: 2026-09-05 Asia/Shanghai (`2026-09-05T00:34:03.119110Z` in the raw
report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.

This is the frozen candidate-policy input to the four-round
[materialization A/B replay](../m1-max-20260905-residual-layernorm-materialization-replay/notes.md).
Each shape was captured and JIT-compiled with
`shared_tile_materialization=preserve`. The Metal mapper proves ownership and
realizes the shared `combined` and `centered` Tile definitions as bounded
worker-private stripes rather than repeating their input reads.

| Rows×width | Native µs | eager Torch MPS µs | Native/Torch | Stripe scalars/worker |
|---|---:|---:|---:|---:|
| 1×127 | 3.305 | 10.839 | 0.305× | 4 |
| 17×257 | 3.625 | 11.851 | 0.306× | 6 |
| 128×1024 | 6.142 | 18.458 | 0.333× | 8 |
| 64×4096 | 9.495 | 27.044 | 0.351× | 32 |

Times are p50 warm synchronized host-wall throughput. Native output is
preallocated; eager PyTorch evaluates `layer_norm(X + residual)` and allocates
its returned output, so the external ratio is API-level. The balanced replay,
not this single-session report, is the causal materialization comparison.

Exact command:

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-preserve \
  --backends metal --operations residual_layernorm \
  --metal-subgroup-reductions \
  --shared-tile-materialization preserve \
  --samples 11 --sample-ms 100 --warmup-ms 100 --capture-sources
```

Artifact identity:

- `results.json`: `c96814bb086fd9de2496b129a9c62887938198c479cd8f16ce754b8273106649`
- benchmark executable: `513af14b6c40163bd93aa3c3eb2189784ba14082c86fa29c8e8c7ab61587a266`
- loaded TIRx bridge: `d032d4c687daa52471df167877f2ffa42d5a74fe81aa26a9cd28cdb0ff1ecfe3`

The report predates only the later JSON field for the already-enforced
64-scalar worker-stripe budget. Raw samples, complete-output checks, execution
plans and generated Metal hashes remain in `results.json` and `sources/`.
