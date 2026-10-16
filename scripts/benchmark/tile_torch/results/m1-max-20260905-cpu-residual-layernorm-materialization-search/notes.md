# CPU residual LayerNorm materialization search

Date: 2026-09-05 Asia/Shanghai (`2026-09-05T00:39:47.241157Z` in the
raw report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.

## Result

This is the CPU counterpart of the Metal shared-Tile search. It uses native
LLVM CPU code generation, proved immutable input views, automatic independent-
element packing and a 64 KiB compiler-local stack budget. The two candidates
are separately captured, JIT-compiled and fully validated. CPU selected
`expensive-only` for every shape: recomputation and LLVM fusion were cheaper
than preserving the two width-sized intermediates.

| Rows×width | Preserve trial µs | Expensive-only trial µs | Fresh winner µs | Eager Torch CPU µs | Winner/Torch |
|---|---:|---:|---:|---:|---:|
| 1×127 | 0.281 | 0.252 | 0.252 | 2.314 | 0.109× |
| 17×257 | 9.169 | 8.745 | 8.799 | 39.125 | 0.225× |
| 128×1024 | 38.392 | 36.449 | 36.271 | 94.897 | 0.382× |
| 64×4096 | 74.929 | 70.771 | 70.599 | 109.877 | 0.643× |

All eight trials and all four fresh winners pass the FP64 oracle. Maximum
native absolute error is `1.734e-5`, within the recorded LayerNorm tolerance.
Times are p50 warm synchronized host-wall throughput with eight CPU threads.
Native output is preallocated; PyTorch's functional result allocation remains
inside timing.

This target split is intentional evidence, not a contradiction: logical SSA
sharing belongs in the structural IR, while materialize-versus-recompute and
the physical stack/workspace choice belong to target planning. Metal benefits
from compact worker stripes and fewer device loads; LLVM benefits from fusing
the cheap expression into vectorized consumers. A universal frontend heuristic
cannot represent both choices.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search \
  --backends cpu --operations residual_layernorm \
  --cpu-input-views --cpu-model native --auto-vectorize \
  --cpu-stack-bytes 65536 \
  --tune-shared-tile-materializations preserve,expensive-only \
  --samples 9 --sample-ms 60 --warmup-ms 100 --capture-sources
```

Artifact identity:

- `results.json`: `e08de35001566bc9acbbb987fcac448141509382a78d5b1dfb67d2b4c0d0e4f7`
- benchmark executable: `f420ebf571dbb348e813ad5cc040dbe33cd068f16f0adce32801b3d85684a0c7`
- loaded TIRx bridge: `2bab5db74c520268c568643562c8e433b9ad47b1b200fb69ee81ad41463d315c`

This is one CPU/operator cohort, not proof that the portable CPU lowering is
uniformly faster than PyTorch. In particular, the 64 KiB stack policy is part
of the measured candidate and must not be omitted when reproducing the table.
