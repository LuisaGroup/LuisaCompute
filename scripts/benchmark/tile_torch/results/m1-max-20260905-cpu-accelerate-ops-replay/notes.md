# Balanced CPU array-math policy replay

This is the authoritative A/B for `CpuMathBackend::reference` versus
`CpuMathBackend::accelerate`. Both variants use the same executable, TileIR,
shape, CPU target, input-view policy, stack budget and SIMD-pack budget. The
plans are frozen from the two initial reports; no parameter search or best-run
selection occurs during replay.

## Protocol

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-reference-ops-v2/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-v2/results.json \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay \
  --operations add,sum,softmax --rounds 6 --samples 7 \
  --sample-ms 40 --warmup-ms 200 --threads 8 --capture-sources
```

Each of 12 cases runs both policies in every round. Shape order rotates;
reference/candidate and native/Torch precedence are balanced. Every run is
freshly captured/JIT-compiled and checks its complete output. There are 144
valid rows, zero failures, and the complete fingerprint set is unchanged.
The source plan hashes are
`17c138a67c1376a948f0d3e5d4e12744109d7078ac211d92624433e0b740a5c7`
and `2c80b288c2dc5a8b18a28cf8dfacf2ada02c206770fe00f24c62f615233bd900`.

## Findings

| Family | Shapes | Paired reference / Accelerate speedup |
|---|---|---|
| add control | 1×127, 17×257, 128×1024, 4096×256 | 1.001--1.022× medians; ranges cross 1 |
| row sum | 1×127, 17×257, 128×1024, 64×4096 | 2.708--6.123× |
| softmax | same four widths/row cohorts | 2.098--5.460× |

The independent add control shows that merely selecting the policy does not
rewrite unrelated code. Candidate medians are below eager Torch for all 12
cases; that comparison includes both runtimes' warm dispatch and is not a
pure arithmetic timer. The detailed per-shape medians and paired min/max ranges
are in [results.md](results.md); all raw timing, output, ordering, source and
artifact evidence is in [results.json](results.json).

This closes the measured FP32 add/sum/softmax cohort for the proved provider
patterns. It does not establish general XIR/reference-loop performance, other
dtypes, fused production LLM kernels, or automatic provider break-even
selection.
