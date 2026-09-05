# Reduction collaboration, packing and ordered unrolling: JIT search

Date: 2026-09-05, Asia/Shanghai. Apple M1 Max, FP32, TIRx Metal.

This search deliberately includes the old automatic planner. Searching only
exact 32/64/128/256-thread cooperating widths excludes its useful packed
short-row programs and can make a purported optimization slower.

The finite product is two thread requests (`auto`, 128), two packing requests
(`auto`, 4), and two ordered stripe unroll factors (1, 4), for eight trials per
case. The sum and softmax shapes separate row count from row width:
1×4096, 64×4096, 1024×4096, 17×257, 1024×257. Explicit packing four means
four independent one-subgroup programs, not four subgroups collaborating on
one program. An explicit simultaneous width must be 128.

Each candidate is recaptured, JIT-compiled and fully checked against FP64.
Resource-invalid candidates remain visible as rejected trials; in particular
a 4096-wide preserved softmax Tile exceeds the 64-scalar worker-stripe budget
at one subgroup per row. Every final selected row passes a fresh validation.
Search minima are not performance claims. The independent
[frozen-plan replay](../m1-max-20260905-reduction-joint-map-replay/notes.md)
is the evidence for improvements and regressions.

```bash
uv run --no-project --python 3.13 --with torch --with numpy \
  python scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY --backends metal --operations sum,softmax \
  --metal-subgroup-reductions --tune-group-threads 0,128 \
  --tune-reduction-packing 0,4 --tune-reduction-unroll 1,4 \
  --row-shapes 1x4096,64x4096,1024x4096,17x257,1024x257 \
  --samples 5 --sample-ms 30 --warmup-ms 100 --capture-sources
```

The candidate list contains overlapping realizations: e.g. auto threads plus
exact packing four and exact 128 threads plus packing four. They remain in
the raw audit rather than being removed after measurement; the search is
not presented as a statistically unbiased estimate. Its winner is validated
again and then tested in separately balanced rounds.

The analytic default still uses the old 1/2/16 scalar/collective/setup prior.
It does not model unrolling, duplicated reads or full-device row occupancy.
The new backend-owned `ExecutionCostPolicy` can override coefficients and the
complete row-candidate score, but these same-device search trials are not
silently relabeled as a calibrated cross-device model.
