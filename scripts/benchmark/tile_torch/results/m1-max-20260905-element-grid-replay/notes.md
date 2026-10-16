# Automatic GPU program/element grid: frozen-binary A/B

Date: 2026-09-05, Asia/Shanghai. Apple M1 Max; macOS 26.6.2 arm64;
PyTorch 2.14.0; FP32. This is the TIRx Metal route, not native MPP or XIR.

## Result

The new mapping removes the program-per-thread serialization of an independent
Tile element nest. Four balanced rounds validate all 32 complete native
outputs. The executable and adjacent-library fingerprints are unchanged
throughout the replay. Each shape appears once per variant per round.

| Add shape | Old worker map, µs | Fused grid, µs | Median paired speedup | Paired new/Torch time |
|---|---:|---:|---:|---:|
| 1×127 | 88.431 | 2.552 | 34.488× | 0.704 |
| 17×257 | 215.877 | 2.735 | 79.173× | 0.705 |
| 128×1024 | 44.836 | 5.201 | 8.504× | 0.826 |
| 4096×256 | 78.358 | 18.622 | 4.202× | 0.832 |

Times are medians of per-round medians; speedups are medians of paired ratios,
not ratios of those table medians. Both implementations include host dispatch,
exclude transfers/JIT, and use preallocated add outputs. These are not GPU
hardware-event timings or a claim about every elementwise kernel. The old
ragged-tile path combines serial element loops with large private snapshots;
the new family commits proved input forwarding together with grid fusion.
The A/B measures that combined realization, not fusion isolated from forwarding.

The reference was copied before this implementation, including its adjacent
dylibs, to `/tmp/luisa-tile-mapping-baseline.WxLF3h`. Absolute paths and all
hashes are recorded in [raw results](results.json). Generated Metal for both
variants is archived under `sources/`. The first pilot supplied configurations,
not timing values used by this replay.

```bash
uv run --no-project --python 3.13 --with torch --with numpy \
  python scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-element-grid-baseline/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-element-grid-pilot/results.json \
  --native /tmp/luisa-tile-mapping-baseline.WxLF3h/benchmark_tile_tirx \
  --candidate-native cmake-build-tirx/bin/benchmark_tile_tirx \
  --operations add --rounds 4 --samples 7 --sample-ms 40 --warmup-ms 100 \
  --capture-sources \
  --output NEW_EMPTY_DIRECTORY
```

On the new binary, `run.py --element-grid reference` versus
`--element-grid auto` also exposes the old/new family for same-binary tests.
Explicit execution bindings, read/write snapshots, unproved custom output
layouts and multiple effect domains retain their existing realization.

Independent validation recomputed every median and paired ratio from raw
rows, checked uniqueness of `(case, round, variant)`, complete validity, and
the replay's before/after fingerprints. Assessment: share with the timing,
cohort, hardware and combined-transformation caveats above.
