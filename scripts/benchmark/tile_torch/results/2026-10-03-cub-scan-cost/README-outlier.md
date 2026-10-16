# Runtime instability: separate fixed/automatic repetition

The original formal BF16 slow observation remains unchanged in `automatic.json`. A new four-phase diagnostic ran fixed T128, automatic, fixed T128, automatic on the two 256 × 16384 narrow-storage cases. All outputs/guards/read-only and graph-v2 receipts passed full upstream validation.

| Type | Fixed 1 µs | Auto 1 µs | Fixed 2 µs | Auto 2 µs | Formal auto µs (separate) |
|---|---:|---:|---:|---:|---:|
| fp16 | 12.856192 | 33.525078 | 12.548796 | 12.306276 | 12.529841 |
| bf16 | 13.320557 | 12.800519 | 12.637586 | 12.866688 | 33.535660 |

The slow state moved from BF16 in the formal queue to FP16 in the first new automatic run. All seven samples of each run are retained. Source SHA, compile key, installed resources, selected T128, launch geometry, fixtures and final-pointer guards matched. The cause is **unresolved**; these facts do not establish stable automatic/fixed runtime parity. The formal engineering gates compare automatic with original Tile, so passing them does not resolve this separate instability.

No Torch ran in this diagnostic. No denominator is borrowed from another round; no measurements are pooled, replaced, filtered or refitted. Event graph samples and ordinary stream/event/host spans remain distinct in the compact data. These are host-selector receipts, not a captured Driver execution trace.

Run `python reproduce_outlier.py` after `python reproduce_automatic.py`. It verifies public bytes, all 56 new graph samples/medians/denominators, frozen automatic scores, selected source/resource/guard identity, both fixed-reference ratios and preservation of the formal observation. It does not rerun tensors or GPU work.

The original local summary hash identifies unredacted source bytes; `manifest-outlier.json` identifies the new compact public bytes. Existing explicit experiments and automatic formal data are unchanged.
