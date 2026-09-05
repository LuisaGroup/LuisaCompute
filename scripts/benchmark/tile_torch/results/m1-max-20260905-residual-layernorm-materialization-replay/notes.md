# Residual LayerNorm materialization A/B replay

Date: 2026-09-05 Asia/Shanghai. Machine: Apple M1 Max, macOS 26.6.2 arm64.
PyTorch 2.14.0.

This four-round replay compares `expensive-only` (reference) with `preserve`
(candidate) using the same benchmark executable and loaded compiler library.
Variant order, shape order and native/PyTorch order are balanced. Every row is
freshly captured/JIT-compiled and every output is checked; all 32 native
variant measurements pass and the fingerprinted artifacts remain unchanged.
The frozen inputs are the
[`expensive-only` source report](../m1-max-20260905-residual-layernorm-expensive-only/notes.md)
and [`preserve` source report](../m1-max-20260905-residual-layernorm-preserve/notes.md).

| Rows×width | Reference µs | Preserve µs | Paired speedup median [range] |
|---|---:|---:|---:|
| 1×127 | 3.692 | 3.506 | 1.057× [1.027, 1.137] |
| 17×257 | 3.648 | 3.632 | 1.008× [0.957, 1.039] |
| 128×1024 | 8.244 | 6.084 | 1.354× [1.313, 1.366] |
| 64×4096 | 13.548 | 9.591 | 1.421× [1.392, 1.471] |

Medians are over within-round p50 synchronized host-wall measurements; ranges
are descriptive min/max paired ratios, not confidence intervals. The compared
binary hash is
`513af14b6c40163bd93aa3c3eb2189784ba14082c86fa29c8e8c7ab61587a266`.
The exact frozen policies, plans, orders, raw samples, generated-source hashes
and artifact fingerprints are in `results.json`.

Artifact identity:

- `results.json`: `ddeea6fc44db5f3ff671779d4b64de1c69b2a0b72bf98dc1071ad635b16ac04a`
- benchmark executable: `513af14b6c40163bd93aa3c3eb2189784ba14082c86fa29c8e8c7ab61587a266`

This replay predates only the later addition of a JSON field reporting the
64-scalar stripe budget; the compute path already contained and enforced that
budget. The current search reports above use the final reporting surface.
