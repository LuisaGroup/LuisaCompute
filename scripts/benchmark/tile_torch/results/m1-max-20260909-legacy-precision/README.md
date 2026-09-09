# Legacy/current Tile and precision evidence, 2026-09-09

Canonical narrative: `docs/source/performance/tile/migration.md`, integrated
into the existing Sphinx performance section. These are diagnostic desktop
measurements, **not** an MPS/Torch acceptance result.

The source baseline is merge `2d02721b89980cd9191976c64b05d00b2342a03d` plus
explicit, hashed task-only overlays. Unrelated user edits and dirty dependency
checkouts are not part of the isolated build. The original legacy frontend is
`ccdfcbebef7fa95431c988e1fcbdd87ffdce9fdc`; its ordinary SIMT AST is replayed
through the same current Metal/SIMD backend and Runtime as the new captures.

`manifest.json` identifies each compressed bundle and every original member
by SHA-256, size, and source path. Extract bundles to **separate new directories**
because attempts can have similarly named files. The archives retain full
inputs, FP64 reference values, complete outputs, generated source, and logs;
passing executable receipts additionally check both output guards, twice.
Repository scripts build/export/replay the experiments without switching the
checkout or installing the old Tile frontend into the current library.

| Bundle | Evidence |
|---|---|
| `initial-fp32-pilot` | First 10-op Metal comparison, including stack and pipeline-creation failures |
| `initial-migration-source` | Exact source overlay for that first pilot |
| `precision-v9-pilot-with-harness-failures` | First precision/scan comparison, including all eight rejected CPU harness visits |
| `precision-cpu-replay` | Clean CPU E2E replay after removing the Metal timestamp setting |
| `precision-final-replay` | Two-order replay matching the frozen final low-precision implementation |
| `precision-final-source` | Exact implementation/test/benchmark overlay for that replay |
| `precision-matrix-v2-source-and-final-validation` | Exact 17-file matrix v2 overlay, 80 XIR/SIMD + 41 Tile + Metal-codegen final regression receipts, and the v2 six-test targeted rerun |
| `precision-final-boundary-source-and-validation` | Later 17-file overlay with explicit TIRx BF16 MMA-accumulator rejection, its negative test, and six passing precision/migration tests; no matrix rerun claimed |
| `legacy-sized-equivalence` | Original and parameterized exporters produce byte-identical AST and launch at all 16 original cases |
| `legacy-build-provenance` | Pinned source/submodules, build commands, exporter hashes and dimension-only derivation |
| `build-test-and-failed-attempts` | Initial merge verification, precision iterations and retained failures; later final regression receipts are supplementary |

The `build-test-and-failed-attempts` bundle is an immutable **intermediate**
snapshot: its full-regression receipt was captured while that run was still
in progress. Its archive/member hashes pass, but it must not be cited as the
completed regression. Use the completed receipt, all four final logs and three
JUnit files under `validation/` in
`precision-matrix-v2-source-and-final-validation.tar.gz`. The 80/41/codegen
regression used v1; v2 changed only the benchmark's explicitly labelled full-K
capture and reran the six precision/migration tests. Both source hashes remain
in their own receipts; they are not presented as one identical source snapshot.
The later `precision-final-boundary` snapshot changes exactly
`src/tile/bridge/tirx/lower.cpp` and `src/tests/unit/tile/test_tile_types.cpp`
relative to matrix v2. It closes a silent-widening boundary by rejecting BF16
MMA accumulation. The matrix cases use FP32 accumulation, but their recorded
timings still belong to v2; the later boundary validation is not substituted
for measured-source provenance.

## Expanded matrix

`matrix-results.json` retains the complete 38-case / 191-route-combination
experiment: 382 planned visits, 282 passing visits, 50 failed attempts and 50
second visits not attempted after their corresponding failure. Old-version
failures remain **Error**; successful current-version data is retained beside
them. No failed baseline receives a speedup ratio. This is the v2 source
snapshot, not a claim about any later source edits.

`matrix-manifest.json` describes every one of the **1,861 original files** in
the matrix directory. It records original byte lengths and SHA-256 hashes,
plus ordered lists of content-addressed chunks. `matrix-evidence-00.zip`
contains the **589 unique chunks**; each ZIP entry's name is its uncompressed
SHA-256. Chunk deduplication and compression preserve the complete 9.62 GB
logical dataset without sampling or dropping failed visits. The manifest also
hashes the compressed shard itself. `matrix-results.json` is checked against
the archived `results.json`, so the convenient summary cannot silently drift.

## Verify and restore

From the repository root, verification needs only Python's standard library
and does not rerun kernels or materialize the large raw dataset:

```sh
python3 scripts/benchmark/tile_torch/verify_legacy_evidence.py \
  scripts/benchmark/tile_torch/results/m1-max-20260909-legacy-precision
```

To restore, add `--restore /absolute/path/to/new-evidence-directory`; its parent
must exist and the destination itself must **not** exist. Allow at least
12 GB of free space. Each tar bundle is restored into its own named directory;
the expanded matrix is restored into `matrix/`. Files are created exclusively,
not overwritten. Absolute/traversal paths, duplicate manifest/ZIP entries and
non-regular tar members are rejected. A failed restore may leave a partial new
directory for inspection; it never deletes an existing directory.

The verifier checks both compressed archives and every uncompressed member,
then reconstructs and hashes each complete matrix file. Passing integrity
checks establish reproducible evidence, **not** benchmark correctness by
themselves; oracle, guard, timing and source-snapshot receipts establish the
separate experiment contracts described in the canonical report.

Audit result (2026-09-09): **11 bundles / 581 tar members**, all 589 unique
matrix chunks and all 1,861 reconstructed matrix files pass size/SHA-256
verification. The existing intermediate bundles were not regenerated. Small
verifier fixtures also pass exact-byte restoration, existing-directory
refusal, eight unsafe-path cases, corrupt-content rejection, duplicate-key
rejection and non-regular tar-member rejection. The full 11.94 GB raw dataset
was streamed for verification; only the small fixture was restored to disk.

Documentation QA: Doxygen XML generation completed (with existing documentation
warnings); strict Sphinx HTML generation passed. The final local-link check
covered 67 pages, 5,106 links/assets and 199 compatibility anchors. The two
post-matrix C++ files pass the repository clangd checker and clang-format.
`docs-qa-final/receipt.json` and screenshots verify the full 38-row matrix and
scan/exchange design at 1280px and 390px, including access to the horizontally
scrollable rightmost matrix column. `docs-qa/` retains the initial visual visit.
Run `qa-docs.cjs` with the built HTML root, a new output directory, and the
installed Playwright module path to repeat the rendering checks.

Timing definitions: `throughput_us` and `latency_us` include host/Runtime
dispatch. Metal's separate `device_timing.throughput[*].compute_ns` must be
divided by **its own** `device_timing.repetitions` and by 1000 to obtain GPU
microseconds per dispatch. Uninstrumented command-buffer controls are retained.
CPU values are not native-entry-only timings. Shapes, dtypes, scheduling
variants and unsupported combinations must remain separate when comparing.
