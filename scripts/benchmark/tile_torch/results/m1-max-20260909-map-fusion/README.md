# Generic XIR map-fusion evidence, 2026-09-09

Canonical narrative: `docs/source/performance/tile/migration.md`, in the
existing Sphinx performance section. This archive supports a **same-current-
compiler, map-fusion off/on** experiment on M1 Max. It is not a new old/new Tile
comparison, and does not establish parity with Torch, MPS or BLAS.

## Timing and source boundaries

The measured v2 cohort has eight cases, two off/on variants, two ABBA cycles
per case, eight visits per case and seven timed samples per visit. The common
C++ replay helper invokes captured native ORC object code on one CPU thread.
The timed region excludes Python, JIT, allocations by the harness, input
initialization, output checking and Luisa Runtime dispatch. It includes the
generated entry, its block traversal, launch-state reset, and calls performed
by that entry. This is native-entry host wall time, not CPU cycles or
multithread throughput. The desktop was not CPU-affinity pinned.

Each prepared variant preserves its actual object, linked dylib, generated
LLVM source, helper/ABI-header source, full inputs, independent FP64 oracle and
captured output. Each replay visit retains every output element, raw samples,
repetition count, oracle/guard checks and artifact hashes. No tensors are
sampled or reconstructed approximately. The diagnostic capture executable's
Runtime timings are retained but are **not** the main pure-entry measurements.

| Snapshot | What it can establish |
|---|---|
| `run/captures`, `run/prepared`, `run/replay` | v2 measured objects and complete reproducible timing/correctness evidence |
| `run/v3/source`, `run/v3/capture-manifest.json` | v3 source snapshot and per-case bytewise LLVM/object comparisons with v2; this is not a second pure-entry timed cohort |
| `final-source`, `run/v4` | Later budget-fix implementation, tests and bytewise-equivalent v2 recapture; not a new timed cohort |
| Root logs and `run/v3-preflight-error` | Retained build, test and preflight failures, not silently discarded attempts |

The baseline is commit `97677a1cec80306989dffee516b3d1a25fb3b0f4` plus explicit
task-only source overlays. The v2 compiler overlay was hashed before capture
but was not copied before later edits; its generated LLVM and actual ORC
objects remain available. The v3 artifact-equivalence checks apply only to
the 16 captured variants and checked artifact kinds, not every possible
kernel or compiler transformation.

The final v4 recapture also completed all 16 variants: **16/16 generated LLVM
files, 16/16 actual ORC objects and 50/50 input/oracle/output files are bytewise
identical to v2**. Its 11 source files and 24 binary paths are unchanged across
capture. This connects the measured objects to the final implementation for
these cases without relabelling the v2 timing samples as a new v4 benchmark.
The exact comparisons and both sources' hashes remain in
`run/v4/capture-manifest.json`; v4 diagnostic Runtime timings stay separate.

The original v2 before/after binary inventory omitted `.so` plugins. The
later v3 inventory includes 24 binary paths (executable, libraries and plugins). Those later
hashes **cannot retroactively prove** the omitted v2 plugins' prior state.
Keep this limitation alongside the positive frozen object/helper evidence.

The main report explains the observed aligned-scan improvement and its
boundary: the change is generic pure-map/index-expression fusion, with
matching relative-work accounting in the planner. It does not add a scan
opcode, change the arithmetic scan graph, introduce Metal warp exchange, or
establish a calibrated cost model. The ragged scan remains substantially less
improved; identical-code controls do not receive a performance-win claim.

## Create and independently verify

The standard-library-only `archive.py` takes a narrow frozen experiment
directory and final-source overlay. It never runs benchmarks, builds, JIT or
archived executables. Creation retains full original files in deterministic
tar/gzip bundles, each below 50,000,000 bytes. It records SHA-256 and size for
every bundle and uncompressed member, rechecks the input inventory and file
bytes after packing, and immediately verifies the completed archive.

From the repository root, after the final validation snapshot is frozen:

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260909-map-fusion/archive.py \
  --create --run /absolute/path/to/frozen-experiment \
  --final-source /absolute/path/to/frozen-v4-source \
  --base-commit 97677a1cec80306989dffee516b3d1a25fb3b0f4
```

The output directory may initially contain only `archive.py`, `README.md` and
the documentation rendering checker `qa-docs.cjs`;
existing evidence is never overwritten. A failed creation leaves its partial
new files available for inspection. Use a different new output directory for
a retry. Source symlinks, special files and unreviewed experiment directories
are rejected. The source-overlay guard rejects a whole Git checkout or a
large/broad file collection. No unrelated working-tree files are searched.

The one explicit directory exclusion is `run/docs-html`: a rebuildable
whole-site Sphinx tree, not benchmark evidence. It is recorded with its reason
in the manifest and is neither deleted nor read recursively by the archiver.
The documentation build logs and targeted desktop/mobile rendering screenshots
and receipts are retained under `run/evidence-qa/docs`; `qa-docs.cjs` preserves
the rendering checker. All capture/prepared/replay tensor data remains included.

Anyone with only this committed directory can verify every original byte,
without the machine's historical `/tmp` paths:

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260909-map-fusion/archive.py --verify
```

`manifest.json` contains a complete member inventory and historical source
paths for provenance only. Verification does not open those external paths.
It validates archive hashes, canonical relative paths, JSON-key/member
uniqueness, regular-file-only tar entries, modes, lengths and the SHA-256 of
every streamed original member. It does not extract files or execute code.
Extract verified bundles into a **new** directory when the raw corpus is
needed; all member names use distinct `run/` or `final-source/` namespaces.
Allow roughly 1.5 GB for the uncompressed corpus. Historical commands and
prepared manifests may contain absolute paths and need deliberate relocation
before a fresh replay; an integrity check is not a replay claim.

Integrity, numerical correctness, timing validity and source equivalence are
different checks. Passing `--verify` establishes only complete, unmodified
evidence storage; the recorded oracle, guard, ABI, timing and source receipts
support the separate experimental claims in the canonical report.

## Archive audit

Final audit (2026-09-09): **982 original files / 1,155,069,201 logical bytes**
are retained in **31 bundles / 19,400,020 compressed bytes**. The largest
bundle is 2,305,989 bytes, below the 50 MB limit. Creation's complete
source-freeze check and two full archive/member verification passes succeeded.
The manifest is 336,975 bytes; these sizes exclude the small explanatory and
verification tools beside it.

`run/evidence-qa/archive-fixture.py`, its initial log and final rerun log retain
the synthetic verifier checks: exact original-byte recovery, verification
after moving the original inputs, unsafe-path rejection, duplicate keys and
members, corrupt contents, size/hash mismatches, existing-output refusal and
non-regular/symlink rejection. The final rerun passed after the reviewed
whole-site-output exclusion was introduced. The documentation evidence
contains the strict Sphinx build log, desktop 1280px/mobile 390px rendering
receipt and ten screenshots, including horizontally scrolled table columns.
