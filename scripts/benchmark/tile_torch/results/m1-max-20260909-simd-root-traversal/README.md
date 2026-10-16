# SIMD GEMM root-traversal diagnostics, 2026-09-09

Canonical narrative: `docs/source/performance/tile/migration.md`. This archive
retains seven completed GEMM diagnostic attempts, full input/oracle/output
buffers, generated source, implementation snapshots and build/test receipts.
The historical matrix's 4096³ SIMD **Error is retained unchanged**. These are
separate later attempts, not replacements for failed matrix cells.

## What the measurements establish

All seven attempts report **one throughput sample and one latency sample** of
synchronized host wall time through Luisa Runtime. They are E2E diagnostics,
not native-entry timings, GPU timestamps, pure-kernel timings or randomized
ABBA comparisons. Compilation, cold call, warmup and total process elapsed
time are recorded separately in each log. In particular, a process timeout
cannot be reported as the duration of one kernel call.

The completed original-shape 4096³ attempt demonstrates that this case can
finish and pass correctness when allowed to complete; it does not establish
that the original timeout budget, whole-process benchmark protocol or broader
performance target has been fixed. The root-traversal option is opt-in, and
its temporal cache cost is explicitly **unmodeled**. It is not an automatically
calibrated planner choice.

| Historical diagnostic label | M × N × K | Throughput E2E sample | Latency E2E sample |
|---|---|---:|---:|
| `baseline` | 4096³ | 13.111 s | 12.816 s |
| `baseline-full` | 4096³ | 10.171 s | 10.594 s |
| `m8n1` | 4096³ | 4.656 s | 4.471 s |
| `m8n1-full` | 4096³ | 2.570 s | 2.498 s |
| `m8n1-w4-full` | 4096³ | 3.478 s | 3.351 s |
| `root32-rect` | 1024 × 2048 × 256 | 0.017782 s | 0.017260 s |
| `root32-4096` | 4096³ | 8.753 s | 9.257 s |

These are historical experiment labels, not automatically comparable planner
candidates. Some attempts change source tile dimensions, execution settings or
SIMD width. All seven use `source_schedule=k_tile_pipeline`. The `-full` suffix
means `LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION=1`, **not full-K**. Only
`m8n1-w4-full` uses W4; the other six use W8, and all use eight CPU workers.
The two `root32-*` attempts additionally set
`LUISA_SIMD_ROOT_AXIS_TILES=32,32`. Their source microblock dimensions remain
the `block` field in each log. All attempts request samples=1, sample-ms=10
and warmup-ms=1; a long cold call or minimum warmup batch can greatly exceed
that nominal warmup target. The handoff preserves the exact settings.
No result in this table establishes a statistically qualified speedup, legacy
Tile parity or superiority to Torch, MPS or BLAS.

Every completed log reports two full-output checks and 34 guard elements per
check, with maximum absolute error zero for these fixtures. Each square case
preserves two 64 MiB FP32 inputs, a 128 MiB FP64 oracle and a 64 MiB FP32 output.
The rectangular case likewise retains complete buffers. No tensor is sampled,
omitted or lossily compressed. `baseline.sample.txt` is an additional retained
diagnostic, not a substitute for full tensor data.

`run/root-v1/source` retains the measured root-traversal source snapshot;
`run/final-source` preserves the later implementation/test snapshot and handoff
receipts. These are separate source states, not a claim that all seven timings
were collected from the final source. Archive-time SHA-256 hashes establish
stored-byte integrity, not retroactive pre-run binary freeze evidence.

## Archive and verification

The standard-library-only `archive.py` creates one tar/gzip bundle per complete
case, plus a source/receipts bundle. It accepts large individual files without
truncation and checks that **each compressed bundle is below 50,000,000 bytes**.
It freezes every original file by size/SHA-256, checks the source inventory
again after packing and verifies every archived member. It never runs builds,
benchmarks, JIT or archived executables and never deletes original data.

Creation requires the explicit narrow experiment directory, not a checkout:

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260909-simd-root-traversal/archive.py \
  --create --run /absolute/path/to/frozen-gemm-diagnostics \
  --base-commit b6b10e6039d2edc1eb85672055af315ec455c5df
```

Existing evidence is never overwritten. If creation fails, its partial files
remain for inspection; retry only into a different new output directory.
Source symlinks, special files and unreviewed directories are rejected. The
manifest records every complete original file and its historical location.

Verification needs only this committed directory and Python's standard library;
it neither reads the historical `/tmp` sources nor extracts or executes files:

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260909-simd-root-traversal/archive.py --verify
```

It checks compressed SHA-256, every streamed member's SHA-256 and byte length,
canonical relative paths, duplicate JSON keys and archive members, and
regular-file-only entries. Extract verified bundles into a **new** directory
if the raw corpus is needed; every member is under the `run/` namespace.
Allow at least 2.1 GB of free space for the full original data. Historical
commands may need deliberate path relocation before any fresh experiment.

Archive verification proves complete storage, not benchmark correctness or
performance. The recorded numerical, timing, source and test receipts support
those separate claims, with the single-sample and source-version limitations
above. No further tuning is part of this archival handoff.

## Archive audit

Final audit (2026-09-09): **79 original files / 2,043,205,157 logical bytes**
are retained in **8 bundles / 29,497,461 compressed bytes**. The largest
bundle is 4,692,738 bytes, below the 50 MB limit. The manifest is 27,404 bytes.
Source-inventory and complete original-file freeze checks passed, followed by
two full archive/member verification passes.

`run/evidence-qa/archive-fixture.py` and its log preserve the synthetic tests,
including a complete 128 MiB member, all seven required case groups, exact
original-byte verification after moving the source directory, unsafe paths,
duplicate JSON keys/members/bundles, corrupt contents, wrong sizes/hashes,
existing-output refusal and symlink rejection. All passed. The archived
documentation checker and `run/docs-qa` receipt/screenshots cover the new
GEMM report section at 1280px and 390px, alongside the successful strict
Sphinx build log. The configured-build, three passing CTests and five C++
syntax-check receipts remain intact in the source/receipts bundle.
