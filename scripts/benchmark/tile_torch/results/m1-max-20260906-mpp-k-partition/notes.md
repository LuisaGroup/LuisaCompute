# MPP K partition: a shape-sensitive tuning dimension

## Technical summary

On September 6, 2026, a fixed-geometry TIRx→MPP view experiment finds K=4096
fastest in both search orders for 1024×1024×1537 and 4096³, but K=1024
fastest for 4096×4096×11008. The latter advantage over K=4096 is only about
1.1–1.4%, so two observations are not enough to establish a robust preference.
No global K default or cost-model coefficient changes follow.

The independent audit verifies **90 complete outputs and 1,038,090,240 elements**,
all max-absolute-error zero for the existing dyadic inputs. Every candidate
retains the same four independent 32×32 SIMD-group outputs, 128 threads, zero
shared allocation, persistent accumulator and direct output store. This is
a staged/JIT parameter diagnostic, **not a new lowering speedup or library
parity result**.

## Scope and timing definitions

Apple M1 Max, macOS 26.6.2 arm64, Torch 2.14.0; FP32 compact row-major
C=A×B, alpha=1, beta=0, no transposes/prepacking. Fixed output block 128×32,
ordered pipeline, copy batch 1 and retained subgroup fences. Candidate K
blocks: 128/512/1024/4096. TVM retains its existing fast-math setting; direct
MPS and eager Torch remain independent controls.

Each cell below is **forward-order / reverse-order** median of five samples,
in microseconds. These labels name two driver configurations: run.py also
rotates candidate and framework order within each shape. They are **not**
a perfectly balanced two-round permutation. The exact commands and order
are preserved in the raw reports.

GPU batch is the no-counter completed-command-buffer interval per invocation,
including GPU work and gaps, not isolated shader time. E2E batch is host wall
time with dispatch/synchronization amortized. GPU single and E2E single are
separate phases and must not be subtracted from independent batch medians.
Instrumented compute-pass samples remain in the raw files, diagnostic only.

## Complete K sensitivity, with fresh MPS/Torch controls

Larger K reduces the emitted outer MPP call count. For a single K iteration,
the existing lowering may additionally use overwrite mode without an initial
accumulator fill. The comparison therefore includes those valid lowering
consequences, not just a measured per-call cost. No cache-hit, occupancy or
instruction-count conclusion is inferred.

| M×N×K | Captured BK | TIRx GPU batch µs | MPS GPU batch µs | Torch GPU batch µs |
|---|---:|---:|---:|---:|
| 1024×1024×1537 | 128 | 578.587 / 577.574 | 435.152 / 430.761 | 440.926 / 435.521 |
| 1024×1024×1537 | 512 | 520.537 / 518.454 | 437.072 / 432.761 | 435.810 / 437.414 |
| 1024×1024×1537 | 1024 | 509.463 / 509.648 | 434.011 / 433.447 | 435.526 / 433.969 |
| 1024×1024×1537 | 4096 | 480.859 / 482.236 | 440.793 / 432.741 | 438.438 / 436.307 |
| 4096×4096×4096 | 128 | 22909.833 / 20507.875 | 17889.792 / 17468.875 | 16685.875 / 15796.125 |
| 4096×4096×4096 | 512 | 20572.708 / 20429.250 | 17822.750 / 16863.750 | 16756.083 / 16832.250 |
| 4096×4096×4096 | 1024 | 19631.125 / 20369.542 | 18010.875 / 18002.250 | 16641.333 / 16895.000 |
| 4096×4096×4096 | 4096 | 18230.583 / 18226.792 | 18136.125 / 16752.250 | 16526.042 / 15802.500 |
| 4096×4096×11008 | 128 | 65308.458 / 64176.000 | 51185.250 / 51488.625 | 48138.750 / 48223.250 |
| 4096×4096×11008 | 512 | 56209.000 / 56650.750 | 50278.125 / 51786.292 | 47885.750 / 47558.375 |
| 4096×4096×11008 | 1024 | 55701.750 / 55364.333 | 51193.417 / 51056.542 | 47248.250 / 48149.917 |
| 4096×4096×11008 | 4096 | 56464.208 / 56001.000 | 50615.750 / 50723.958 | 48050.917 / 47951.292 |

The 4096³ K=4096 candidate is 0.929× / 0.895× the K=1024 GPU time.
For 1024×1024×1537 it is 0.944× / 0.946×. The long-K shape reverses that
direction: K=4096 is 1.014× / 1.011× K=1024. Keep this shape interaction in
the next candidate family instead of fitting a universal monotonic K reward.

## E2E and single-call metrics remain separate

| M×N×K | Captured BK | TIRx E2E batch µs | TIRx GPU single µs | TIRx E2E single µs |
|---|---:|---:|---:|---:|
| 1024×1024×1537 | 128 | 591.243 / 582.935 | 574.083 / 579.917 | 866.041 / 798.166 |
| 1024×1024×1537 | 512 | 535.816 / 533.974 | 514.208 / 516.792 | 785.291 / 775.791 |
| 1024×1024×1537 | 1024 | 524.871 / 518.110 | 518.958 / 516.833 | 786.292 / 755.208 |
| 1024×1024×1537 | 4096 | 493.295 / 491.724 | 468.958 / 470.167 | 800.875 / 699.750 |
| 4096×4096×4096 | 128 | 23544.916 / 23857.625 | 23273.333 / 20556.458 | 23461.209 / 22563.375 |
| 4096×4096×4096 | 512 | 20531.875 / 20750.042 | 20434.708 / 19547.542 | 20857.458 / 20783.500 |
| 4096×4096×4096 | 1024 | 20110.084 / 20635.916 | 19746.917 / 19956.125 | 19869.042 / 20308.708 |
| 4096×4096×4096 | 4096 | 18214.291 / 18445.417 | 18236.417 / 18109.042 | 17994.208 / 18188.042 |
| 4096×4096×11008 | 128 | 66206.208 / 65182.000 | 66694.875 / 63941.458 | 65544.083 / 65311.583 |
| 4096×4096×11008 | 512 | 56506.708 / 56583.542 | 56602.208 / 56192.375 | 56695.250 / 56723.750 |
| 4096×4096×11008 | 1024 | 56017.709 / 56071.958 | 55761.375 / 55811.833 | 55860.958 / 55765.750 |
| 4096×4096×11008 | 4096 | 56846.125 / 56810.333 | 56041.958 / 56386.083 | 56678.667 / 57209.417 |

Single-call improvements are not universal; for example, the first
1024×1024×1537 K=4096 E2E single call is slower than K=1024 despite better
batch throughput. Exact MPS/Torch E2E and single-call values are retained in
the independent audit, not replaced with their GPU batch timings.

## Fresh recapture does not establish library parity

The harness reruns the numerically valid search minimum with a fresh
capture/JIT and measurement. Those are not reused search minima:

| M×N×K | Fresh BK | GPU TIRx/MPS | GPU TIRx/Torch |
|---|---:|---:|---:|
| 1024×1024×1537 | 4096 | 1.108 / 1.113 | 1.097 / 1.101 |
| 4096×4096×4096 | 4096 | 1.009 / 1.052 | 1.086 / 1.122 |
| 4096×4096×11008 | 1024 | 1.104 / 1.094 | 1.174 / 1.166 |

All six fresh GPU comparisons remain slower than Torch and MPS. The search
is exploratory; no repeated held-out acceptance or new planner default is
claimed.

## Validation, uncertainty and reproducibility

- [Protocol](protocol.md) and [driver](experiment.py) were written before timing.
- [Forward raw report](forward/results.json) and [reverse raw report](reverse/results.json)
  retain every trial, its controls, and the fresh post-selection result.
- [Independent audit](audit.py) imports no benchmark statistics or validators.
  It checks all numerical receipts, sample lengths/finiteness, no-counter
  scope, source hashes and exact same subgroup plans; five corrupted-report
  probes are rejected. It does not independently rerun the saved numerical
  outputs, which the benchmark validates in full before deleting its temporary
  output file.
- The five TVM/FFI library hashes match before/after in [execution.json](execution.json).
  The later audit also matches all **11** current compiler/Luisa/timing artifacts
  against their recorded hashes. run.py itself records the Luisa hashes at
  startup only; do not describe it as having a new end-of-run artifact check.
- No production source or coefficients changed during the K experiment.
  The user-owned barrier flag 3→2 change remains present and uncommitted.
- No concurrent builds, hardware tests or GPU profilers were launched by this
  task. A separate later walk cohort was unstable; its times are not mixed
  into this one.
- Full selected-tree CMake build passed before binary execution. Small
  transposition/non-dyadic bounded-K tests remain in the prior checkpoint;
  this cohort does not extend that correctness claim to new operators.

Re-run the diagnostic only in a new output directory. Audit without rerunning
benchmarks:

```bash
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-k-partition/audit.py \
  --current-artifacts --self-test
```

## Documentation verification

The existing performance and matrix-mapping pages were rendered with Sphinx
and checked at 1440-pixel desktop and 390-pixel mobile widths. The rendered
six-row summary table matches the audited numbers exactly; manual image
inspection confirmed readable headings, diagrams and table contents. The
mobile table scrolls locally without overflowing the page. The check script
and receipt are [docs_qa.cjs](docs_qa.cjs) and [docs-qa.json](docs-qa.json).

The local documentation checker passes: 48 HTML pages, 3,642 local links and
assets, and 199 compatibility anchors. A fresh strict Sphinx build reports
the ten pre-existing missing-Doxygen-XML warnings in `api_reference.rst`;
the subsequent incremental build passes. This is not a claim that the fresh
whole documentation build is warning-free.

## Next decision and open questions

Retain K partition as a joint staged/JIT candidate with program geometry and
physical walk. Do not erase the ordered pipeline or its fences merely because
input views eliminate shared storage. First seek a repeatable held-out
improvement; only then update a backend-owned policy or its analytic prior.
How much of the remaining gap is physical K phase alignment versus the input
working set still requires separate profiling or a better controlled ablation.

Report layout: native repository Markdown/Sphinx, following the user's docs
structure. Exact two-order lookup tables are used instead of a curve or
statistical error bars, because there are only two exploratory observations
per candidate. Summary, evidence/definitions, method, uncertainty and next
questions are kept together; no new report site or documentation tree is added.
