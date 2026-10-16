# Common LLM coverage and partitioned-output mapping

September 7, 2026, Apple M1 Max, FP32. The RoPE mapping defect is repaired by
a generic output-region proof. Attention and direct XIR/SIMD remain far behind
Torch in the newly measured cases. This is not MPS/Torch goal completion.

The reader-facing results belong to the existing Sphinx hierarchy:
`docs/source/performance/tile/results.md`, sections “Partitioned outputs
remove the RoPE mapping fallback” and “Attention and direct SIMD still need
richer execution mappings”. Architecture belongs to
`docs/source/internals/tile/lowering.md`; no new documentation tree is added.

## Evidence inventory and metric definitions

| Directory | Role | Protocol | Expected rows |
|---|---|---|---:|
| metal-screen | Initial SwiGLU/RoPE and tiny/pilot attention | Two orders, 3 samples, 10 ms, 100 ms warmup | 16 |
| simd-screen | Initial SwiGLU/RoPE and decode compile limit | Two orders, 3 samples, 10 ms, 100 ms warmup | 12; 2 timeout failures |
| metal-replay | Frozen old/new compiler and Torch; four shapes each of RoPE/SwiGLU | All six orders, 5 samples, 20 ms, 100 ms warmup | 144 |
| simd-operators | Six non-GEMM operators | Two orders, 5 samples, 20 ms, 100 ms warmup | 24 |
| metal-prefill | GQA `(1,4,2,64,128,64,64)`, block 8×16 | Two orders, 3 samples, 10 ms, 100 ms warmup | 4 |
| metal-decode | GQA `(1,8,2,1,2048,64,64)`, block 1×32 | Two orders, 3 samples, 10 ms, 100 ms warmup | 4 |

Each directory contains `results.json`, native process logs and generated
source. Every native invocation uses the same shared fixture as
`test_tile_xir_llm`; inputs are exported before compilation so even a failed
native path retains its paired Torch measurement. The native full FP64 oracle
and 34 sentinel checks run before and after timing. Python independently
checks all native/Torch elements against FP64 formulas. Array sizes, absolute
errors, tolerances, exact input/output hashes and commands are recorded.
Transient tensors are deleted with their temporary directories after checks;
their contents are not archived. Re-run `compare_llm.py` with a compatible
build to regenerate them. FP32 inputs are deterministic transcendental data,
not an adversarial accuracy distribution or trained-model activations.

Warm E2E batches include dispatch/encoding/submission and synchronization,
excluding fixture construction, JIT, initial allocation/upload and downloads.
Native uses one Runtime CommandList per batch. Single-call E2E is separate.
GPU timings use the existing Metal helper in a separate phase: both the
instrumented compute-pass samples and uninstrumented command-buffer controls
are saved, with repetitions and probe perturbation. Reported GPU numbers use
the **controls**, including work/gaps inside command buffers, not isolated
kernel time. Medians of round medians describe times; ratios are medians of
paired round ratios, not ratios of displayed medians. Ranges are descriptive,
not confidence intervals. Desktop activity is not isolated.

## What changed, what did not

The production change is in TIRx element-grid admission. After retaining the
existing noalias, compact-layout, pure-producer, injective-output and effect
checks, two stores to one buffer may fuse when their linear addresses are
ordered for every pair of independent local-coordinate tuples within the
same program. Bounds and volume checks protect int64 linearization. Unknown,
overlapping, read/write or escaping regions retain original loops. No kernel
names, dimension tables, coefficients or runtime timing select the result.
This extends the planner's legal realization set; it does not fit a new cost
model or change the DSL, XIR, MPP matrix path or memory abstraction.

The old runtime/compiler was frozen, with adjacent ABI-coherent libraries,
after adding the common benchmark and before the proof change. Its paths are
in replay commands/artifact fingerprints. The exact same RoPE Tile capture
changes from whole-row private arrays to a fused element grid. All four
SwiGLU old/new Metal sources remain byte-identical. The replay validates 144
complete outputs, 757,027,260 elements in the Python check alone. Native's
additional two checks are not added to that count. Binary inventories remain
unchanged during each cohort. Initial screens predate the runner's optional
three-path replay; their script fingerprints therefore differ.

## Negative results and scope

RoPE wins all 24 GPU/E2E batch comparisons against old/Torch, but one ragged
single-call E2E round loses to Torch. Tiny SwiGLU loses four of six E2E latency
pairs, and 1024×4096 loses one. The initial two-round 37×1537 SwiGLU GPU screen
was slower than Torch; the six-order replay reversed that small result even
though its source is unchanged. Keep this pilot instead of presenting an
unqualified small-shape win.

Attention's larger prefill/decode pilots are approximately 127×/1316× Torch
GPU time. Both still have zero cooperative group plans and scalar loops over
whole private Tiles; source inspection does not isolate all spill/occupancy
costs. Fixed blocks are pilots, not a global autotuned optimum. Torch SDPA
uses an explicit bottom-right causal mask with GQA, allocating its returned
output inside timing. Tiny attention's win does not generalize. Native MPP
does not implement these operations; `benchmark_tile_native llm` explicitly
requests **TIRx/Metal** through Luisa Runtime.

Six direct XIR/SIMD operators lose all twelve paired E2E batch comparisons,
at 1.22–16.35× Torch. Requested CPU worker count is eight, packet width eight;
this is not a measurement of provider thread utilization. Two SIMD decode
attempts hit a 90 s process limit before source export. Inputs had already
been produced and Torch completed. A separate 8×16 block unit-test attempt
was stopped after 186.49 s; `execution-tests.log` preserves it. The final
unit suite uses 1×1, 2×4 and 3×5 blocks to keep routine SSA expansion bounded.
Neither this adjustment nor the passing tests certifies large Tile compile
scalability. Reduction, CNN/filter, sort, Top-K, FP16/BF16, paged KV, varlen,
autograd, other devices and end-to-end models have no new parity claim here.

## Validation and reproducibility

- `audit.py` / `audit.json`: independent inventory and raw-timing
  recomputation, input consistency, source SHA256, control identities,
  complete-output metadata, failed-case preservation and three mutation tests
  (missing row, forged ratio, incorrect GPU divisor). Run with Python 3.13.
- `execution-tests-final.log`: all three selected CTests pass after a complete
  build. CPU/Metal execution tests add 64 generic partition/policy cases;
  the Metal suite passes 818,965 assertions. `test_tile_xir_llm` now covers
  24 captures through both XIR/SIMD and TIRx CPU.
- `all-tile-tests.log`: 33/35 pass in the full Tile CTest rerun (231.09 s).
  The two retained failures are `test_tile_tirx_cooperative_metal` and
  `test_tile_tirx_memory_metal`, the existing user-owned barrier-flag/source
  expectation mismatch. Neither the local edit nor the assertions changed.
- All five changed C++ translation units passed the repository syntax checker;
  the benchmark Python suite passed 110 tests.
- Each cohort's `build.log` is the complete selected-configuration build gate,
  not a target-only build. Source/header fingerprints, toolchain/provider
  versions and library inventories are in `results.json`.

Use the main benchmark README's `compare_llm.py` commands with a fresh output
directory. Provide the actual external compiler/runtime libraries with
`--compiler-artifact`. Native command paths in archived records identify
this machine's build and ephemeral tensors, not portable existing inputs.
The Metal replay includes explicit TVM compiler/runtime hashes; auxiliary
screens record the adjacent Luisa libraries but not every external TVM
dependency. No simultaneous benchmarks/builds/profilers ran during timing.

## Report/QA contract and next work

The requested surface is the repository's existing Sphinx documentation.
Technical summary, definitions, exact per-shape findings, methods, negative
results/uncertainty and next steps are preserved there and in these supporting
notes. Exact tables are used because the relevant comparisons have multiple
timing/ratio fields and widely different scales; a single aggregate bar or
trend would hide the shape/method distinctions. No charts, dashboard, Sites
publication or replacement report hierarchy is added. `qa_docs.cjs` and
`docs-qa.json` record six sections at 1280/390 px and four additional narrow
table-right views: no page-level horizontal overflow or JavaScript errors,
expected table dimensions, and accessible final columns. Visual inspection
checks the revised text, figures and tables; screenshots remain temporary
at the receipt paths rather than becoming a second published report.

The fresh Sphinx tree passes `scripts/check_docs.py`: 48 HTML pages, 3,728
local links/assets and 199 compatibility anchors. The strict Sphinx build
still exits 1 with ten existing missing-Doxygen-XML warnings in
`api_reference.rst`; none is suppressed. A first link-check invocation omitted
the Sphinx dependency and failed at import; rerunning with `--with sphinx`
passes. Final syntax checking and all 110 Python tests pass. Inspection of all
270 evidence files finds only text artifacts; all 124 successful native raw
logs agree with their recorded operation, shape, samples, repetitions,
correctness and realization metadata.

Priority: a generic bounded Tile-element realization on XIR, and a composed
Metal MMA/reduction recurrence with correct ownership of carried state. Only
after those legal candidates exist should their resource costs and staged/JIT
ranking be tuned. Scalar-DAG reuse, matrix profitability and the broader
MPS/Torch goal remain unfinished.
