# Torch collective implementation evidence and Tile scheduling candidates

Inspected locally on 2026-10-03 without importing Torch, compiling, or launching GPU work.
Installed Torch is `2.14.1+cu130`, git `5c4886908584029761b579af026dcfb627c84070`;
Triton is the installed Windows distribution `3.8.0.post29` (compiler reports 3.8.0).
{download}`The compact evidence <../../../scripts/benchmark/tile_torch/results/2026-10-03-torch-collective-source/evidence.json>` records nine historical validated case packets, generated wrapper hashes, retained autotune configurations and installed source receipts. Each matching retained TTGIR/PTX/metadata artifact was selected by first load tensor shape plus warp count. These are compiled-artifact observations, not
a new launch trace or new measurements.

## Inspected cases

These native/Torch medians are from the same historical October 2 default cohort;
they are not mixed with October 3 timing denominators. All share the existing full
FP64 output oracle. Retained `.best_config` files are inspected now and their new
hashes are recorded; they were not independently hashed in the original packet.

| Case | Native/Torch us | Actual generated route and retained configuration |
|---|---:|---|
| SUM 17x1024 FP16, native BR4 | 2.1991 / 0.9081 | persistent, XBLOCK=1, width 1024, 8 warps |
| MAX 17x1024 FP16, native BR4 | 2.1648 / 0.9136 | persistent, XBLOCK=1, width 1024, 8 warps |
| Scan 17x1024 FP16, native BR4 | 2.6952 / 1.0026 | persistent, XBLOCK=1, width 1024, 16 warps |
| Scan 128x8192 FP16 | 12.3441 / 5.4653 | looped scan, XBLOCK=2, R0_BLOCK=2048, 16 warps |
| Scan 128x8192 BF16 | 12.3568 / 5.4798 | same schedule and algorithm as FP16 |
| RMSNorm 1x8192 FP16 | 2.8901 / 1.4144 | looped template, XBLOCK=1, R0_BLOCK=8192, 16 warps |
| RMSNorm 128x8192 BF16 | 7.7322 / 4.8822 | looped two-pass template, XBLOCK=1, R0_BLOCK=2048, 16 warps |
| LayerNorm 128x8192 BF16 | 11.2161 / 8.0911 | looped Welford + reread, XBLOCK=1, R0_BLOCK=2048, 16 warps |
| Softmax 1024x512 FP16 | 5.7206 / 3.2822 | persistent fused MAX/EXP/SUM/DIV, XBLOCK=2, 4 warps |

Each inspected wrapper launches one generated Triton kernel, with no external
ATen/cuBLAS call. This does not imply every Torch operation uses one kernel.

The long narrow scan is **not** a decoupled-lookback implementation: its generated
body loops over four 2048-element chunks, calls `tl.associative_scan`, keeps an
FP32 carry, selects the chunk's last prefix, adds the carry, and finally stores
the narrow output. Its unique matching TTGIR has sizePerThread `[1,8]`,
threadsPerWarp `[1,32]`, warpsPerCTA `[2,8]`, order `[1,0]`; PTX contains 128-bit
global vector loads and stores and shared-memory exchanges. No local-memory
instruction form was found in that retained PTX. This is PTX evidence, not a SASS
register/spill count. Merely adding a 2048 chunk loop is insufficient: our measured
streaming candidate also chunks but regresses. Candidate identity must include
independent rows, reduction width, legal worker arrangement and memory layout.

Softmax's unique retained layout is `[1,8]` elements/thread, `[1,32]` threads/warp,
`[2,2]` warps/CTA, with 128-bit vector memory forms. RMS 128-row's retained layout is
`[1,4]`, `[1,32]`, `[1,16]`, with 64-bit vector memory forms. These are concrete
layout differences, not evidence that a single vector-width flag will reproduce
the performance in CUDA Tile.

RMS 128-row keeps a chunk-shaped square accumulator, reduces after the input loop,
then rereads input and gamma in a second loop. It uses `libdevice.rsqrt`, followed
by multiply. LayerNorm uses Welford state in its first pass, then rereads input,
gamma and beta. These trade extra global reads against live values. They are
different arithmetic expression graphs from the strict original Tile RMS/LN;
they cannot be silently substituted under an exact math contract. Existing
bounded-error benchmark passes are evidence for those fixtures, not a proof of
unrestricted equivalence. Prefer scheduling-only changes first.

## Where the installed decisions come from

Paths below are under `.deps/torch-cuda-venv/Lib/site-packages/`.

- `torch/_inductor/choices.py:442`: persistent reductions default to INNER width
  <= 1024 (other reduction hints use 64), with explicit symbolic padding and
  cooperative/multi-kernel conditions. This is a heuristic, not an oracle.
- `choices.py:532`: split reductions depend on output count versus SM count,
  reduction length and device capacity. On this pre-SM100 target, INNER width
  <= 8192 or outputs >= 2*SM generally avoids split. Our nine inspected wrappers do
  not demonstrate a split reduction win.
- `heuristics/triton_codegen/reduction.py:182`: ordinary candidates vary both
  XBLOCK and R0_BLOCK, not just num_warps. Candidate seeds include contiguous,
  tiny, outer, 64x64, 8x512 and 64x4. CUDA R0_BLOCK seed cap is 2048 on this target;
  high loads+reductions and many rows reduce it to 1024.
- The same file `:330`: persistent candidates keep the whole reduction width,
  vary independent XBLOCK, and cap the usual block product at 4096 except the
  XBLOCK=1 fallback. `runtime/triton_heuristics.py:3984` chooses/configures warp
  counts from contiguous work and register-intensity estimates.
- `runtime/triton_heuristics.py:915`: **post-compilation** register feedback can
  halve R blocks when registers limit occupancy and enough blocks exist to
  benefit. `:1472` rejects non-custom candidates exceeding the spill threshold.
  These use actual compiled resources; our IR live-byte estimate is not a
  substitute for physical register counts.
- `runtime/coordinate_descent_tuner.py:116,192` searches X/Y/Z/R block fields and
  num_warps using neighboring powers of two. `triton_heuristics.py:2283` disables
  order-changing reduction tuning in deterministic/strict-sum modes.
- `codegen/triton_split_scan.py:20`: true split scan uses grid
  `(ceil(reduction/RBLOCK), rows)` and a zeroed global communication workspace,
  with decoupled lookback. It is a separate algorithm candidate, not what the
  inspected long scan uses. It needs progress/memory-order/graph workspace proof.
- `triton/backends/nvidia/compiler.py:278`: conversion to TTGIR is parameterized
  by num_warps, followed by coalescing, layout-conversion removal and thread
  locality passes. `language/core.py:3033,3141` exposes generic combine regions
  for reduce/scan. The row/width/layout features are not tied to RMS or softmax
  names in the compiler.

Autotune is not an infallible stable reference. The old/new FP32 long scan uses
an identical Triton template `1a1498957a1cf680bfcc3e57705d251a0f6692733935cec75710d158647f46f9`, but old retained configuration is
XBLOCK=1/R0_BLOCK=4096/16 warps and fresh is XBLOCK=8/R0_BLOCK=512/4 warps. Their measured
medians are 10.480 and 19.808 us respectively. Seven fresh samples are 19.674 to 19.962 us.
The compiled-template identity and different cached configuration are a concrete
confound; whole-process telemetry does not isolate its causal contribution.
The public validation report preserves fresh denominators and makes no claim of
a new or stable native FP32 scan win.

## Shared Tile IR representation and next priorities

Keep scheduling alternatives in compiler analyses and lowering candidates whenever
the existing Tile values, collective operations and execution nests can express
them. A different worker layout, memory representation or implementation library
does not by itself justify another DSL primitive. Before adding an execution-nest
entity, provide a minimal example and demonstrate why the existing primitives
cannot express its semantics correctly and efficiently. The candidates below
operate on existing IR and require no new DSL entities.

An inefficient lowering in one backend is not by itself such a counterexample.
First check whether a transformation of existing execution nests, a different
memory layout, or a backend implementation choice can remove the limitation.
Keep logical work and state facts separate from physical resource measurements;
neither a smaller Tile nor fewer collective calls alone proves a faster plan.

1. **Repartition independent rows into more programs.** Prove the complete SSA
   expression, memory operands and result independently separable along a named
   axis. Candidate records original rows/program, target rows/program, original
   and candidate grid, masks/tail and exact root spans. BR4 on 17 rows produces
   five native programs versus seventeen at BR1; that is a concrete parallelism
   difference. The historical BR1 sum/max/scan samples around 0.9 to 1.0 us are useful
   evidence for investigating this direction. The implementation keeps overlapping
   input/output views on the original entry/grid; a per-program snapshot must not
   be turned into cross-program races. `IndependentCollectivePlan` records the
   proof independently from CUDA entry generation.
2. **A real plan space over retained versus streamed reduction state.** Candidates
   should distinguish whole-row persistent, sequential contribution chunks with
   FP32 carry/state, and reread/rematerialize epilogues. Describe state bytes,
   reread bytes, chunk count/serial dependency length, independent programs and
   epilogue work. The current streaming scan is measured negative on 8/8 cases;
   do not select it from logical state reduction alone. For normalization, looped
   reread candidates require explicit numerical permissions or an unchanged
   expression/order proof. Multi-pass/split reduction is later work, not needed
   to validate the first row-partition candidate.
3. **Backend resource feedback as optional cost inputs.** Shared analysis should
   expose contribution width, independent extent, program count, tile axes and
   contiguous stride, storage/compute widths, mask utilization, sum/max/prefix
   algebra, elementwise and transcendental work, actual load/store repetitions,
   and alias-disjoint proof. A backend may add measured compiler registers,
   shared/local memory, eligible vector width and documented worker candidates.
   Unknown resource data stays unknown. Do not hardcode operation names, benchmark
   IDs, or a claim that logical bytes are registers. Predict candidates relative
   to the original plan; preserve fallback reason and cost of extra work.

CUDA Tile's documented supported worker hints are 4/8, unlike the inspected
Triton 16-warp launches. Do not expand the CUDA Tile hint whitelist based on
Triton's separate API. The checkpoints below measure individual candidates;
none establishes that one shared plan or resource model closes every gap.

## Implementation status at this checkpoint

Commit `0154c82ec` implements the opt-in independent-program partition for
SUM/MAX with FP32 computation and FP32/FP16/BF16 storage. Host plan/codegen and
GPU correctness checks passed, including tail rows/columns, full output bounds,
read-only buffers and guards, overlapping-view fallback, actual graph template
CUfunction/grid observation, and graph updates in both directions. Direct
launches have full numerical checks; this report does not claim a separate
Driver trace for those launches.

The request is `LUISA_CUDA_TILE_PROGRAM_ROWS=1/2/4`, for an original BR4/8 and a
strictly smaller target. The additional entry is `luisa_tile_partition`.
Metadata records the requested rows, availability, exact input/output slots and
byte spans, and both launch grids. Default/Torch processes clear the opt-in;
simultaneous experimental scheduling settings are rejected.

The subsequent {download}`16-case calibration checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-program-partition/README.md>`
contains the original, rows1, rows2 and original-recheck measurements, raw samples,
negative candidates and a reproducible nonnegative fit. For example, SUM
65x2048 FP16 changes from 6.3262 to 1.1212 us with rows1, while rows1 regresses
SUM 1024x512 FP16 from 2.7475 to 3.1183 us. The plan must compare candidates
against the original; more programs alone is not a universal improvement.

`IndependentCollectiveWorkFacts` supplies exact logical program counts,
valid/padded extents, per-program collective volume, storage/compute component
sizes and mask facts for each proved candidate. These component sizes are not
added into a fabricated peak or interpreted as physical register allocation.
The optional `LUISA_CUDA_TILE_PARTITION_COST=1` policy evaluates original,
rows1 and rows2 with the frozen SM89/24-SM/CUDA13.4 profile, requiring more than
5% lower predicted score. Unavailable or unsupported candidates retain the
original. The profile and fit identifiers, original/selected scores, rows and
reason are recorded; actual invocation alias guards still govern entry/grid
selection. Other targets and fast math do not use this strict profile.

This fit is a ranking proxy with substantial timing residuals, not an accurate
runtime predictor. Its calibration cross-validation is separate from the
{download}`12-case independent validation <../../../scripts/benchmark/tile_torch/results/2026-10-03-program-partition-heldout/README.md>`.
That frozen six-geometry inventory passed all numerical checks in two policy
runs and an original recheck. Eight cases improved, two retained the original,
and SUM/MAX 257x128 BF16 regressed about 9.4%/5.8%; the regressions repeated.
The fit and admission rules were not changed after observing those outcomes.
The policy remains explicitly experimental and off by default. These failures
motivate adding structured-memory representation and observed compiler-layout
costs rather than treating logical collective volume as a sufficient model.

The {download}`aligned structured-load checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-aligned-view/aligned-view-sm89.md>`
uses the existing `LUISA_CUDA_TILE_IR_ALIGNED16=1` alternate entry to preserve
rectangular load information with `tensor_span` and `partition_view`. The proof
requires full bounds and chunk-aligned origins; the host checks final buffer
pointer alignment. The original entry and all arithmetic remain unchanged.
Across 48 native and 16 same-cohort Torch runs, all output checks passed.
Four 256x128 SUM/MAX cases improved by 31--35% relative to the original and
5--23% relative to Torch. However, FP16 scan128x8192 regressed by 37%, so the
representation remains opt-in. All 448 timing samples and the unchanged-source
and default-recheck evidence are retained; no operation-name exclusion masks
the negative result. This is another reason to model compiled layout costs
alongside logical work rather than select from vector width alone.

The aligned load proof also admits a zero-padded partial partition when each
chunk origin is nonnegative, aligned to its Tile extent, and still intersects
the corresponding logical dimension. Index arithmetic must remain in range.
Custom fills, entirely out-of-bounds chunks and unproved origins retain the
ordinary masked load. This extends representation coverage; it is not evidence
that every newly admitted shape becomes faster.

`analyze_closed_prefix` shares the existing load/cast/prefix/cast/store closure
proof between CUDA realizers. It records actual dimensions, root ranges and
logical work, without adding an execution primitive or choosing a thread layout.
The existing streaming Tile emitter consumes that proof through its original
CUDA-specific constraints.

For explicit experiments, `LUISA_CUDA_TILE_CUB_SCAN=128|256|512|1024` selects an
ordinary CUDA/CUB realization with that many physical threads, eight contiguous
storage elements per thread, and an FP32 carry between chunks. It also requires
`LUISA_CUDA_TILE_IR=1`. The current subset is one complete row per program,
FP16/BF16 storage, unordered FP32 inclusive addition, and a logical width divisible
by `8 * threads`. It is mutually exclusive with other experimental scheduling
options. The original Tile source and module are retained; this option does not
enable an automatic cost policy.

The alternate module uses strict FP32 NVRTC options. The shared runtime selector
checks final buffer pointer alignment and complete input/output byte intervals;
overlap or failed alignment restores the original function, grid and block.
Graph construction and typed updates use the same selector. Both modules remain
owned by the shader, whose lifetime must cover graph use. Ordinary NVRTC compile
errors and candidate module/entry/resource failures keep the original shader;
the existing compiler helper's fatal infrastructure errors remain unchanged.

Candidate source is dumped separately when `LUISA_DUMP_SOURCE` is set. Its
64-bit compile key is not a SHA256 or a complete transitive header identity.
Optional diagnostics retain actual loaded-function registers, static shared and
local bytes, with unknown/error status preserved. The ordinary CUDA candidate
also records the occupancy API's resident CTA capacity for its physical block.
That capacity is not measured occupancy. The original Tile entry receives only
static resource queries: its logical block `(1,1,1)` does not reveal physical
workers and is not used to compute occupancy.

`LUISA_CUDA_TILE_CUB_SCAN_COST=1` optionally chooses among the original and
the four ordinary CUDA scan recipes. It uses the same shared closed-prefix
proof and unchanged source generator, with no additional Tile DSL operation
or execution nest. It is mutually exclusive with fixed-thread CUB and the
other experimental scheduling options. It is off by default and currently
admits only strict math on the frozen SM89/24-SM/CUDA 13.4 profile; other device
or compiler versions retain the original.

The ranking inputs are actual row/column extents and storage bytes from IR,
physical threads and elements per thread from the candidate, and maximum
resident CTA capacity queried from that candidate's live compiled function.
For rows R, width N, SM count S, physical threads T, warp width w and queried
capacity A, the CUB features are `[1, 4*R*N/(S*2^20), 8*H*J, H*J*ceil(T/w)]`,
where J=N/(8*T) and H=ceil(R/(S*A)). H is a capacity-batch proxy, not observed
execution waves. The original features are
`[1, 4*R*N/(S*2^20), ceil(R/S)*N]`. Frozen nonnegative coefficients rank these
features; benchmark names and current-dispatch timings are not inputs.
The selected score must be strictly below 95% of the original score.
Unknown resources, nonzero local memory or inconsistent launch facts exclude
the candidate rather than becoming a zero cost.

The original is compiled first. Cost mode visits at most four recipes,
keeps at most the best module plus the current module, and unloads losers in
their owning CUDA context. It installs the winner once. Every invocation still
checks the final buffer alignment and disjoint intervals; a model prediction
does not authorize aliasing changes. Only the installed winner receives an
`installed-entry` resource receipt. Other resource records describe historical
queries of a `loaded-candidate` and carry their final disposal status.
The reported compiler-call count can include in-process PTX cache hits, and
search time is host setup time excluding original Tile compilation; neither
is a count of NVRTC cache misses or a dispatch measurement.

The profile was frozen before independent shapes and a second selected-recipe
measurement. Those measurements establish recipe ranking separately from
automatic selection and graph tests. The policy is a local experimental model,
not a claim that all scans or reductions beat Torch. Its provenance and measured
scope are retained in the
{download}`scan cost evidence <../../../scripts/benchmark/tile_torch/results/2026-10-03-cub-scan-cost/README.md>`.

The separate
{download}`automatic-mode measurements <../../../scripts/benchmark/tile_torch/results/2026-10-03-cub-scan-cost/README-automatic.md>`
retain the subsequent real factory-search results. Their eight independent
cases average about 45% of original Tile dispatch time and 69% of same-round
Torch time, but two cases remain slower than Torch. A large scan also alternates
between roughly 13 and 33 microseconds despite unchanged selected source and
resources; the cause remains unresolved. These observations do not establish
stable automatic/fixed-recipe performance parity. Candidate search adds seconds
of host setup time, excluded from dispatch ratios. The failed eligibility
control and all timing outliers remain recorded separately.

The nine source-review observations above remain fixed to October 2;
the separate calibration report retains its own October 3 denominators and
the original Torch process failure followed by an independent retry. Neither
report claims that all operations close the Torch performance gap.

The source and generated-artifact paths in the evidence are provenance labels; those local files are not bundled. Their SHA256 values refer to the original bytes. The compact evidence projection has a distinct receipt in its own directory. No new program-partition measurements are included in this source review.

## Existing-IR streamed SUM prototypes

The separate {download}`streamed SUM checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-streamed-sum/README.md>`
tests two owned-IR transformations using existing SERIAL, Tile carries and
REDUCE. V3 reduces each contribution chunk and carries a scalar. V4 carries an
FP32 chunk-shaped Tile through the loop and performs one final reduction. Both
preserve the original unordered FP32 policy, root ABI, logical tail checks and
row-only epilogue; a full input/output range-disjoint check guards the experiment.
Ordered reducers, escaping full-width values and unsupported effects reject.
Neither prototype adds a DSL primitive or changes production defaults.

All 24 V3 candidates were slower than their same-round original recheck, up to
2.369 times its time. V4 improves small-row FP32 by about 4--6%, while most
128-row cases remain close to the original. It also regresses: BF16 3x16384 with
a 2048-element carry takes 2.177 us versus the original recheck's 1.405 us.
Reducing collective calls and logical state therefore does not suffice as a
selection model. No automatic streamed-SUM policy is installed from these data.

The packet retains all 96 native measurements, 12 fresh Torch measurements,
756 primary graph samples, generated source, fixture manifests and correctness
receipts. Fresh Torch consumes the exact V4 inputs and original FP64 bounds,
after the four native stages. Original native code is faster in nine cases;
three remain slower. Torch's two large FP32/FP16 cases are particularly slow in
this round, so these ratios alone do not establish stable cross-run superiority.
The independent reproducer checks included bytes and timing arithmetic; large
tensor binaries are retained locally rather than bundled. V4's host, finite
GPU and exceptional-value checks pass, including actual graph function and
argument observation. These correctness results do not establish a speedup.

The separate {download}`aligned-load isolation checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-streamed-sum-alignment/README.md>`
tests the existing structured-load realization against plain loads for the
original and both V4 carry widths. All 16 aligned streamed candidates are
slower than their same-carry plain controls, by 1.54--2.41 times. Every timed
graph's actual entry and final pointers were checked. The aligned variants
report 8 or 16 KiB of static shared memory versus 16 bytes for plain loads;
this is an observed resource difference, not a proven explanation.

The original whole-row kernels behave differently: several FP16 cases improve
by about 6--9%, whereas BF16 3x16384 becomes about 82% slower even though its
register count falls from 137 to 46. Register count alone is therefore not a
sufficient selection criterion. The packet retains all 56 measurements and
392 primary samples, including regressions and both plain controls. Its Torch
appendix contains static generated-code observations from the earlier run;
it supplies no timing denominator for this cohort. Neither the load variant
nor the streamed transformation becomes an automatic policy from these data.

## Optional CUDA subgroup realization through TIRx

`planner.cuda_subgroup_reductions` enables the existing proved reduction-program
mapper on the CUDA-source device-artifact route. It is off by default and adds
no Tile DSL operation or execution-nest primitive. The existing packing,
lane-elements, stripe-cache and unroll controls describe the physical mapping.
They are experimental constraints, not a calibrated CUDA cost model.

The initial scope is unordered FP32 SUM/MIN/MAX with recognized identities,
supported typed snapshots, proved ownership, complete 32-lane participation and
uniform block synchronization. FP16/BF16 storage retains its explicit conversion
points. The enabled planner, CUDA-source target and noalias contract are required;
NVPTX, conflicting Metal capabilities and unproved mappings reject. Runtime
dispatch retains full buffer-range alias checks, including nonzero view offsets.

Strict subgroup compilation disables FTZ and contraction and requests precise
division/square root. Explicit fast math retains CUDA's elementwise policy;
local and warp reduction merges use non-FTZ PTX in both modes. MIN/MAX suppress
a single NaN, preserve signed-zero ordering and retain their original infinity
identities for all-NaN inputs. NaN payload preservation is not promised. Existing
default CUDA and Metal realizations do not enable this capability implicitly.

Host tests cover target/resource rejection, typed fused programs, reordered
bindings and synchronization counterexamples. CUDA runtime tests additionally
check subnormals, signed zeros, infinities, NaNs, packed tails and offset views.
The first six-case RMSNorm/softmax experiment used exact T128/P2/lane8/unroll1,
without input caching. All six candidates were slower than the same-round native
recheck, by 1.33--3.00 times, with native drift below 2%. This is a correctness
baseline for further lowering work, not a selection policy or performance win.
The original sources, all 168 primary timing samples and actual graph/resource
observations are retained in the
{download}`CUDA subgroup checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-cuda-tirx-subgroup/README.md>`.

Three further isolated comparisons retain 54 executions and all 378 timing
samples in the
{download}`CUDA subgroup tuning checkpoint <../../../scripts/benchmark/tile_torch/results/2026-10-03-cuda-tirx-tuning/README.md>`.
At lane8, increasing the existing unroll constraint from 1 to 8 improved the
BF16 tail RMSNorm case by about 30%; most other changes were small. The lane1
experiment improved five cases but regressed that tail case by about 32%, with
304 local bytes reported by the compiled function. This experiment used a
temporary narrower second-level collective, so its timings do not validate
lane1 on the original collective implementation.

A separate full-tree/narrow-tree/full-tree comparison then found a 15.7%
large-softmax regression against the final control, despite identical register,
shared-memory and local-memory attributes. The narrow-tree implementation was
withdrawn. Fewer shuffle instructions and smaller source-level state are not
sufficient profitability proofs. These experiments reuse existing execution
and layout primitives, preserve their losses, and contain no new Torch
denominator or automatic CUDA selection policy.
