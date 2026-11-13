OCR benchmark protocol (Phase 0)

Status: this file defines the benchmark protocol, workload catalog, result
schema, and qualification language for the OCR engine. The source baseline had
no harness (example_ocr_bench was only reserved); the current tree now provides
a dual-build, correctness-gated synthetic harness implementing the protocol
below. No result is valid without the provenance below. Real-model rows remain
unvalidated, and this document itself makes NO measured performance claim.

This document distinguishes at all times between:

- Protocol requirements: rules and definitions that future harness runs
  must obey (sections below).
- Catalog: declared, immutable workload IDs / shape families / execution
  order. A catalog entry is a specification, not a measurement. Only cells
  actually implemented and run by a later harness phase may produce rows;
  the catalog does not assert that every cell exists or runs.
- Planned expansion: dimensions and tiers defined here but scheduled for
  later implementation (Tier-2 targets, real-model cells, additional
  backends).

1. Source baseline and revision tracking

- Source baseline: 7269933deec007b60a5f4e0ee9aff163e7de0aee
  ("OCR examples"; parent 21b23c16545074611682ebce3ef53f57d70b77ba).
  This is the agreed correctness/performance baseline commit.
- The current tree HEAD may differ from the baseline (this document was
  authored at HEAD 267e0df815e2a018ebbe3c438a767a19c36276ec, which is a
  descendant of the baseline). Every result record MUST carry the exact
  commit it was measured on, plus a dirty-tree flag; readers must not
  assume the tree equals the baseline.
- Every result records: source commit, parent, tree state, build
  configuration and compiler/options, backend and plugin versions,
  GPU/driver/CUDA (or equivalent) runtime versions, OS, and power/clock
  state where available, plus input/package hashes for the workload.

2. Hard boundaries (must remain true in every phase)

1. Every real model is planned/unvalidated. Session::create and
   Session::recognize fail closed with `unvalidated`; the unsafe registry
   override does not enable inference. Benchmarks must not bypass this
   boundary and must never present synthetic output as real OCR output.
2. Synthetic timing is ENGINE timing only. It characterizes declared small
   graphs and operators, not official model throughput, quality, memory
   fit, or accuracy. No number in a synthetic row is a real-model result.
3. Real performance becomes publishable only after real qualification
   (section 16). No RTX 4060 figure may be inferred from RTX 4080 SUPER
   numbers; no full-model figure may be inferred from synthetic slices.
4. Reference paths remain available. Every new fast path is independently
   selectable, has A/B parity coverage and a kill switch, and enters the
   default path only for parity-proven recipe/precision/shape combinations.
5. Unsupported means unsupported: paged KV, true batched text decode,
   grouped persistent MoE GEMM, native shader fast paths, GDN/MoE/Surya
   CUDA graphs, W4A16, GLM MTP, Hunyuan DFlash, and host offload must not
   be silently emulated or relabeled (section 14).
6. Tests and benchmarks never regenerate goldens: the frozen synthetic
   fixtures in `src/tests/data/ocr/` are regenerated only by
   `tools/generate_fixtures.py`, never by tests or benchmarks; official
   goldens are exported explicitly by `tools/export_golden.py` into a
   separate directory and compared read-only.
7. Minimal, portable changes: Status/Result error style, no project-owned
   exceptions or RTTI, luisa::span transfers, host-only TUs separate from
   DSL/device TUs, and every dispatch routed through
   `ocr/runtime/dispatch_adapter.h`.
8. No benchmark result is produced by editing or "repairing" a backend;
   the documented upstream Vulkan device-creation regression is reported
   as blocked, never pass or skip-pass.

3. Non-goals

- No claim that any official checkpoint currently runs, fits, or is
  accurate.
- Assertion counts and test durations are not inference performance.
- No absolute latency/throughput promises before a measured baseline.
- No optimization of synthetic tiny geometries presented as full-model
  evidence.
- No wall-time performance thresholds in shared, variable-hardware
  correctness CI; benchmark executables are not CTest timing tests and
  carry no CI timing assertion (a correctness-gated smoke run of the
  harness families is acceptable; a timing assertion is not).
- No regeneration of synthetic or official goldens by tests or benchmarks.

4. Measurement layers

| Layer | Coverage | Primary questions |
|---|---|---|
| L0: host primitives | package read/validation, tokenizer/detokenizer, image decode/preprocess, DB/CTC/layout decode, scheduler planning | How much end-to-end time is host-only work? |
| L1: device operators | image, GEMM/GEMV and packed weights, norm/reduction, RoPE, attention, convolution/pooling, vision transforms, decode ops, GDN, MoE routing/gather/scatter | Which kernels dominate, and are they launch-, latency-, bandwidth-, or compute-bound? |
| L2: graph stages | dense prefill/decode, CNN detector/recognizer, layout overview/crops, P7 ragged vision, hybrid recurrent/chunked GDN, MoE prefill/decode | Where are device intermediates, readbacks, allocations, and synchronization boundaries? |
| L3: synthetic request paths | representative, then all 17 synthetic targets | Does an operator win survive executor/orchestrator overhead? |
| L4: pinned real requests | only officially supplied and qualified artifacts | What are real accuracy, latency, throughput, and memory on the declared matrix row? |

5. Suites and required command shape

The harness takes an explicit backend as its FIRST argument, then a suite
subcommand. Suites: `ops`, `decode`, `cnn`, `layout`, `gdn`, `moe`,
`model`, `all`.

- Explicit backend first: no implicit default backend; `fallback` and any
  future backend must be named explicitly.
- Selection: target/case/variant selection plus a workload manifest
  reference; warmup/iteration controls; JSON and CSV output paths.
- Optimization switches recorded per row: graph off/on
  (`enable_cuda_graph` forwarding), fusion off/on
  (enable_fused_residual_rms_norm), cooperative_norm off/on
  (--cooperative-norm; hidden-sized RMSNorm fast path with reference kill
  switch), layout cache miss/hit/disabled, recurrent/chunked GDN,
  prefill chunk size, candidate tile/block
  configurations, reference/fast-path arm. Every switch state is part of
  the immutable row.
- Raw samples are emitted incrementally so a later failing case cannot
  erase earlier evidence; skip/block/failure rows carry reason fields.

6. Timing definitions (noninterchangeable)

- `record_ms`: host command-list construction time only.
- `submit_sync_ms`: queue-idle commit through completed `Stream::synchronize`.
- request_ms: the full public/executor operation, including its inherent
  readbacks and synchronization.
- With --instrument on, supported request rows additionally emit raw
  samples[].stage and statistics.stage summaries. Defined fields are load_ms,
  preprocess_ms, vision_ms, prefill_ms, decode_ms, and postprocess_ms; absent
  stages are omitted. Qwen/P7 currently measure vision record+sync, host decode
  request assembly (preprocess_ms), and cache setup plus prefill/decode
  generation (decode_ms). These wall intervals use existing synchronization
  boundaries, add none, and need not sum to request_ms.

A host timer
"GPU time" must never be derived by subtracting independently sampled host
minima. Per-dispatch device attribution comes from the profiler protocol
(section 13).

7. Timing rules

1. Timing uses a release build with identical compiler/options for all
   compared arms. Debug/validation runs establish correctness but are not
   performance rows.
2. L1 kernels: correctness first (section 8); warm up at least 3 times;
   calibrate a repeated-dispatch batch to roughly 100 ms where legal;
   synchronize at the end; report the amortized dispatch cost together
   with the batch size. Retain a single-dispatch latency series when
   launch overhead matters.
3. L2/L3 requests: synchronize before and after each measured request; use
   full independent requests rather than repeatedly enqueueing stateful
   decode without resetting it. Exclude golden downloads, file output,
   profiling, and correctness comparison from the timed region.
4. Warm all lazy shaders, workspaces, graph variants, and allocators before
   warm timing; record the actual warm-up count/duration and cache state.
   Cold startup requires a fresh process; a same-process second create is
   warm, not cold.
5. Keep greedy output length and stopping behavior identical across A/B
   arms. Record generated steps, emitted tokens, and EOS/stop/truncation
   separately so an early stop cannot masquerade as a speedup.
6. Any failed correctness gate, unexpected fallback, timeout, cancellation,
   truncation mismatch, or nonfinite metric invalidates the timing row.
   Emit an explicit status; never report zero latency.
7. Report cold and warm processes separately (startup: device creation,
   package open/validate, component host staging, H2D upload, executor
   creation, tied-head preparation, shader compilation count, graph
   capture/instantiation, first request).

8. Correctness gates before timing

Correctness runs before and independently of timing, reusing the existing
independent double-accumulating CPU oracles and the existing per-suite
tolerances; no new blanket tolerance is invented.

- Linear/norm/patchify stages: the current 5e-4 absolute + 5e-4 relative
  gates; GELU-containing stages commonly 3e-3 + 3e-3 relative; dense
  decoder KV/logits and standalone softmax/argmax/rope keep their tighter
  existing gates (3e-4 + 3e-4 and 2e-4 respectively, per phase docs).
- CNN: existing conv/CTC tolerances, DB coordinate/score/confidence gates,
  exact CTC collapsed ids/text, and repeat-run determinism.
- GDN: exact Conv ring semantics, current recurrent/chunked/gate/norm
  tolerances, exact cache reuse behavior, and hybrid end-to-end parity.
- MoE: exact expert ids/counts/offsets/permutation and custom-mask bytes;
  current router/expert/model tolerances; validated routing readback
  before subviews.
- W4A32: compared against the dequantized-weight oracle, never against
  FP32 as though quantization were lossless; the declared synthetic
  quality metric is reported separately from real accuracy.
- All generative models: exact greedy token ids against the applicable
  oracle/reference arm and repeat-run identity. Batch-versus-1 output
  order and content must match exactly.
- Every optimized variant gets a small ragged proxy plus checks at the
  actual timed shape, including tail, empty, fully masked, NaN/degenerate,
  and boundary cases relevant to that operator. A failing variant is not
  timed.

9. Sample counts, warm-up, and distributions

- At least 20 timed samples for the initial micro benchmark; at least 10
  for long end-to-end cases; increase the count until the distribution is
  stable.
- Every raw sample is retained (immutable) and the summary reports: n,
  min, median, mean, p95, max, standard deviation or MAD, and an
  appropriate confidence interval for accepted comparisons.
- For an accepted optimization, repeat the full A/B in at least 3
  independent process runs and interleave/reference-randomize arm order
  to expose thermal and cache drift.
- Layout-cache miss/hit are separate series; fresh-cache and reused-cache
  decode series are separate; never blend them into one median.
- Seeded non-greedy sampling, where used, is a separate
  correctness/determinism series; greedy deterministic sampling is the
  performance default.

10. Workload catalog (declared, immutable IDs)

Every benchmark row is reproducible from a named, immutable workload
manifest: workload ID, seeds, actual shape/work counts, requested and
effective options, switch states, and expected refusal/truncation
outcomes. A published ID's shapes/seeds/options never change; new shapes
receive new IDs (planned expansion). Correctness suites map to catalog
entries without duplicating or weakening an oracle.

The catalog below is the INITIAL catalog: it declares scalable shape
families and the Tier-1 execution order. It does not assert that every
cell is implemented; harness cases are implemented in later phases and a
row exists only when measured under this protocol.

10.1 L0 host primitives (catalog v1)

| ID prefix | Cases |
|---|---|
| `host.package` | open/validate `.lcocr` v1 packages of declared sizes; hash-recorded inputs |
| `host.tokenizer` | tokenize/detokenize declared token-id streams; incremental UTF-8 across tokens |
| `host.preprocess` | decode + resize/normalize/patch at the page scales below |
| `host.db` / `host.ctc` / `host.layout_decode` | DB postprocess, CTC decode, layout decode on declared synthetic inputs |
| `host.scheduler` | region scheduling at declared page/region counts |

10.2 L1 scalable operator shape families (catalog v1)

Frozen parity fixtures in `src/tests/data/ocr/` are never altered for
performance; these scalable families are separate declared workloads.
Shapes are scalable points, not synthetic parity geometries.

| ID prefix | Family | Initial scalable axes (planned expansion adds points) |
|---|---|---|
| `gemm.square` | FP32 GEMM, all epilogues | M=N=K in {256, 512, 1024, 2048} |
| `gemm.decode` | decode GEMV / small-M | M in {1, 4, 16, 64}; N, K in {1024, 2048, 4096, 8192} |
| `gemm.gated` | SwiGLU/GeGLU gated linear | decode-axis points as `gemm.decode` |
| `gemm.packed` | packed-B FP16/BF16, grouped INT4 sym/asym, INT8 (W4A32 only; W4A16 refused) | N, K as `gemm.decode` and `gemm.square` midpoints; group 64/128 |
| `norm.rows` | LayerNorm/RMSNorm weighted + parameterless, L2, row-sum, residual | rows in {32, 256, 1024}; row length in {512, 1024, 2048, 4096, 8192} |
| `rope.seq` | 1D, 2D axial, M-RoPE, interleaved MRoPE, XD-RoPE (separate variants) | seq in {128, 512, 2048, 8192}; head dim in {32, 64, 128} |
| `attn.mask` | reference scores+softmax+PV vs online-softmax A/B; GQA/MHA; softcap; relative bias | mask kinds BIDIRECTIONAL/CAUSAL/WINDOWED/PREFIX_CAUSAL/CUSTOM_BLOCK_MASK; batch {1, 4}; heads {4, 16, 32}; q len {1, 64, 512}; kv len {128, 1024, 8192, 32768} |
| `conv.spatial` | Conv2D general/depthwise/groups/dilation, 1x1, Conv1D causal | spatial {256, 512, 1024, 2048}; channels {32, 64, 128, 256}; stride {1, 2}; asymmetric pad where declared |
| `image.resize` | fused color-convert + bilinear/area resize + normalize; crop; letterbox; patchify | source {256..4096 px, non-square/ragged}; target {224..2048}; patch {14, 16} |
| `vision.merge` | pixel-unshuffle, spatial merge/token packing, position add, DeepStack injection | merge {2, 4}; token counts at the patch buckets below |
| `decodeops` | embedding lookup (tied-head), multimodal scatter, greedy argmax, temperature softmax | vocab rows {1024, 16384, 131072}; hidden {256, 1024, 4096} |
| `gdn.state` | causal Conv1D ring, delta-rule recurrent vs chunked prefill, gate/norm | tokens {128, 1024, 8192}; state d {64, 128}; chunk {64, 128, 256} across declared boundaries |
| `moe.route` | router top-k, token-to-expert permutation, gather/scatter, active-expert loop | tokens {16, 128, 1024}; experts {8, 64}; top-k {2, 6}; balanced vs skewed routing; active-expert count reported as a covariate; two shared experts for the DeepSeek contract |

10.3 L2/L3 stage and request dimensions (catalog v1)

- Input geometry: blank, sparse text, dense text, multi-column,
  table/formula, mixed figure/text, wide line, non-square/ragged, and
  camera/deformed cases where declared.
- Page sizes: representative 256/512/1024/2048-pixel scales plus wide and
  ragged shapes. For real benchmarks: fixed 150/300-DPI document pages and
  the actual supported aspect-ratio/patch buckets.
- Document scale: 1, 2, 8, and 32 pages; input order and shuffled-page
  order; blank pages; pages with zero, one, few, and many regions.
- Vision/region batching: max_patches_per_vision_batch in
  {256, 1024, 4096, 16384} where legal; max_regions_per_batch in
  {1, 2, 4, 8}; exact-capacity and ragged-tail cases; record actual
  graphs/crops, not only requested limits.
- Decode: prompt/prefill lengths and visual-token counts; 1/8/64 output
  steps and longer real outputs; early EOS, stop string, max-token
  truncation, and cache-capacity truncation. max_decode_batch remains
  planning-only until a real batched executor exists.
- Precision: FP32 reference first; packed FP16/BF16 storage where the
  operator supports it; grouped INT4 sym/asym and INT8 coverage; W4A32
  model A/B against the dequantized-weight oracle. Never label W4A32 as
  FP16 activation math or W4A16.
- Memory envelopes: no explicit limit; exact-fit limit; one-byte/1-KiB
  insufficient limit; an RTX 4060 8 GB budget where that hardware is
  available. Expected fail-before-allocation results are preserved as
  refusal rows. In particular, the declared olmOCR-2 7B INT4/32768-context
  preflight exceeding 8 GiB remains a refusal case, not a fit claim.
- Modes: plain/spotting/markdown/extract only where the target declares
  and eventually qualifies them; strict schemas and exact language hints.

10.4 Tier-1 execution order (L3 synthetic request paths)

Run all targets eventually; establish distinct execution-path coverage
first. Synthetic request rows are engine rows and never upgrade any recipe
support state.

| Order | Target(s) | Execution-path reason |
|---|---|---|
| 1 | qwen3_vl_2b_instruct | Dense vision+decoder, interleaved M-RoPE, QK-Norm, DeepStack, graph/fusion A/B |
| 2 | pp_ocrv6:medium | Full CNN detector -> DB -> crop -> recognizer -> CTC path |
| 3 | paddle_ocr_vl_1_6 | Layout scheduler, NaViT, adaptive reduction, ERNIE decode, graph counters |
| 4 | open_doc_0_1b | Two-stage layout+UniRec CNN/CTC path with no autoregressive KV |
| 5 | deepseek_ocr_2 | Custom visual/query mask, 64 experts/top-6/two shared, routing readback and expert loop |
| 6 | surya_2_5 and ovis_ocr2 | Hybrid GDN recurrent/chunked/cache paths; Surya also covers detector/crops |
| 7 | hunyuan_ocr_1_5 | Only custom-VLM/XD-RoPE path and adaptive-pool connector |
| 8 | dots_mocr, olm_ocr_2 | W4A32 and memory-envelope-critical dense paths |
| 9 (memory/vision) | got_ocr2_0 | Window/global relative-position attention and its large expanded-bias scratch |

Tier 2 (planned expansion, after Tier-1 baselines): pp_ocrv6:tiny,
pp_ocrv6:small, fire_red_ocr, tele_ocr, glm_ocr,
mineru_2_5_pro_2605_1_2b. The tier order is an execution order, not a
claim that Tier-2 models need less qualification.

11. Memory accounting

Rows report, per component and total: actual resident weights including
executor-owned tied-head buffers; live and retired workspace; KV/GDN/Conv
state; shader-cache allocation where measurable; host staging; peak
host/device bytes; planned versus measured bytes; H2D/D2H bytes when
instrumentation can provide them honestly. Accounting reuses the existing
`memory_budget` / `workspace` / `weights` / `decode_cache` /
`shader_cache` mechanisms; no universal free-VRAM query is invented. A
claimed memory optimization must report byte changes and satisfy
planned==measured accounting where that invariant already exists.

12. Hardware/backend matrix

| Cell | Role | Status rule |
|---|---|---|
| CUDA on RTX 4080 SUPER | Baseline correctness/profile environment (see QUALIFICATION.md); release build for timing | Current documented evidence environment; timing rows name driver/runtime/build |
| CUDA on RTX 4060 8 GB | Primary resource target | Measured only when that hardware is available; record actual GPU, driver, CUDA/runtime, OS, build, and power/clock state. NO inference from RTX 4080 SUPER numbers. 8-GB boundary tests include honest refusal rows |
| DX and Vulkan | Portability | Only after device creation is healthy; rerun tuning rather than copying CUDA tile choices. The current upstream Vulkan device-creation regression is reported as blocked, never pass or skip-pass |
| Fallback | Optional correctness/oracle coverage | Only when its LLVM+Embree requirements are available; CPU timing is a separate backend result |
| HIP/Metal/other | Other backends | Only when explicitly available and supported; no unmeasured matrix cell is ever marked qualified |

13. Profiling protocol

- CUDA: Nsight Systems FIRST (timeline: per-kernel duration, launch gaps,
  synchronization boundaries, host/device attribution), then Nsight
  Compute on the kernels Systems identifies as hottest (achieved
  occupancy, register/shared usage and spills, memory throughput, dominant
  stalls). Roofline arithmetic intensity uses an explicit memory level and
  honest byte/FLOP accounting; masked-attention FLOP/byte formulas state
  whether they are dense upper bounds or use actual allowed pairs.
- DX/Vulkan: RenderDoc may inspect dispatch duration and resource contents;
  compute-only binaries need explicit RenderDoc application-API capture.
  RenderDoc is not an API validator and replay is not benchmark-identical
  execution, so replay numbers never become benchmark rows.
- Profile before structural changes; prioritize by end-to-end impact on
  representative OCR workloads (attributable time, not just the slowest
  isolated kernel, including host preprocess/DB/CTC and
  synchronization/readback cost).

14. Unsupported and refusal rows

Unsupported configurations are reported as explicit rows with reason
fields, never silently emulated, relabeled, or dropped: paged KV (fails
closed), true batched text decode (`plan_decode_batches` /
`max_decode_batch` stay planning-only), grouped persistent MoE GEMM (a
padded per-expert loop is not grouped GEMM), native shader fast paths,
CUDA graphs for the GDN/MoE/Surya executors (only dense/P7/Paddle paths
implement them), W4A16 (independent variant, fails closed), GLM MTP,
Hunyuan DFlash, and host offload. Memory-limit refusals (including the
declared olmOCR-2 8 GiB preflight case) and real-artifact `unvalidated`
refusals are expected outcomes and are recorded as such - a refusal is a
result, not a missing row. A row with any unexpected fallback, timeout,
cancellation, or truncation mismatch is invalid for timing.

15. A/B promotion and noise rules

- Do not promote on min-of-N alone. Establish a per-cell noise floor from
  the baseline distributions.
- Initial practical promotion targets: at least 5% for an operator or 3%
  for end-to-end request time, with the paired improvement outside measured
  noise and reproduced across independent runs. Smaller changes land only
  for another clear benefit such as lower memory or simplified invariants.
- A claimed memory optimization reports byte changes and satisfies
  planned==measured accounting where that invariant already exists.
- Investigate any confirmed critical-workload regression above the greater
  of 5% or the established noise floor; do not hide it by selecting a
  different shape or backend.
- Automated performance regression gates run only on controlled hardware.
  A proposed initial alert is a confirmed median regression above the max
  of 5% and the noise floor, or a p95 regression above 20%; calibrate these
  thresholds from baseline history before making them blocking.
- Every accepted optimization keeps the reference arm available behind an
  explicit switch and is independently reversible; rejected experiments are
  recorded with their evidence. Dense hidden RMSNorm uses the cooperative
  256-thread/shared-tree variant only when enable_cooperative_norm is true and
  hidden_size >= 32; QK-Norm remains reference. The benchmark flag
  --cooperative-norm off selects reference_norm rows without changing the
  independent fusion switch.

16. Real qualification gates (L4)

Real benchmark rows exist only for officially supplied, fully pinned
artifacts and only after every gate passes:

1. Exact checkpoint revision + config fingerprint + package
   hash/version + exact tensor signature, pinned in the registry.
2. Official processor/tokenizer parity.
3. Layered official goldens, exported explicitly (read-only comparison).
4. Deterministic end-to-end decode on real documents.
5. Accuracy/parser/coordinate gates per target (CER/edit distance,
   detection/line precision/recall and polygon IoU, layout mAP,
   reading-order accuracy, table structure score, formula and
   JSON/schema accuracy, coordinate error, blank-page hallucination
   rate) as applicable per MODELS.md.
6. Measured memory/performance on the declared hardware/backend/precision/
   mode row, recorded with full provenance.

While the registry remains unvalidated, the harness real mode refuses
(closed) for every real artifact; there is no unsafe "run anyway" option.
A synthetic benchmark never moves a recipe to `validated`; QUALIFICATION.md
rows change only to measured facts after all gates pass. Tier-2 and real
cells are planned expansion until measured.

17. Result schema and output layout

- Output root: `benchmark_results/ocr/` (repository root), a runtime-only
  area kept out of version control. Harness runs create per-suite
  subdirectories; no file under it is committed, and nothing under it is
  ever treated as a golden or reference image.
- JSON report per run, containing:
  - provenance: source commit/parent/tree state, build configuration,
    compiler, backend + plugin versions, GPU, driver, runtime versions,
    OS, power/clock state, harness version, workload manifest ID + hash,
    seeds, requested and effective options, switch states;
  - per row: suite, workload ID, backend, artifact_class
    (`synthetic` in all current phases; `real` only after section 16),
    model target when applicable, correctness status with the tolerances
    used, warm-up count/duration and cache state, n, raw samples
    (incremental, immutable), summary statistics (min, median, mean, p95,
    max, stddev or MAD, confidence interval), the three timing definitions
    plus stage breakdowns, work counts (pages, regions, patches, prompt/
    generated tokens, steps, dispatches), engine counters (shader
    compiles, layout-cache hits, P7 vision graphs/crops, CUDA-graph
    steps/updates/fallbacks, MoE active experts), memory categories
    (planned vs measured, peaks), and outcome (`ok`, `unsupported`,
    `refused`, `failed`) with reason;
  - nonfinite floats are rejected at serialization, consistent with the
    engine's strict JSON output.
- CSV companion: one row per raw sample (same provenance keys repeated)
  plus a summary block, so distributions can be recomputed without the
  JSON file.
- Throughput ratios always carry the actual work counts used (pages/s,
  regions/s, patches/s, prompt tokens/s, generated tokens/s, lines/s for
  CNN/CTC, requests/s).

18. Command examples (interface contract; NOT executed)

The `example_ocr_bench` executable does not exist at the source baseline;
the reserved notes in `xmake.lua` / `CMakeLists.txt` remain until the
harness phase wires it in. The following examples define the required
interface and placeholder output paths. They are illustrative command
templates only - no command below has been run and no output file exists.

```bash
# L1 operator family sweep on the baseline backend, correctness-gated:
example_ocr_bench cuda ops --workload gemm.square --samples 20 --warmup 3 \
    --out-json benchmark_results/ocr/ops/gemm.square.cuda.json \
    --out-csv benchmark_results/ocr/ops/gemm.square.cuda.csv

# decode stage with optimization switches recorded per row:
example_ocr_bench cuda decode --target qwen3_vl_2b_instruct \
    --graph off --graph on --fusion off --fusion on \
    --samples 10 --warmup 3 \
    --out-json benchmark_results/ocr/decode/qwen3_vl.graph_fusion.cuda.json

# full CNN path (detector -> DB -> crop -> recognizer -> CTC):
example_ocr_bench cuda cnn --target pp_ocrv6:medium \
    --pages 1,2,8,32 \
    --out-json benchmark_results/ocr/cnn/pp_ocrv6_medium.cuda.json

# layout orchestration with cache-state-separated series:
example_ocr_bench cuda layout --target paddle_ocr_vl_1_6 \
    --layout-cache miss,hit,disabled \
    --out-json benchmark_results/ocr/layout/paddle_ocr_vl.cache.cuda.json

# GDN recurrent-vs-chunked A/B:
example_ocr_bench cuda gdn --target ovis_ocr2 \
    --gdn-mode recurrent,chunked --chunk 64,128,256 \
    --out-json benchmark_results/ocr/gdn/ovis_ocr2.chunking.cuda.json

# MoE routing under balanced and skewed distributions:
example_ocr_bench cuda moe --target deepseek_ocr_2 \
    --routing balanced,skewed --topk 6 \
    --out-json benchmark_results/ocr/moe/deepseek_ocr2.routing.cuda.json

# Tier-1 synthetic model request paths:
example_ocr_bench cuda model --target qwen3_vl_2b_instruct \
    --steps 1,8,64 \
    --out-json benchmark_results/ocr/model/qwen3_vl.steps.cuda.json

# everything, explicit backend first:
example_ocr_bench cuda all --samples 20 --warmup 3 \
    --out-json benchmark_results/ocr/all.cuda.json \
    --out-csv benchmark_results/ocr/all.cuda.csv
```

19. Phase-0 exit gate

A reviewer can reproduce any future row from its manifest, can tell from
the artifact_class and provenance whether a row is synthetic engine
timing or a qualified real-model result, and cannot mistake synthetic data
for an official model result. QUALIFICATION.md remains the authoritative
qualification matrix and is not changed by benchmark phases; synthetic
benchmark results, when they exist, live in clearly separate sections or
files with this protocol referenced.

See README.md (README.md) for the phase status, MODELS.md (MODELS.md)
for the support matrix and unmet official gates, and QUALIFICATION.md
(QUALIFICATION.md) for the qualification matrix.
