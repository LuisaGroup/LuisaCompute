# OCR engine architecture

This document describes the frozen Phase 0 contracts, the Phase 1 host
infrastructure, the Phase 2 portable tensor runtime and operators, the
Phase 3 dense autoregressive decoder core, the Phase 4 Qwen3-VL vertical
slice + FireRed-OCR recipe + structured outputs, the Phase 5 CNN OCR executor
+ PP-OCRv6 contracts, the Phase 6 layout/region infrastructure + PaddleOCR-VL
+ OpenDoc vertical slices, the Phase 7/8 custom and dense ViT slices, the
Phase 9 hybrid GDN subsystem, the Phase 10 MoE subsystem, the Phase 11 opt-in
execution features (exact-capacity P7 ragged crop chunking, optional CUDA
graph decode, optional fused residual+RMSNorm), and the Phase 12 offline
qualification surface (--list-models support report + QUALIFICATION.md).
Nothing here is a claim that a REAL model runs. Graph contracts are transcribed
from the requirement as contracts to verify against pinned official
checkpoints/configs/reference implementations before any recipe is implemented
(plan Phase 0 task 1). No official artifact has been inspected yet; all
fixtures are synthetic. Every implemented slice is validated only against
synthetic CPU oracles, not against any checkpoint; Session fails closed with
unvalidated. See QUALIFICATION.md for the current 17-target matrix.

## Layering

```text
example_ocr.cpp CLI entry: UTF-8 argv (Windows wmain) -> parse -> validate -> Session
ocr/cli.{h,cpp} reusable UTF-8 argument parser + structured status report
ocr/api.h stable request/result types (host-only)
ocr/session.{h,cpp}        one Session = one model; orchestration + preflight
ocr/model_registry.{h,cpp} exact-signature, fail-closed recipe selection
ocr/model_recipe.{h,cpp}   identity + graph/preprocessing/prompt contracts
ocr/status.{h,cpp} Status / Result / Diagnostic (no exceptions)
--- Phase 1 host artifacts ---
ocr/package.*, tokenizer.*, image.*, preprocess.*, ... host artifacts
ocr/output/* structured CPU output (reading order, parsers)
--- Phase 2 portable runtime + operators (implemented) ---
ocr/runtime/tensor.* logical shape/stride/dtype/layout + non-owning views
ocr/runtime/weights.* immutable per-component device weights + staging lifetime
ocr/runtime/workspace.* grow-only aligned arenas + host planning
ocr/runtime/shader_cache.* typed per-owner cache + stable variant keys
ocr/runtime/dispatch.* TOTAL-thread plans, chunking, flattened 2D grids
ocr/runtime/memory_budget.* checked accounting + limit preflight
ocr/ops/* device-free DSL kernel factories + independent CPU oracles
--- Phase 3 dense decoder core (implemented; synthetic parity only) ---
ocr/models/qwen2_5_vl.* shared decoder geometry/config/tensor validation
ocr/runtime/decode_cache.* contiguous + optional paged KV, device metadata
ocr/runtime/executor.* command-recording executor contract + ownership
ocr/runtime/dense_vlm_executor.* dense blocks + prefill/decode + controller
--- Phases 4-10 per-model slices (implemented; synthetic parity only) ---
ocr/models/* per-model graphs; CNN/layout/GDN/MoE executors
```

Rules:

- Host-only code never includes DSL/device headers. Kernel factories (later
  phases) are device-free and compiled once per model/shape/format variant.
- Recoverable failures are `Status`/`Result` values. No project-owned `throw`,
  no RTTI. `LUISA_ASSERT`/`LUISA_ERROR` only for broken internal invariants.
- All host transfers use luisa::span (safe-mode compatible); host spans and
  buffers stay alive until command completion; replaced workspaces/graph
  resources retire only after the owning stream drains.
- CLI encoding: Windows narrow argv is the ANSI code page, NOT UTF-8. The
  entry point uses wmain + WideCharToMultiByte(CP_UTF8) so paths AND
  prompt/schema/language/guidance reach the parser as UTF-8; POSIX main
  passes UTF-8 through. The parser decodes paths with path_from_utf8 and
  rejects invalid UTF-8; ACP bytes are never silently treated as UTF-8.

## Execution-path families

`ArchitectureFamily` distinguishes the paths directly; recipes must not
collapse them:

| Family | Executor (implemented; synthetic parity only) | Models |
|---|---|---|
| `cnn` | CnnExecutor | PP-OCRv6 tiny/small/medium (detector + CTC) |
| `dense_vlm` | DenseVlmExecutor | TeleOCR, FireRed-OCR, Qwen3-VL-2B-Instruct, dots.mocr, GOT-OCR2.0, olmOCR-2 |
| `layout_vlm` | LayoutVlmExecutor | PaddleOCR-VL-1.6, GLM-OCR, MinerU2.5-Pro, OpenDoc-0.1B |
| `hybrid_gdn` | HybridGdnExecutor | OvisOCR2, Surya 2.5 |
| `custom_vlm` | CustomVlmExecutor | HunyuanOCR-1.5 (XD-RoPE) |
| `moe_vlm` | MoeVlmExecutor | DeepSeek-OCR-2 (custom visual/query mask, 64 experts/top-6/2 shared) |

Qwen3-VL must not collapse into Qwen2.5-VL; hybrid GDN models are not KV-only
transformers; DeepSeek never gets a generic causal mask.

## Request flow (contract)

1. Preflight (host, every phase): validate_session_config,
   Session::preflight_request (validate_request for pages/budgets/sampling/
   policies, THEN the cancellation callback — validation precedes
   cancellation), validate_request_capabilities (mode/language/schema/
   guidance against the recipe's declared capabilities). Failures are
   invalid_argument, unsupported, or cancelled — before any device work.
2. Selection: the registry matches the exact artifact signature. A match is
   validated ONLY when the recipe is validated, pins revision + fingerprint +
   package format version, the artifact equals all three exactly, AND the
   canonical tensor signature matches exactly (all required/known names, no
   duplicates, no unexpected names). An outer HF model_type alone never
   selects a recipe (TeleOCR regression test). Anything else fails closed
   (unvalidated); the explicit allow_unsafe_experimental override returns an
   unvalidated candidate and never upgrades support state or permits
   inference by itself.
3. Load: Phase 1 implements the checked host-side .lcocr v1 reader/writer
   (ocr/package.*), the safetensors inspection reader, tokenizer/prompt/
   detokenizer, image/document input, preprocessing reference, and the
   .ocrfeat interchange (see FORMAT.md). Component-selective DEVICE weight
   upload and the checked memory preflight against MemoryLimits are
   implemented by the Phase 2 runtime (runtime/weights.*, memory_budget.*).
4. Recognize (synthetic executors implemented; real models fail closed):
   preprocess -> overview/layout -> crops -> vision batches (P7 ragged
   multi-crop graphs with exact-capacity chunking; Paddle/OpenDoc per-crop) ->
   independent batch-1 decode (decode-batch planning is planning-only) ->
   structured output, with cancellation
   polled at safe boundaries and source coordinates preserved through every
   transform (overview resize, layout crop, perspective warp, tile
   composition, cross-page merging).
5. Result: canonical text plus lines/blocks (type, reading order, source
   page/box/polygon, content), explicit has_confidence, ModelMetadata
   (identity, backend, RESOLVED CONCRETE precisions, actual policy state,
   effective caps, enabled optimizations, support state, and explicit
   router/fallback provenance), diagnostics (truncation/repair/offload/
   fallback are always visible), and performance counters. Metadata never
   carries the automatic precision sentinel and never implies a silent
   fallback.

Steps 1–2 are implemented (Phase 0) and the session then refuses with
unvalidated; Phase 1 adds the host artifact/input/output infrastructure of
step 3 and the full result JSON serializer for step 5 without changing the
refusal. There is no mock output path.

## Policies

- Precision: each component defaults to the automatic sentinel, which
  resolves (resolve_precision) to the recipe's validated default, else FP32
  (the portable correctness baseline). FP16/BF16/INT8/INT4 are explicit
  opt-ins that must be validated per recipe; automatic never silently
  changes model semantics. Result metadata only carries resolved concrete
  values.
- **Cache:** contiguous KV is the baseline; paged KV is a later, separately
  validated optimization; GDN models carry recurrent + conv ring state plus
  periodic full-attention KV.
- **Offload:** never implicit; `explicit_host` only, reported in metadata.
- Graph: CUDA-graph capture is an OPT-IN Phase 11 feature of the dense decode
  controller (DenseDecodeOptions.enable_cuda_graph) and the P7/Paddle region
  recognizers (SchedulerLimits.enable_cuda_graph forwarding), with explicit
  steps/updates/fallbacks counters; unsupported backends fall back to the
  ordinary stream path and capture/update failures fail closed. It covers
  DSL-only command lists; native shader dispatches must not be assumed to
  compose with CudaGraphExt. The GDN/MoE/Surya executors have no graph path.
- **Fast paths:** every fusion/native path needs a validated A/B arm and a
  kill switch; `automatic` may only pick parity-proven paths.
- **Sampling:** greedy by default; non-greedy requires explicit finite
  temperature > 0 and recipe validation.

## Capability negotiation

`ModelCapabilities` declares intended modes/languages/coordinate/confidence/
guidance support. `validate_request_capabilities` rejects:

- modes the recipe does not declare;
- spotting without declared source coordinates (declared intent; actual
  validated grounding is gated separately by SupportState);
- extract without schema/prompt where required;
- languages not EXACTLY declared — "multi" is a literal tag, never a
  wildcard (olmOCR-2 is English-only; a nonsense hint like "xx" is rejected
  even on multilingual recipes);
- region/color guidance the recipe does not declare.

These rules run in every phase, so later implementations inherit them
unchanged. Declared capability is *intent*, not a support claim: support
state (`planned` → `load_only` → `experimental` → `validated`) is governed by
the qualification gates in MODELS.md.

## Memory and time

`MemoryLimits` (device/host bytes) are preflighted before allocation;
`max_wall_time_ms` and the cancellation callback bound long documents.
Engine-side accounting is required on every backend; CUDA free-memory queries
are advisory only. Pixel budgets are separate for overview and crops, with
`max_pixels` as a compatibility ceiling that caps both (min wins).
## Phase 2 runtime and operators (implemented)

The portable correctness baseline is FP32 activations with FP32 accumulation.
Weight STORAGE encodings (FP16/BF16 packed, grouped INT4 sym/asym, grouped
INT8) are explicit and separate from compute precision; dequantization happens
in the operator kernels, never by reinterpreting one storage format as another.

Runtime contracts:

- `runtime/tensor`: logical shape/stride/dtype/layout metadata and non-owning
  `BufferViewDesc` descriptors. All element/byte/range math is checked 64-bit;
  conversion to the 32-bit kernel envelope goes through `checked_u32`. No host
  element values are stored. View footprints use the checked STRIDED extent
  (max indexed element + 1), so padded/strided views reserve their full range;
  `overlaps()` distinguishes legal in-place elementwise aliasing from illegal
  overlapping GEMM operands.
- `runtime/weights`: immutable per-component device weights loaded
  component-selectively from `PackageReader`. Resident bytes are the actual
  device buffer bytes. Host staging is releasable ONLY after upload completion
  (`complete()` after the stream synchronizes); `upload_component_sync` is the
  safe synchronous wrapper. All transfers are `luisa::span`.
- `runtime/workspace`: grow-only aligned arenas. `reserve()` between requests
  grows and retires the old buffer (kept alive until `retire_after_sync()`
  proves the stream drained); `reset()` rewinds the cursor AND bumps a reset
  epoch, so each allocation carries (generation, epoch) and a stale pre-reset
  region never `aliases()` a fresh post-reset region at the same offset.
  `WorkspacePlan` computes the full checked layout host-side for budget
  preflight before any allocation.
- `runtime/shader_cache`: `TypedShaderCache<KernelT>` compiles each device-free
  factory once per stable variant key, checks `compile_ok()`, and retains the
  shader for the owner lifetime. Shaders are held via `unique_ptr` so returned
  pointers stay valid across map rehashes (ankerl unordered_dense moves values).
  `ShaderKeyRegistry` records compile counts. Concrete type tags, no RTTI, no
  type-erased fake caching.
- `runtime/dispatch`: every factory has a matching `plan_*` helper. Plans use
  TOTAL-thread semantics with tail guards, checked uint32 conversion, row/query
  chunking, and flattened 2D grids when a 1D launch would exceed the portable
  65535-group cap. Work is never clamped or dropped; zero work yields an empty
  plan the caller skips. All ceil-division and sqrt searches are overflow-safe
  (no `a+b-1` wrap near UINT64_MAX), and every plan proves the flattened
  linear-id envelope `total_x*total_y <= UINT32_MAX` or is refused. GEMM/GEMV
  plans additionally fail closed when any logical buffer product (M*K, K*N,
  M*N) exceeds the uint32 index envelope.
- `runtime/dispatch_adapter.h`: the ONLY launch path. `dispatch_1d`/`dispatch_2d`
  (and the `OCR_DISPATCH_1D`/`OCR_DISPATCH_2D` stream macros) skip empty plans
  and dispatch flattened 1D plans through the `span<const uint3>` multi-dispatch
  form, so a flattened grid can never be silently truncated to `total_x`.
  Commands are moved into the stream/list; buffer handles are encoded at invoke
  construction and must outlive completion; the uint3 span is copied by the
  encoder. Every OCR 1D/2D dispatch site (tests included) routes through it.
- `runtime/memory_budget`: checked per-component weight/workspace/KV/shader/
  host-staging accounting with limit preflight that reports required vs
  available bytes before allocation. There is no invented universal
  device-memory query.

Operator contracts (naive, correct, portable; one thread per output element or
row, no shared memory, no warp requirement in the baseline):

- `ops_image`: fused color-convert + bilinear/area resize + normalization
  matching the Phase 1 `preprocess_image` convention (half-pixel centers, clamp
  edges, `(pixel/255-mean)/std`, CHW output); crop with out-of-range zero fill;
  letterbox; patchify. Verified differentially against Phase 1.
- `ops_gemm`: ragged M/N/K FP32 GEMM, compile-time epilogues, separate decode
  GEMV/small-M, packed FP16/BF16 B, grouped INT4 W4A32 (Phase 1 low-nibble/
  group/scale/zp semantics, groups 64/128, odd nibble counts), grouped INT8,
  SwiGLU/GeGLU gated activation. The `bias_residual` epilogue reads aux as
  `[bias N | residual M*N]` (independent bias and residual). Plans fail closed
  on uint32 index-envelope overflow; the epilogue-aware plan also bounds the
  aux element count (N + M*N) to uint32. `make_gemm_w4a16_kernel` is an INDEPENDENT
  variant that fails closed with `unsupported` (FP16 activation arithmetic is a
  later, separately gated arm); it is never an alias for W4A32.
- `ops_norm`/`ops_decode`: LayerNorm/RMSNorm (weighted + parameterless),
  QK-Norm shape, L2, row-sum, residual add; embedding lookup (tied-embedding
  compatible), multimodal scatter, greedy argmax (lowest-index ties),
  temperature softmax. Host validators reject invalid softmax temperature;
  the kernel writes defined zeros (never NaN) if validation is skipped.
- `ops_attention`: explicit `AttentionMaskDescriptor`
  (BIDIRECTIONAL/CAUSAL/WINDOWED/PREFIX_CAUSAL/CUSTOM_BLOCK_MASK). Reference
  scores+FP32 softmax+PV and an online-softmax path (running FP32 max/sum) are
  A/B tested separately. GQA/MHA, softcap, relative bias. DeepSeek semantics
  (visual bidirectional, queries see all visual + prior queries) are expressed
  as a caller-built block mask, not hardcoded. Separate contiguous-KV decode
  with per-batch lengths and absolute positions; decode clamps `kv_len` to
  capacity in-kernel (an invalid length can never read OOB) and honors
  CUSTOM_BLOCK_MASK (block_size 0 or an unknown kind masks everything, never
  silently bidirectional). The kernel takes the mask's exact byte count and
  disallows any out-of-range mask index WITHOUT reading, so a large position
  can never read OOB; a position-aware host validator also rejects masks that
  do not cover the maximum position's query block. Host validators reject
  invalid lengths/masks before launch. No paged cache yet.
- `ops_rope`: 1D, 2D axial, M-RoPE, interleaved MRoPE, XD-RoPE as SEPARATE
  named variants with position buffers and axis-mapping data; sinusoidal table.
  One variant is never substituted for another. Long-context (~1e6 position)
  tests compare against an explicit FP32-angle reference with an angle-ulp-aware
  tolerance (a double-angle gate is impossible at that magnitude).
- `ops_conv`: Conv2D general kernel dims/stride(incl. asymmetric)/pad/dilation/
  groups/depthwise; Conv1D + causal depthwise; BatchNorm inference; avg/
  adaptive/global pool; concat/split/shuffle. Host validators reject invalid
  groups/channel divisibility; the kernels write defined zeros (never silently
  drop channels) if validation is skipped. CPU contours/postprocessing stay
  in `output/`, not here.
- `ops_vision`: pixel-unshuffle, spatial merge/token packing, absolute/learned
  position addition, DeepStack injection with caller-selected layer ranges as
  data. Patch projection is `ops_gemm` over patchified tokens.
- `ops/cpu_reference`: independent naive double-accumulating host oracles for
  every device op; tests compare two separate code paths with explicit
  tolerances.

Portable numerical notes: the HLSL target (dx / vk-HLSL route) has no `erf`
intrinsic, so GELU-erf uses the Abramowitz & Stegun 7.1.26 rational
approximation (max abs error 1.5e-7) composed of abs/exp/select, identical on
every backend. FP16/BF16 storage decode is exact bit extension / integer
upcast; accumulation stays FP32.

Phase 2 verification: host unit suites (`test_ocr_memory_budget`,
`test_ocr_runtime`) and device integration suites
(`test_ocr_ops_{image,gemm,norm,rope,attention,conv,vision}`,
`test_ocr_runtime_device`) pass on vk and cuda with synthetic CPU-oracle cases
(ragged sizes, tails, aliasing, NaN/degenerate), including >65535-group
flattened-dispatch full-coverage regressions, shader-cache rehash-stability,
uint32/uint64 envelope-overflow refusal, strided-view footprints, workspace
reset-epoch alias tracking, attention invalid-length clamping and fail-safe
masks, and large-position FP32-angle RoPE. The fallback backend is not enabled
in the reference configuration (requires LLVM+Embree). No model is validated;
all 15 requirement models remain planned.
## Phase 3 dense decoder core (implemented; synthetic parity only)
The dense autoregressive language path is implemented correctness-first in
FP32, reusing the Phase 2 operators. It is model-agnostic: geometry and tensor
names come from a shared contract, and the SAME executor serves the synthetic
test decoder and (later) real Qwen2.5-VL-derived decoders. No vision tower,
MTP/DFlash, paged KV, or batching beyond batch 1 is present. (Phase 11 added
OPT-IN single-sequence CUDA graph capture/update and an optional fused
residual+RMSNorm to this path; both default off — see the Phase 11 section.)
- `models/qwen2_5_vl`: SHARED dense decoder components only — `DenseDecoderConfig`
  geometry (validated: GQA divisibility, heads*head_dim==hidden, even head_dim,
  finite eps/theta) with an EXPLICIT `rope_pair_layout` field (`adjacent` pairs
  (2d,2d+1) vs `rotate_half` pairs (d, d+head_dim/2));
  `DenseDecoderTensorNames` (prefix `model.` or `language_model.`; packed
  `[K,N]` linear weights, QKV biases required; the standalone lm_head is
  `[hidden, vocab]` packed kn, NOT `[vocab, hidden]`),
  `dense_decoder_expected_tensors` + `validate_dense_decoder_tensors` (exact
  name/shape/layout, fp32-only baseline) + `validate_decoder_weight_desc`
  (per-weight shape+layout+element-count gate the executor applies BEFORE
  binding any view), and `parse_qwen2_5_text_config` (strict; selects
  `rotate_half` — the Qwen2.5 HF text convention — and fails closed on
  Qwen3/QK-Norm/MoE/GDN markers so a Qwen3-VL config is never silently accepted
  as Qwen2.5; integer-valued rope_theta/rms_norm_eps parse correctly). A merely
  present but DISABLED sliding window (`use_sliding_window=false` or a null
  window) is accepted because the ordinary dense path is exact; an ENABLED
  sliding window remains `unsupported`. This is NOT a claimed full model and
  makes no statement about olmOCR/TeleOCR.
- `runtime/decode_cache`: DISTINCT interfaces. `ContiguousKvCache` (batch-1
  correctness) owns per-layer K/V buffers shaped
  `[layers, capacity, kv_heads, head_dim]` matching the Phase 2 decode-attention
  layout, plus `DecodeState` device-visible `kv_len/positions/step/state`
  buffers that the attention kernel reads in place (no per-step host re-upload).
  Capacity is checked 64-bit math from the actual geometry
  (`plan_kv_cache_bytes`) and preflighted against a `MemoryBudget` BEFORE
  allocation; overflow/OOM return clear Status. BOTH the preflight and the
  recorded requirement include the DecodeState metadata bytes.
  `DecodeState::create` zero-initializes the metadata synchronously (drains
  the stream) so an unbound sequence can never be attended; `record` copies
  each snapshot into heap-owned staging retained through command completion
  (safe for stack-local spans and multiple records into one CommandList).
  `bind_sequence`/`append_tokens` bound growth; `release()` frees a completed
  sequence's HOST slot immediately (no device commands), and
  `record_finished()` is the real device-side completion API (state=finished,
  kv_len=0) the controller records at the completion boundary, so a released
  slot is never attended stale on reuse. `reset()` uses the same
  completion-retained staging (no dangling stack spans).
  `PagedKvCache` is a separate optional interface that fails closed
  (`unsupported`) in Phase 3 — the contiguous path is NOT built on top of it.
- `runtime/executor`: the command-recording contract. `record_prefill` and
  `record_decode_step` append to a caller-owned `CommandList` and never
  synchronize internally (host decode scheduling decides when to drain).
  Ownership: weights immutable/shared (borrowed `WeightStore`), workspaces
  per-request (`DeviceWorkspace`), shaders cached for the executor lifetime,
  all transfers `luisa::span`, no exceptions/RTTI.
- `runtime/dense_vlm_executor`: embedding lookup + optional multimodal row
  replacement; dense blocks (input RMSNorm, QKV projections with bias, 1D RoPE
  on Q/K using the config-selected pair layout — `make_rope_1d_kernel` for
  `adjacent`, `make_rope_rotate_half_kernel` for `rotate_half`, never silently
  substituted — contiguous-KV append, GQA causal attention — prefill reads the
  growing cache prefix with an end-aligned causal diagonal, decode reads the
  committed prefix via device metadata — o-proj+residual, post RMSNorm, SwiGLU
  MLP, down+residual); final RMSNorm + LM head (standalone, or tied via a
  one-time transpose whose device buffer + host staging are preflighted and
  included in `weight_bytes`). `create()` validates EVERY weight's TensorDesc
  shape+layout+element_count against the exact expected inventory before
  binding (undersized/wrong-layout/transposed tensors and unexpected component
  tensors are refused before any dispatch), and prefill rejects every
  out-of-vocabulary prompt token before staging. Prefill is chunked
  (`prefill_chunk_rows`, TDR guard) and produces either last-row logits or
  every-row logits (layered goldens). All id/position/axis uploads use
  per-record heap-owned staging retained through command completion, so
  multiple prefill/decode records into ONE CommandList are safe. The
  `DenseDecodeController` drives prefill + single-step decode with greedy or
  validated deterministic sampling, EOS ids (decoder config + options +
  tokenizer, deduplicated) + stop strings through the Phase 1 incremental
  detokenizer (UTF-8 preserved, stop text never emitted), bounded output
  tokens and cache capacity, early-EOS-at-first-token, and a truncation
  `Diagnostic` for later Session integration. An RAII guard releases the
  cache slot on EVERY post-bind exit (success, sampling/detokenizer/decode
  failure, cancellation, timeout) and records the device-side finished state.
  `DenseDecodeOptions` carries a `cancel_callback` and `max_wall_time_ms`
  polled at each per-step synchronization boundary, returning `cancelled` or
  the explicit `timeout` StatusCode (cache always released).
- `ops/ops_decode` additions: a contiguous-KV append kernel (`kv_append`,
  checked `validate_kv_append`) and HOST deterministic sampling building blocks
  (`DecodeSamplingConfig` + `validate_decode_sampling` + `DeterministicSampler`:
  explicit seed, temperature/top-k/top-p validation, HF-style repetition
  penalty, no-repeat-ngram ban, fixed-seed SplitMix64 categorical draw). The
  sampler is host-side by design and is the SAME code path for the CPU oracle
  and the device executor, so determinism is by construction. `-inf` is an
  allowed masked/banned entry; NaN/+inf are invalid. If no finite selectable
  logit remains (all NaN/+inf/banned) the sampler returns `invalid_argument`
  — never a silent token 0 or a banned token — and the RNG advances exactly
  once per SUCCESSFUL non-greedy draw (a failure does not advance it).
Phase 3 verification: device integration suites `test_ocr_decode_{prefill,
cache,sampling,stop}` pass on vk AND cuda (xmake and CMake builds) against an
independent double-accumulating CPU oracle composed from the Phase 2 host
references, plus the host-only `test_ocr_decoder_config` suite. Covered:
embedding/multimodal replacement, first-layer and per-layer KV content,
full-prefill logits, first-token logits, EVERY decode-step KV+logit,
chunked-prefill parity, tied/standalone LM head, greedy argmax +
temperature-softmax device parity, sampling determinism (same seed =>
identical tokens), top-k/top-p/repetition/no-repeat-ngram, invalid-parameter
refusal, all-NaN/all--inf/all-banned refusal with no RNG advance on failure,
capacity math + budget preflight + overflow/OOM refusal, cache
growth/release/reuse, workspace bounds, paged fail-closed, early EOS,
multi-token output, repeat-run exact greedy tokens, stop strings across
tokens/UTF-8, and max-token/cache-capacity truncation. Review-fix regressions:
undersized q_proj / transposed / wrong-layout lm_head refused at create()
before dispatch; out-of-vocabulary prompt tokens refused before staging;
reset -> commit -> stack-churn -> download reads zero (no UAF) and reset/reuse;
two decode steps in ONE CommandList match the oracle (per-record staging);
config-only EOS distinct from the tokenizer EOS; failure/cancel/timeout always
release the cache for reuse; prefill- and decode-boundary cancellation and a
deterministic decode-boundary timeout; tied-head device+host accounting and
DecodeState-metadata-inclusive cache budgeting; integer rope_theta parsing;
rotate-half prefill/KV parity AND direct rotate-half rope-op parity, with
rotate-half proven distinct from adjacent. Synthetic geometry: vocab 32 /
hidden 16 / 2 layers / 4 Q heads / 2 KV heads / head dim 4 / intermediate 32
(GQA group 2), FP32; the parity decoder uses the adjacent convention by
default and a rotate-half variant exercises the Qwen2.5-style layout.
Tolerances: 3e-4 abs + 3e-4 rel for decoder logits/KV; 2e-4 for standalone
softmax/argmax/rope. The stop tests use a compact synthetic Unigram tokenizer
(vocab 10, raw UTF-8 pieces) and a scripted dominant-head decoder so the
greedy chain is exact. These are SYNTHETIC parity gates only: no official
checkpoint/tokenizer/processor has been inspected, no real model is claimed
to work, and Session still fails closed with unvalidated.

## Phase 4 Qwen3-VL vertical slice + FireRed-OCR + structured outputs (implemented; synthetic parity only)

- `models/qwen3_vl`: the DISTINCT Qwen3-VL graph contract. `Qwen3VlConfig`
  (text + vision + DeepStack + interleaved-MRoPE sections), strict config
  parsing with an explicit Instruct/Thinking identity boundary (Thinking
  fails closed; Qwen2.5 markers refused), the exact expected tensor inventory
  (QK-Norm weights required, QKV biases absent), native/dynamic image packing
  (`qwen3_vl_pack_image`) with source transforms, and the host interleaved
  M-RoPE position semantics (`qwen3_vl_build_position_ids`, explicit
  temporal/height/width axes). Host-only; never includes the ops enums.
- `runtime/qwen3_vl_executor`: records the vision graph (patch embed ->
  bidirectional QK-Norm vision blocks with 2D axial RoPE + GELU MLP -> spatial
  merger -> DeepStack mergers) from the Phase 2 operators, and drives the
  shared Phase 3 `DenseVlmExecutor` with the additive Qwen3 contract switches
  (`qk_norm=true`, `qkv_bias=false`, `rope_axes=3`, `deepstack_layers`). The
  Phase 3 Qwen2.5-style path is unchanged (the switches default to it).
  `runtime/qwen3_vl_preprocess` is a host-only bridge to the Phase 1
  preprocess reference, isolated so api.h/model_recipe.h (host
  AttentionMaskKind) never meets the ops enum in a device TU.
- `models/fire_red_ocr`: reuses the SAME Qwen3-VL graph only through its own
  tensor map (`language_model.` / `model.visual.` prefixes), prompt template
  (`fire_red_ocr`), and output parser (`fire_red_markdown`). Real artifacts
  fail closed unconditionally; a training/GRPO-node guard rejects forbidden
  component roles. No GRPO/training operators exist anywhere.
- `output/markdown`, `output/table`, `output/extract`: strict structured
  outputs. Markdown block parsing/composition with reading order + source
  spans; HTML and OTSL table parsing/validation (malformed input is an error,
  never a text fallback); LaTeX delimiter/content sanity; JSON extraction
  against a deliberately supported schema subset (unknown features fail
  closed). JSON repair is OFF by default; enabled repair produces diagnostics.

Phase 4 verification: device integration suites test_ocr_model_qwen3_vl
(9 tests, 264 asserts) and test_ocr_model_fire_red (2 tests, 26 asserts) on
vk and cuda; host suites test_ocr_output_{table,markdown,extract} and
test_ocr_model_contract. The review-fix pass adds permanent regressions:
wrong 1-axis prefill/decode entry points on a 3-axis executor are refused
(M1); forged/non-divisible packing metadata is refused before any GPU work
with checked patch/merged arena bounds (M2); canonical OTSL 2x2/3x3 spans
parse and convert losslessly with no double span increments, U-under-L and
out-of-table spans rejected (M3); quoted-attribute substring, heading
closing-#, token-only OTSL detection, display-math leading-content close,
invalid numeric character references, post-rounding pixel-budget recheck,
strictly-increasing DeepStack vision layers, exact model_type identity +
attention_bias/unknown-field/sliding-window config strictness, and the exact
FireRed prefix pair are all covered (S1-S9). The unused vision_chunk_patches
field was removed (S10; max_vision_patches remains the real budget) and
download_vision_block0 was renamed download_vision_final (S11).
The synthetic Qwen3-VL geometry (vocab 32 / hidden 32 / 3 decoder layers /
4 Q + 2 KV heads / head dim 8; vision 16 hidden / 2 heads / 2 layers / patch 2
/ merge 2; DeepStack at decoder layers {0,1}) is a DECLARED contract, not an
official checkpoint. Layered golden points (patches, first vision block,
merger, DeepStack projections/injections at the exact layers, first decoder
layer, full-prefill logits, decode KV/logits, deterministic greedy tokens)
match an independent double-accumulating CPU oracle. Tolerances: 5e-4 abs +
5e-4 rel for linear/norm stages; 3e-3 abs + 3e-3 rel for GELU-erf stages;
greedy token ids must match exactly and repeat-run identically. Blank/
repeated/adversarial cases exercise the implemented synthetic graph and
strict parsers honestly — they do NOT establish hallucination safety for any
real VLM, and no VLM confidence is produced. These are SYNTHETIC parity gates
only; every real model remains planned and Session fails closed with
unvalidated. The unmet official gates are recorded in MODELS.md.
## Phase 6 layout/region infrastructure + PaddleOCR-VL + OpenDoc (implemented; synthetic parity only)

Phase 6 adds the layout-routed execution path on top of the Phase 2 operators,
the Phase 3 dense decoder, and the Phase 5 CNN recognizer. It is deliberately
split into device-safe graph/orchestration TUs and host-only composition TUs so
the host AttentionMaskKind (model_recipe.h/api.h) never meets the ops enum
(ops_attention.h) in one translation unit.

Device-safe pieces (no api.h/model_recipe.h):
- models/pp_doc_layout.h/.cpp: PP-DocLayoutV2/V3 contracts (SEPARATE per-version
  configs/tensor maps) + the CPU decode (score threshold, clamped exp box decode,
  NMS AND class-aware NMS, region scoring, stable reading order, source
  transforms). The decode is a host algorithm over device-safe plain types.
- runtime/scheduler.h/.cpp: RegionScheduler (token-budgeted vision/prefill/decode
  batch packing, restore to page/reading order), LayoutCache (bounded
  detector-output cache keyed by an explicit content/geometry FNV-1a hash), and
  the plain LayoutRunResult/LayoutDocument types.
- runtime/layout_vlm_executor.h/.cpp: PpDocLayoutExecutor (device conv graph ->
  head maps), the RegionRecognizer interface, PaddleOcrVlRecognizer (NaViT vision
  + adaptive average-pool token reduction + LayerNorm/MLP connector + the shared
  DenseVlmExecutor ERNIE decoder), OpenDocRecognizer (the shared CnnExecutor
  recognizer graph in recognizer-only mode; UniRec is CNN+CTC, NOT a dense VLM),
  and LayoutVlmExecutor (full overview->layout->crop->region->composition
  orchestration with cancellation/time/memory checks, immediate sequence/
  workspace release, and a documented cpu_reference layout option that consumes
  head maps from the independent host oracle — never a silent model fallback).

Host-only pieces (include api.h; never included by a device TU):
- models/paddle_ocr_vl_result.*, models/open_doc_result.*: type-specific adapters
  (text/formula/table) + strict Markdown/JSON/plain/spotting composition.
  Malformed table/formula/JSON output fails closed (no plain-text fallback).
- runtime/layout_result.*: the shared LayoutDocument -> public RecognizeResult
  bridge (lines/blocks/source page/box/polygon/type/content/reading order).
  Generative content carries NO confidence; only validated detector/CTC scores
  are carried, explicitly labeled internal/test via a diagnostic.

Additive Phase 5 change: CnnExecutor gained recognize_line() (per-region CNN+CTC
entry) and create_recognizer_only() (no detector weights/workspace) so OpenDoc's
UniRec stage reuses the real recognizer graph without a fake detector. The full
detector->recognizer recognize() path is unchanged and fails closed in
recognizer-only mode.

PaddleOCR-VL is a DISTINCT graph from Qwen3-VL: NaViT vision uses QKV bias and
NO QK-Norm/DeepStack; token reduction is an adaptive average pool over the patch
grid (not a spatial merger); the ERNIE decoder uses the Phase 3 contract
(rotate-half 1D RoPE, QKV bias, no QK-Norm). OpenDoc is a TWO-STAGE recognizer
(PP-DocLayoutV2 + UniRec CNN+CTC) and is never forced into the dense VLM/KV path.

  Phase 6 verification: host unit suites test_ocr_scheduler, test_ocr_layout_decode,
  test_ocr_layout_compose; device integration suites test_ocr_layout,
  test_ocr_model_paddle_ocr_vl, test_ocr_model_open_doc, test_ocr_region_scheduler
  on cuda (vk blocked upstream, see below). Each compares against an independent
  double-accumulating CPU oracle (layout head maps + decoded boxes/scores/classes;
  NaViT patches + connector features + ERNIE greedy tokens; UniRec CTC ids/text;
  region batching vs batch-1 ORDERED equivalence; repeat-run determinism).
  Tolerances: layout conv chain 3e-3 abs + 3e-3 rel, decoded boxes 0.75 source px /
  scores 5e-3; NaViT patches 5e-4, connector 3e-3 (GELU-erf); greedy tokens / CTC
  ids match exactly. The host preprocessing oracle mirrors the declared FLOAT
  stretch-resize convention so the conv-chain parity isolates the graph.

  Review fix pass (M1/S1-S8): the layout source transform is source/INPUT (the
  decode folds the backbone stride into the box exactly once; a source/map scale
  double-counted it) — pinned by an absolute-geometry host regression at a known
  cell and a full-pipeline crop-content test proving bright bars are hit. A Paddle
  full-pipeline orchestrator test mirrors OpenDoc (layout->crop->connector/decode->
  ordered output). max_output_tokens_per_page is enforced during accumulation with
  a visible output_truncated diagnostic; max_output_tokens_per_region is the single
  per-region cap forwarded to the recognizer. Returned pages are sorted by source
  page_index (stable); region_id is document-global across pages. UniRec resolves
  its DISTINCT UniRecTensorNames through a CnnExecutor recognizer name-map override
  (no silent PP-OCRv6 alias), with wrong/missing-tensor refusal tests. The per-region
  KV create receives a MemoryBudget; peak_device_bytes = weights + workspace + KV
  peak. Phase 11 P7 recognizers execute ragged multi-crop vision graphs with
  exact-capacity chunking; Paddle/OpenDoc remain per-crop. Text decode is still
  independent batch-1, and decode-batch planning has no batched executor yet.

  Backend note (S8, upstream blocker): cuda passes all Phase 6 suites in xmake and
  CMake. vk is blocked by an upstream post-merge regression: vkCreateDevice chains
  VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT (sType 1000657000), rejected
  by the installed validation layer 1.4.321.1 (fatal); the baseline test_ocr_ops_gemm
  fails identically, confirming it is not Phase 6. The merged vk backend also fails
  CMake's non-unity compile (missing opacity_micromap.h / command_buffer_sync.h
  includes); xmake unity masks it. Phase 6 vk parity was green pre-merge; no backend
  edits are made here.

  These are SYNTHETIC parity gates only; every real model remains planned,
  PaddleOCR-VL is NOT a validated complex-document default, and Session fails closed
  with unvalidated. The unmet official gates (layout mAP, reading order,
  table/formula/Markdown/JSON, determinism, memory, RTX 4060) are recorded in
  MODELS.md.

Phase 8 remaining dense ViT models (implemented; synthetic parity only)

Phase 8 adds dots.mocr, GOT-OCR2.0, and olmOCR-2 on the shared dense executor
and the Phase 7 slice, with ADDITIVE engine support (all prior suites green):
- DenseLinearStorage (models/qwen2_5_vl.h) + DenseExecutorConfig.linear_storage:
  an explicit per-linear W4A32 path. Packed grouped-INT4 B operands (Phase 1
  low-nibble/group/scale/zp semantics) are read by the Phase 2 packed GEMM/GEMV
  kernels with FP32 activations and FP32 accumulation; embeddings/norms/QKV
  biases stay FP32; a quantized recipe requires a standalone packed lm_head
  (tied embeddings refused). create() validates the exact storage kind + group
  size per weight and refuses any FP32/FP16/INT8-as-INT4 reinterpretation.
- P7SliceConfig.linear_storage/linear_group_size forward the policy to the
  embedded dense executor; text_qwen2_5_contract selects the Qwen2.5-VL text
  identity (QKV bias, NO QK-Norm, rotate-half M-RoPE via the distinct named
  make_rope_mrope_rotate_half_kernel) instead of the Qwen3 QK-Norm contract.
- P7VisionConfig.rel_pos_window: decomposed ViTDet/SAM relative-position bias.
  The immutable per-layer tables are downloaded ONCE at create (stream drained
  there; no borrowed stream retained); record_vision expands them host-side
  per image grid into completion-retained staging and the attention kernel
  adds them to the scores. Distinct from RoPE/absolute positions; mutually
  exclusive with vision RoPE in validation.
- P7ConnectorConfig.rms_norm: the Qwen2.5-VL merger RMSNorm (weight only);
  the LayerNorm default and inventory are unchanged for prior models.
- models/dots_mocr: ViT + spatial-merge merger + dense GQA decoder contract;
  the FULL 28-layer/12-query/2-KV geometry is declared and validated explicitly
  (dots_mocr_declared_decoder_config) while the executing slice is scaled;
  dots_mocr_preflight_decoder reports checked weight/workspace/KV bytes before
  allocation; W4A32 arm + strict signature/real-artifact gates.
- models/got_ocr2: ViTDet/SAM window/global + relative bias, linear connector,
  Qwen decoder; EXACT prompt contracts (plain/formatted/region/color) with
  round-trip parsing, invalid-guidance refusal, normalized->pixel region
  transforms, and reading-order multi-crop composition (strict Markdown in
  formatted mode; malformed fails closed).
- models/olm_ocr2: complete Qwen2.5-VL encoder/RMSNorm-merger/decoder contract;
  English-only enforcement (exact language match, "multi" refused); the
  declared 8 GB path REQUIRES INT4; olm_ocr2_preflight reports the full
  vision+decoder+workspace+KV requirement and fails BEFORE allocation;
  explicit_host offload fails closed unsupported (no performance claim).
Phase 8 verification (post-review): cuda, xmake AND CMake, debug. New suites:
dots 6/113, GOT 8/143, olm 5/109 (tests/asserts). 21 affected prior device
suites + 26 host suites green in both build systems (CTest unit_ocr 26/26).
Review fixes S1-S4 landed with tests (prefix-restricted quantizable linears +
wired package-load gates; owned tied-head bytes in measured residency with
preflight==measured assertions; defensive packed-buffer count gates; olm
languages non-array refusal). GOT adds a memory preflight, page_index in
multi-crop composition, and locale-independent d.dddd region parsing. vk
remains upstream blocked (backend validation callback abort; no backend
edits). Measured synthetic memory and the W4A32 quality/preflight numbers are
recorded in MODELS.md. These are SYNTHETIC parity gates only; all three models
remain planned, real artifacts fail closed unconditionally, and Session fails
closed with unvalidated. No official accuracy or RTX 4060 claim is made.

## Phase 9 hybrid GDN subsystem + OvisOCR2 + Surya (implemented; synthetic parity only)

Execution-path family: hybrid_gdn (NOT a KV-only transformer). New modules:
- ops/ops_gdn: device-free DSL factories + INDEPENDENT double-accumulation host
  oracles for the declared GDN contract: Q/K/V/z/a/b projections; causal
  depthwise Conv1D + SiLU with an in-place ring state (last K-1 PRE-activation
  taps, slot = absolute position mod (K-1), overwritten only after the output
  reading it was computed); required per-head L2/QK normalization; beta=sigmoid,
  softplus(threshold 20) decay g = -exp(A_log)*softplus(a+dt_bias); the delta
  rule S <- alpha*S, S += beta*k*(v - S^T k)^T, o = S^T q on a [d,d] per-head
  state (S[k,v] layout); gated output RMSNorm * silu(z); o-proj + residual.
  Straight recurrent prefill/decode never materializes a sequence x sequence
  matrix. The chunked prefill is the WY/UT-transform chunk algebra (L matrix,
  forward-substituted w/u, pseudo values ue, carried-state transition,
  inter+intra outputs) — correctness-first SERIAL substitution inside a chunk,
  explicitly NOT chunk-parallel and NOT a sequential rename; it is a distinct
  arithmetic path verified against BOTH the recurrent oracle and a full-L
  host oracle at every chunk boundary. Also zero-fill + elementwise sigmoid
  utilities reused by the cache/detector.
- runtime/hybrid_gdn_config: HOST-SAFE geometry/tensor-name contract
  (GdnLayerConfig, HybridCacheConfig with checked KV/GDN/Conv/metadata capacity
  math from actual dimensions, HybridGdnExecutorConfig with the EXPLICIT
  per-layer HybridLayerKind schedule, HybridGdnTensorNames, expected-tensor
  inventory + fp32 package gate). No api.h/ops_attention.h, so host model TUs
  and device TUs share one contract.
- runtime/hybrid_gdn_executor: HybridGdnCache (contiguous KV for FULL layers
  only + per-layer/head/request recurrent [H,d,d] + Conv ring [3*H*d,K-1] +
  Phase 3 DecodeState metadata; synchronous zero-init; budget preflight before
  allocation; bind/append/release/record_finished/reset with no cross-request
  leaks; paged KV fails closed). HybridGdnExecutor: separate prefill/decode
  command-recording loops mixing GDN and full GQA layers, dense embedding +
  mm-scatter, RMSNorm/SwiGLU/LM head (tied transpose owned buffer), chunked
  prefill entry (chunk_size bounded by the workspace row capacity; oversized
  chunks fail closed), per-record heap-owned upload staging. HybridDecodeController
  reuses the Phase 3 DeterministicSampler + Phase 1 IncrementalDetokenizer
  (greedy/seeded sampling, EOS/stop strings, bounded output, wall-time/cancel,
  release guard recording device finished state on every exit). HybridVlmGraph
  pairs P7VlmExecutor::create_vision_only (ADDITIVE: identical vision path, no
  embedded dense text executor) with the hybrid decoder; resets the cache per
  run; optional borrowed tokenizer for end-to-end text.
- models/ovis_ocr2: dynamic-preprocessing ViT + spatial-merge merger + the
  Qwen3.5-0.8B-style hybrid decoder contract. The FULL 24-layer / 18-GDN /
  6-full schedule (full at {3,7,11,15,19,23}) is validated EXPLICITLY; the
  executing slice is scaled (6 layers = 4 GDN + 2 full) and must not pass the
  full gate. Strict config parse (explicit layer_kinds required; MoE markers
  refused), exact signature gate, unconditional real-artifact refusal, checked
  memory preflight, plain/markdown composition (spotting/extract fail closed).
- models/surya + runtime/surya_executor: EfficientViT-Segformer-style line
  detector (scaled 4-stage CNN + DB head + sigmoid; shared conv/act/sigmoid
  ops + output/db_postprocess) -> source-coordinate line crops -> the same
  vision merger + hybrid decoder per line -> stable top-to-bottom reading
  order. The FULL 24/18/6 schedule contract is validated explicitly. Confidence
  is ONLY the detector mean box probability (labeled); generative text carries
  none. Plain/spotting/markdown via the shared strict composer; extract fails
  closed; no fabricated table HTML. Blank pages produce zero lines.

Phase 9 review fixes (S1/S2):
- S1: HybridGdnExecutor::record_state_reset zeroes THIS request's GDN recurrent
  state AND Conv ring on EVERY from-scratch prefill (after bind_sequence, before
  any chunk work, same command list ahead of all layer dispatches) via the
  compiled zero-fill shader. Required because the delta rule accumulates onto the
  state and the Conv reads prior taps. record_finished() is documented
  METADATA-ONLY; release() frees only the host slot; reset() still clears all
  sequences, and bypassing prefill requires reset() first. Full-attention KV is
  never zeroed (bounded by kv_len / the causal prefix). HybridVlmGraph's per-run
  reset() is belt-and-braces, not the safety mechanism.
- S2: SuryaConfig::tokenizer is a BORROWED const Tokenizer* (caller-owned; must
  outlive create() and every recognize(); never copied/freed/retained). create()
  refuses null or an undersized vocabulary and recognize() re-checks it, so a
  text-capable run never silently emits empty content; it is plumbed into
  HybridVlmGraph::run (used only inside generate()). Line EOS comes from the
  declared decoder contract; tokens-without-text (no EOS/stop) is an
  internal_error, while a token-less line and a blank PAGE stay valid empty.

Phase 9 verification (post-review): cuda, xmake AND CMake, debug. New suites
(tests/asserts, identical in both build systems): gdn_conv 3/31,
gdn_recurrent 4/26, gdn_chunked 3/25, gdn_cache 12/237, model_ovis_ocr2 8/49,
model_surya 10/61 (40 tests / 429 asserts total). All 26 host suites (2507
asserts / 240 tests; CTest unit_ocr 26/26) and 23 affected prior device suites
(4537 asserts / 167 tests) green in both systems; example_ocr builds in both.
New regressions: gdn_hybrid_reused_cache_no_leak (NONZERO GDN weights, two
generate() calls on ONE cache with no reset between them; second run's logits /
tokens / recurrent + Conv state BIT-EXACT vs fresh caches) and
surya_line_text_detokenized + surya_tokenizer_required (real synthetic Unigram
tokenizer; non-empty line text, page text, composed block content; null /
undersized tokenizer fails closed). Both were negative-controlled (fix disabled
=> the regression fails), then restored; no probe code remains. Declared tolerances and the exact synthetic
equations/architectures are recorded in MODELS.md. vk remains upstream blocked
(no backend edits). These are SYNTHETIC parity gates only: OvisOCR2 and Surya
remain planned, real artifacts fail closed unconditionally, and Session fails
closed with unvalidated. No official accuracy or RTX 4060 claim is made; the
chunked path makes no performance claim (serial substitution, not parallel).

## Phase 10 DeepSeek-OCR-2 MoE vertical slice (implemented; synthetic parity only)

Phase 10 adds the MoE execution family. It is a SEPARATE engine, never a dense
ViT+LLM alias and never a generic causal transformer:

- `ocr/ops/ops_moe.*` — device-free MoE factories + independent double-accumulating
  host oracles: FP32 router softmax, deterministic top-k (lowest-index tie-break,
  renormalized weights), histogram/counts, exclusive-prefix offsets,
  token-to-expert permutation (stable within an expert by ascending slot),
  gather, and weighted scatter restore + two shared dense experts.
- `ocr/runtime/moe_vlm_executor.*` — the complete synthetic graph: SAM-style local
  perception -> BOTH convolutional downsamplers -> Qwen2-style visual LM encoder
  with causal-flow queries and the EXACT custom visual/query block mask ->
  connector -> DeepSeek MoE decoder (contiguous KV, ordinary target decoding).
- `ocr/models/deepseek_ocr2.*` — host-safe identity/tensor/mask/output contract,
  exact 64/top-6/two-shared expert gate, signature gate, real-artifact fail-closed
  gate, EXACT shared workspace plan + memory preflight, grounded-Markdown adapter.

Custom visual/query mask (mandatory, plan §3/§7.3): built host-side as DATA at
`block_size 1`, never a generic causal mask. Visual rows are bidirectional among
visual rows and NEVER see queries; query row q sees ALL visual rows plus query
rows <= q (including itself) and never later queries. The text decoder attention
IS causal (autoregressive) — the custom mask lives only in the visual LM encoder.

MoE reference progression (plan §7.3): the router/permutation run on device, the
host reads back the per-layer counts/offsets, and then a loop over ACTIVE experts
records per-expert gate/up/SwiGLU/down GEMMs on that expert's gathered rows.
Inactive experts are skipped entirely. There is NO grouped persistent GEMM
framework and NO native graph for the MoE executor (Phase 11 added CUDA graph
capture only to the dense/P7/Paddle decode paths, not here).

Phase 10 correctness fixes found by the new suites (both were real defects):
- The GEMV/small-M path implements only none/bias/residual epilogues. A gated
  epilogue (`bias_gelu_erf`, used by the connector at M == num_queries == 4) MUST
  take the GEMM path; routing it to GEMV silently dropped the bias AND the GELU
  activation. `record_linear` now gates GEMV on the supported epilogue set.
- The single-thread offsets kernel used `set_block_size(1,1,1)`, which CUDA
  rejects (block size must be a multiple of 32). It now uses 32 with the
  `ocr_linear_id() == 0u` guard.
- The encoder mask and all uploaded host arrays (pixels, positions, axis map,
  per-layer masks) live in per-record heap staging retained through command
  completion, so a later `record_vision` can never dangle a borrowed upload.

Memory is EXACT, not estimated: `deepseek_ocr2_workspace_plan` is the single
shared arena plan used by BOTH `reserve_workspace` and the preflight, so the
preflighted workspace bytes EQUAL the measured arena capacity. The device suite
asserts measured == planned for vision weights, decoder weights, workspace, and
KV, and that a tiny explicit budget refuses BEFORE allocation with
required-vs-available details.

Phase 10 independent-review fixes (all must-fix + should-fix, each with a
permanent test; no Phase 11 / backend work):
- M1 parse: `num_attention_heads == 0` (and a non-dividing hidden_size) is refused
  BEFORE deriving the default head_dim, so the config parser can no longer divide
  by zero — even when an explicit head_dim is supplied.
- M2 record_vision: `patch_count` must equal the CHECKED `grid_h*grid_w`, and
  `rows` is derived from that product; a forged/undersized patch_count or a zero
  grid dimension is refused before any upload.
- M3 record_vision: zero source width/height, an empty pixel buffer, and a zero
  resized edge are refused before the preprocess clamp/scale math.
- M4 downsamplers: `DeepSeekConvDownsampler::validate` proves the conv can never
  ENLARGE the grid (stride 1 requires 2*pad+1 <= kernel; stride >= 2 is checked at
  the smallest legal input, where out-in is maximal), so the
  `max_vision_patches + num_queries` workspace bound stays an actual upper bound.
  record_vision additionally checks the concrete grid per stage.
- M5/M6 oracles: `ref_moe_gather` validates the x extent and every perm_token
  index; `ref_moe_scatter` validates the expert_out extent and every destination
  slot, before any read.
- M7 `ref_moe_layer` rejects tokens == 0 before dividing ids.size() by tokens, and
  validates the routing table geometry.
- M8 `deepseek_ocr2_compose` now GROUNDS every region through the parser: markers
  are stripped into canonical content, the union box + per-box polygon corners
  populate the source geometry, and malformed/out-of-range grounding fails closed
  (previously markers leaked with an empty box and bad coordinates passed).
- M9 router softmax: the running max is seeded from a REAL element (not a finite
  -3.0e38 sentinel), so a finite extreme row can no longer underflow every exp()
  to a zero sum and produce NaN; an explicit zero-sum guard matches the oracle.
- S1 tied word embeddings are refused for the MoE decoder and the preflight no
  longer invents tied-head transpose bytes.
- S2 `create_cache` refuses max_seq_len > decoder.max_position_embeddings.
- S3 the device routing readback is validated (counts sum == slots, exclusive
  prefix, off+count <= slots) through the SHARED `validate_moe_routing_tables`
  before it drives any subview; it returns a Status, never aborts.
- S4 upload staging is retained on EVERY exit path by `UploadStagingGuard`:
  caller-owned lists get the completion callback (so a post-error commit is
  safe); executor-owned lists are committed AND drained on error, which both
  releases the staging and satisfies the runtime's committed-before-destroy rule.
- S5 MemoryLimits are enforced end-to-end (weights + exact workspace + KV at the
  actual max sequence) in create() and create_cache(), before allocation.
- S6 the MoE input validators run on the actual execution path.
- S7 grounding coordinates are the CLOSED [0,1000] (header + docs agree) and
  trailing commas are rejected.
- S8 the decoder oracle consumes the INDEPENDENT vision-oracle features, with the
  device features separately asserted equal in the same test.
- S9 the device custom-mask BYTES are downloaded and compared against an
  independent structural expectation; the executor and the oracle share ONE
  canonical builder (`MoeVlmExecutor::build_visual_query_mask`).
- S10 the empty-assignment case runs the real device router/top-k/histogram/
  offsets/permutation chain and drives the expert loop from the validated DEVICE
  counts, instead of uploading oracle routing.
- S11 the composed-route test compares EXACT device-vs-host expert ids derived
  from the SAME x/W via the host's independent projection.

Phase 10 verification (post-review): cuda, xmake AND CMake, debug. New suites
(tests/asserts, identical in both build systems): deepseek_contract 11/105 (host),
moe_router 7/755, moe_permutation 4/59, moe_experts 2/12,
model_deepseek_ocr2 12/193 (36 tests / 1124 asserts total). All 27 host suites
(2612 asserts / 251 tests; CTest unit_ocr 27/27) and the FULL OCR device set of
36 suites (6652 asserts / 265 tests) are green in BOTH systems, i.e. the 32 prior
device suites (5633 asserts / 240 tests) show no regression; example_ocr builds
in both. Declared tolerances, the exact synthetic architecture, and the measured
memory facts are recorded in MODELS.md. vk remains upstream blocked (no backend
edits). These are SYNTHETIC parity gates only: DeepSeek-OCR-2 remains planned,
real artifacts fail closed unconditionally, and Session fails closed with
unvalidated. No official accuracy, no 8 GB fit claim, and no RTX 4060
performance claim is made.

Phase 11 opt-in execution features (current; synthetic parity only)

Phase 11 is additive and default-off; the reference command sequences and all
prior suites are preserved:
- Exact-capacity P7 ragged crop chunking: P7 region recognizers record ragged
  multi-crop vision graphs and split them by EXACT packed-patch capacity
  (runtime/p7_vlm_executor.*). Paddle/OpenDoc recognizers remain per-crop.
  Batch-vs-1 tests assert actual output equality, including the chunked path.
- Optional CUDA graph decode: DenseDecodeOptions.enable_cuda_graph captures
  and updates a single-sequence decode graph; unsupported backends use the
  ordinary stream path and report an explicit fallback; capture/update
  failures fail closed (no partial graph output). DenseDecodeResult and the
  P7/Paddle recognizers expose cuda_graph_steps/updates/fallbacks counters.
  SchedulerLimits.enable_cuda_graph forwards the opt-in through the layout
  orchestrator to each generative region recognizer.
- Optional DenseExecutorConfig.enable_fused_residual_rms_norm fuses the output
  projection residual with post-attention RMSNorm; default false preserves
  the reference sequence.
Boundaries: text decode remains independent batch-1 per region;
plan_decode_batches / max_decode_batch stay planning-only (no batched decode
executor exists). The GDN/MoE/Surya executors have no CUDA graph path; paged
KV still fails closed; native fast paths, GLM MTP, and Hunyuan DFlash remain
unsupported. These are engine features verified by synthetic parity tests on
cuda (RTX 4080 SUPER) — NOT real-model support, and no performance claim.

Phase 12 offline qualification surface (current)

- ocr/cli.*: CliArgs.list_models + --list-models short-circuit like --help
  (no backend/package/input required). model_support_report_json() emits
  valid JSON derived from the SAME default planned table Session will consult
  once real loading is enabled:
  target_count=17, one entry per target (target/model/family/support_state/
  validated), and explicit real_model_validated=false /
  session_status="unvalidated". Synthetic parity never upgrades a recipe.
- example_ocr.cpp prints the report and exits 0 BEFORE session validation.
- QUALIFICATION.md records the honest 17-target matrix (synthetic evidence,
  backend actually measured, real checkpoint/package/precision/mode/
  performance explicitly NOT QUALIFIED) and the exact host/device
  qualification commands, including how real rows must be populated from
  external licensed artifacts without tests modifying goldens.
- Session::create / Session::recognize still fail closed with unvalidated
  for every real artifact; no weights are bundled and no runtime downloads
  exist.
