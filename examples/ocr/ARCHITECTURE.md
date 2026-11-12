# OCR engine architecture

This document describes the frozen Phase 0 contracts, the Phase 1 host
infrastructure, the Phase 2 portable tensor runtime and operators, the
Phase 3 dense autoregressive decoder core, the Phase 4 Qwen3-VL vertical
slice + FireRed-OCR recipe + structured outputs, and the architecture later
phases implement. Nothing here is a claim that a REAL model runs. Graph
contracts are transcribed from the requirement as contracts to verify against
pinned official checkpoints/configs/reference implementations before any
recipe is implemented (plan Phase 0 task 1). No official artifact has been
inspected yet; all Phase 1/2/3/4 fixtures are synthetic. The Phase 3 dense
decoder and the Phase 4 Qwen3-VL slice are validated only against a synthetic
CPU oracle, not against any checkpoint.

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
--- later phases (not present yet) ---
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

| Family | Executor (later phase) | Models |
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
   upload and the checked memory preflight against MemoryLimits arrive with
   the runtime phase.
4. **Recognize (later phases):** preprocess -> overview/layout -> crops ->
   vision batches -> decode batches -> structured output, with cancellation
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
- **Graph:** CUDA-graph capture (later) is DSL-only command lists; native
  shader dispatches must not be assumed to compose with `CudaGraphExt`.
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
MTP/DFlash, graph capture, paged KV, or batching beyond batch 1 is present.
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