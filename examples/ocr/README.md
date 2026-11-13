# LuisaCompute OCR engine (`examples/ocr`)

C++20 OCR / document-inference engine for LuisaCompute with replaceable model
recipes and separate CNN, dense autoregressive, layout-routed, hybrid Gated
DeltaNet (GDN), and MoE execution paths.

> Phase 12 status (current): Phases 0-11 provide a portable FP32 tensor
> runtime, correctness-first operators, a dense autoregressive decoder core,
> a Qwen3-VL vertical slice (vision + interleaved M-RoPE + QK-Norm/GQA Qwen3
> decoder + DeepStack), a FireRed-OCR recipe contract reusing that graph,
> strict structured outputs (Markdown/HTML/OTSL/LaTeX/JSON), a conventional
> CNN OCR executor (detector -> DB -> polygon crop -> recognizer -> CTC) with
> per-size PP-OCRv6 contracts, layout/region infrastructure
> (PP-DocLayoutV2/V3 + token-budgeted region scheduling) with a PaddleOCR-VL
> vertical slice (native-resolution NaViT + adaptive visual-token reduction +
> MLP connector + ERNIE decoder) and an OpenDoc two-stage vertical slice
> (PP-DocLayoutV2 + UniRec CNN+CTC), four custom dense/layout vertical slices
> (TeleOCR, GLM-OCR, MinerU, HunyuanOCR; Phase 7), the remaining dense ViT
> models (dots.mocr, GOT-OCR2.0, olmOCR-2) with a real packed-INT4 W4A32
> decoder path, relative-position vision attention, and fail-before-allocation
> memory preflights (Phase 8), the hybrid GDN subsystem (causal Conv1D ring
> state + delta-rule recurrent prefill + hybrid cache + a decoder that
> explicitly mixes GDN and periodic full GQA layers) with the OvisOCR2 / Surya
> slices (Phase 9), and the MoE subsystem (FP32 router, deterministic top-k,
> token-to-expert permutation, an active-expert reference loop, and two shared
> dense experts) with the DeepSeek-OCR-2 slice (SAM-style local perception,
> BOTH convolutional downsamplers, a visual LM encoder with causal-flow
> queries and the EXACT custom visual/query block mask, and exact measured
> memory preflights; Phase 10). Phase 11 added exact-capacity P7 ragged
> multi-crop vision graph chunking, optional DenseDecodeOptions.enable_cuda_graph
> (single-sequence CUDA graph capture/update with explicit fallback counters),
> SchedulerLimits.enable_cuda_graph forwarding, P7/Paddle graph counters, and
> optional DenseExecutorConfig.enable_fused_residual_rms_norm (all opt-in;
> text decode remains independent batch-1 and decode-batch planning stays
> planning-only). Phase 12 added the offline qualification surface:
> `example_ocr --list-models` prints the registry-driven support report JSON
> (17 targets, all planned/unvalidated) without any backend/package/input,
> and QUALIFICATION.md records the honest 17-target matrix. Small SYNTHETIC
> slices execute end-to-end and match reference intermediate tensors at
> every layered golden point (current device evidence: cuda on an RTX 4080
> SUPER; historical vk notes are labeled per phase). No REAL model works yet;
> all models remain planned and real artifacts fail closed.
> All 15 requirement models (17 selectable targets counting the three
> PP-OCRv6 sizes) are planned. Session creation and recognition validate
> their arguments and then fail closed with unvalidated — there is no
> mock OCR output, no bundled weights, and no network download. See
> [MODELS.md](MODELS.md) for the honest support matrix and the unmet official
> gates, and [QUALIFICATION.md](QUALIFICATION.md) for the Phase 12
> qualification matrix and the exact synthetic/real qualification commands.

What exists in Phase 0

| Piece | File | Purpose |
|---|---|---|
| Public API types | ocr/api.h | Modes, options (pixel/token/region budgets, prompt/schema, guidance, sampling, precision/cache/offload/graph/fast-path policies, memory/time limits, cancellation), result schema (canonical text, lines/blocks with source coordinates, explicit confidence presence, metadata, diagnostics, counters) |
| Status/Result | ocr/status.h/.cpp | Recoverable failures without exceptions or LUISA_ERROR |
| Recipe contracts | ocr/model_recipe.h/.cpp | Identity/revision/fingerprint, preprocessing, component graph contracts, tensor mappings, prompt/stop/parser, capabilities, context/license/support state |
| Registry | ocr/model_registry.h/.cpp | All 17 planned targets; exact-signature fail-closed matching (validated support + pinned identity + package format + exact tensor signature); explicit unsafe experimental override that stays unvalidated |
| Session | ocr/session.h/.cpp | One session = one model; preflight_request (validation then cancellation), then fail-closed refusal |
| CLI parser/report | ocr/cli.h/.cpp | Reusable UTF-8 argument parser + structured status JSON report (host-testable) |
| CLI entry | example_ocr.cpp | Windows wmain UTF-8 argv; --help needs no device/model; never claims inference succeeded |

What Phase 1 adds (host-only; no device/model required)

| Piece | File | Purpose |
|---|---|---|
| Quantization codecs | ocr/quantization.h/.cpp | Checked 64-bit add/mul/align; explicit RNE FP16/BF16; grouped INT4 sym/asym + INT8 pack/unpack/dequant/quantize (group 64/128) |
| .lcocr v1 package | ocr/package.h/.cpp | Versioned little-endian reader/writer; on-demand payload reads; component manifest; fail-closed validation |
| Safetensors reader | ocr/safetensors_reader.h/.cpp | Checked host inspection reader (not a production inference path) |
| Tokenizer | ocr/tokenizer.h/.cpp | yyjson HF tokenizer.json; byte-level BPE + Unigram/byte-fallback subsets; fail-closed on unsupported semantics; explicit invalid-UTF-8 rejection |
| Prompt builders | ocr/prompt.h/.cpp | Recipe-ID-specific; unknown templates fail closed (no general Jinja) |
| Detokenizer | ocr/detokenizer.h/.cpp | Incremental UTF-8 across tokens, EOS, stop strings (no split chars / stop leakage) |
| Image input | ocr/image.h/.cpp | Checked JPEG/PNG decode (stb), limits, RGB/BGR, row stride, checked EXIF orientation |
| Document input | ocr/document.h/.cpp | Externally-rasterized pages + manifest with limits; NO in-process PDF parser |
| Preprocess | ocr/preprocess.h/.cpp | Checked resize/normalize/patch + packed ragged batches; source-coordinate provenance |
| Feature bundle | ocr/feature_bundle.h/.cpp | .ocrfeat v1 checked interchange for official-processor differentials |
| Result JSON | ocr/output/{types.h,json_output.h/.cpp} | FULL RecognizeResult schema + strict status JSON; nonfinite floats rejected |
| Python tools | tools/*.py | pack_model (streaming safe_open), validate_package, preprocess, rasterize_pdf, export_golden, generate_fixtures (offline) |
| Frozen fixtures | src/tests/data/ocr/ | Small SYNTHETIC package/safetensors/tokenizer/image cases + golden_manifest.json |

What Phase 2 adds (portable tensor runtime + correctness-first operators)

Runtime (ocr/runtime/):

| Piece | File | Purpose |
|---|---|---|
| Tensor metadata | tensor.h/.cpp | Logical shape/stride/dtype/layout; non-owning BufferView descriptors; checked 64-bit element/byte/range math; storage vs compute precision kept separate; no hidden host values |
| Device weights | weights.h/.cpp | Immutable per-component weights; component-selective loading from PackageReader; actual resident-byte accounting; explicit host staging lifetime (releasable only after upload completion); synchronous upload-and-synchronize init; all transfers luisa::span |
| Workspace | workspace.h/.cpp | Grow-only aligned arenas with checked offsets; reset bumps an epoch so stale regions never alias fresh ones; retire-after-sync never invalidates in-flight work; host planning + device buffer ownership |
| Shader cache | shader_cache.h/.cpp | Typed per-owner cache + stable variant-key registry; compile each device-free factory once per variant; compile_ok() checked; unique_ptr ownership keeps returned pointers stable across rehash; no RTTI |
| Dispatch | dispatch.h/.cpp, dispatch_adapter.h | TOTAL-thread plans with tail guards; checked uint32/uint64 envelopes (overflow-safe ceil-div/sqrt); row/query chunking; flattened 2D grids to avoid the 65535-group cap; never clamps/drops work. dispatch_adapter is the single launch path: skips empty plans and uses the uint3-span form for flattened 1D grids so no work is silently truncated |
| Memory budget | memory_budget.h/.cpp | Checked weight/workspace/KV/shader-cache/host-staging accounting per component; limit preflight with required-vs-available Status before allocation; no invented universal device-memory query |

Operators (ocr/ops/, naive correct portable FP32 DSL; device-free factories):

| Piece | File | Purpose |
|---|---|---|
| Image | ops_image.h/.cpp | Fused color-convert + bilinear/area resize + normalization (matches Phase 1 preprocess convention); crop; letterbox; patchify/packing |
| GEMM/GEMV | ops_gemm.h/.cpp | Ragged M/N/K FP32 GEMM with FP32 accumulation; compile-time epilogues (none/bias/GELU-erf/GELU-tanh/ReLU/ReLU6/SiLU/hard-swish/residual/bias-residual); separate decode GEMV/small-M; packed FP16/BF16 B; grouped INT4 sym/asym W4A32 (Phase 1 low-nibble/group/scale/zp semantics); grouped INT8; SwiGLU/GeGLU gated act; W4A16 is an INDEPENDENT variant that fails closed (no FP16 activation arithmetic yet) |
| Norm | ops_norm.h/.cpp | LayerNorm/RMSNorm (weighted + parameterless), QK-Norm shape, L2, row-sum reduction, residual add; explicit cooperative 256-thread/shared-tree LayerNorm/RMSNorm variants for large H, with dense hidden-RMSNorm threshold routing and a reference kill switch |
| RoPE | ops_rope.h/.cpp | 1D, 2D axial, M-RoPE, interleaved MRoPE, XD-RoPE as SEPARATE variants with position buffers + axis mapping data; sinusoidal table |
| Attention | ops_attention.h/.cpp | Explicit AttentionMaskDescriptor (BIDIRECTIONAL/CAUSAL/WINDOWED/PREFIX_CAUSAL/CUSTOM_BLOCK_MASK); reference scores+FP32 softmax+PV and online-softmax A/B; GQA/MHA; softcap; relative bias; DeepSeek-style custom block mask expressible; separate contiguous-KV decode |
| Conv | ops_conv.h/.cpp | Conv2D general kernels/stride(asym)/pad/dilation/groups/depthwise; Conv1D + causal depthwise; BatchNorm; avg/adaptive/global pool; concat/split/shuffle |
| Vision | ops_vision.h/.cpp | Pixel-unshuffle; spatial merge/token packing; absolute/learned position addition; DeepStack injection with caller-selected ranges (data, not hardcoded) |
| Decode | ops_decode.h/.cpp | Embedding lookup (tied-embedding compatible); multimodal scatter; greedy argmax; temperature softmax |
| CPU oracles | cpu_reference.h/.cpp | Independent naive double-accumulating host references for every device op |

Phase 2 tests: host unit suites (memory_budget, runtime dispatch/tensor) plus
device integration suites (ops_image/gemm/norm/rope/attention/conv/vision and
runtime weights/workspace/shader_cache) run with an explicit backend
(xmake run test_ocr_ops_gemm vk|cuda). All pass on vk and cuda. The fallback
backend is not enabled in the reference configuration (requires LLVM+Embree).

What Phase 3 adds (dense autoregressive decoder core; synthetic parity only)

| Piece | File | Purpose |
|---|---|---|
| Shared decoder components | ocr/models/qwen2_5_vl.h/.cpp | Recipe-driven dense decoder geometry with an explicit RoPE pair-layout field (adjacent vs rotate-half); strict Qwen2.5 text-config parsing (selects rotate-half; fails closed on Qwen3/MoE/GDN and ENABLED sliding-window markers, accepts a disabled window), exact tensor-name/layout inventory + per-weight shape/layout/element-count validation (standalone lm_head is [hidden,vocab] packed kn). NOT a model; no vision tower, no M-RoPE wiring. Qwen2.5-VL and Qwen3-VL contracts are kept distinct. |
| Decode cache | ocr/runtime/decode_cache.h/.cpp | Distinct contiguous-KV (batch-1 correctness) and optional paged-KV (fails closed unsupported) interfaces; device-visible position/step/state metadata synchronously zero-initialized and updated through completion-retained heap staging; checked capacity from actual geometry (layers/KV heads/head dim/bytes/max seq/batch, INCLUDING metadata bytes) preflighted before allocation; immediate host-slot release on completion plus a real record_finished device API; no pool churn, no dangling reset spans. |
| Executor contract | ocr/runtime/executor.h/.cpp | Command-recording prefill/decode-step interface; ownership rules (immutable shared weights, per-request workspaces, cached shaders, luisa::span transfers, no exceptions/RTTI). |
| Dense executor | ocr/runtime/dense_vlm_executor.h/.cpp | Embedding lookup + multimodal replacement, dense blocks (QKV+bias, config-selected adjacent/rotate-half 1D RoPE, GQA causal attention, o-proj/residual, RMSNorm, SwiGLU MLP), contiguous-KV append/read, final norm + LM head (tied or standalone, transpose preflighted/accounted). Every weight validated (shape/layout/element count) and every prompt token bounds-checked before dispatch; per-record completion-retained upload staging (multiple records per CommandList safe). Separate chunked prefill and single-step decode loops; host decode controller with greedy/deterministic sampling, config+options+tokenizer EOS (deduplicated), stop strings via the incremental detokenizer, bounded output/cache, cancellation + wall-time timeout at step boundaries, RAII cache release on every failure, truncation diagnostics. |
| Decode ops | ocr/ops/ops_decode.h/.cpp | Adds the contiguous-KV append kernel + host deterministic sampling building blocks (explicit seed, temperature/top-k/top-p validation, repetition penalty, no-repeat-ngram; -inf allowed as masked, all-NaN/+inf/banned rows return a clear error without advancing the RNG). |

Phase 3 tests (device integration, explicit backend): test_ocr_decode_prefill
(embedding/multimodal replacement, first-layer KV, full-prefill logits,
first-token logits, chunked prefill, rotate-half parity, undersized-q_proj /
transposed / wrong-layout lm_head refusal at create, OOV prompt-token refusal),
test_ocr_decode_cache (capacity math, budget preflight incl. metadata bytes,
append/release/reuse, EVERY decode-step KV+logit parity, two-steps-in-one-list
parity, reset stack-churn zero + reuse, synchronous metadata zero-init +
record_finished, tied-head accounting, capacity bounds, workspace bounds, paged
fail-closed), test_ocr_decode_sampling (greedy/softmax device parity, sampling
determinism, top-k/top-p/repetition/no-repeat-ngram, invalid parameters,
all-NaN/all--inf/all-banned refusal with no RNG advance on failure),
test_ocr_decode_stop (early EOS, multi-token output, repeat-run determinism,
config-only EOS, stop strings across tokens/UTF-8, max-token and cache-capacity
truncation, failure/cancel/timeout cache release, prefill/decode-boundary
cancellation, deterministic decode-boundary timeout). Host-only
test_ocr_decoder_config covers the config parser (integer rope_theta,
rotate-half selection, Qwen3/MoE/GDN refusal, disabled-vs-enabled sliding
window) and the tensor inventory/validation. All pass on vk and cuda against an
independent double-accumulating CPU oracle built from the Phase 2 references.
The synthetic decoder geometry is vocab 32 / hidden 16 / 2 layers / 4 Q heads /
2 KV heads / head dim 4 / intermediate 32 (GQA group 2); FP32 throughout. The
parity decoder defaults to the adjacent RoPE convention; a rotate-half variant
(and direct rope-op parity) exercises the Qwen2.5-style layout selected by
parse_qwen2_5_text_config. Tolerances: 3e-4 absolute + 3e-4 relative for decoder
logits/KV, 2e-4 for the standalone softmax/argmax/rope kernels. These are
SYNTHETIC parity gates only — no official checkpoint, tokenizer, or processor
has been inspected, and no real model is claimed to work.

What Phase 4 adds (Qwen3-VL vertical slice + FireRed-OCR + structured outputs; synthetic parity only)

| Piece | File | Purpose |
|---|---|---|
| Qwen3-VL contract | ocr/models/qwen3_vl.h/.cpp | DISTINCT Qwen3-VL recipe/config/tensor contract: vision (patch embed, bidirectional QK-Norm attention, 2D axial RoPE, GELU MLP, spatial merger), interleaved M-RoPE with explicit temporal/height/width axis sections, Qwen3 text decoder (QK-Norm, GQA, NO QKV bias), DeepStack projection/injection at exact configured early decoder layers. Strict config parsing with an Instruct/Thinking identity boundary (Thinking fails closed; Qwen2.5 markers refused). Native/dynamic image packing + source transforms + host M-RoPE position semantics. Exact expected tensor inventory (QK-Norm required, QKV bias absent). NOT an official checkpoint. |
| Qwen3-VL executor | ocr/runtime/qwen3_vl_executor.h/.cpp | Records the vision graph from Phase 2 operators and drives the shared Phase 3 dense executor with additive Qwen3 contract switches (qk_norm, no qkv_bias, rope_axes=3, deepstack_layers). The Phase 3 Qwen2.5-style path is unchanged. Layered download helpers for golden tests. |
| Preprocess bridge | ocr/runtime/qwen3_vl_preprocess.h/.cpp, qwen3_vl_image.h | Host-only bridge to the Phase 1 preprocess reference, isolated so the host AttentionMaskKind never meets the ops enum in a device TU. |
| FireRed-OCR recipe | ocr/models/fire_red_ocr.h/.cpp | Reuses the SAME Qwen3-VL graph only through its own tensor map (language_model./model.visual.), prompt (fire_red_ocr), and output parser (fire_red_markdown). Real artifacts fail closed unconditionally; a training/GRPO-node guard rejects forbidden roles. No GRPO/training operators. |
| Markdown output | ocr/output/markdown.h/.cpp | Strict block parsing/composition (title/text/list/figure/code/table/formula) with reading order + source spans; unclosed fence/math/table and bad structure fail closed; LaTeX delimiter/content sanity. |
| Table output | ocr/output/table.h/.cpp | HTML and OTSL table parsing/validation (colspan/rowspan, caption, entities; OTSL fcel/eced/lcel/ucel/xcel/nl). Malformed input is an error, never a text fallback; canonical HTML re-serialization. |
| JSON extraction | ocr/output/extract.h/.cpp | Strict JSON parse + deliberately supported schema subset (object/array/string/number/integer/boolean/null, properties/required/additionalProperties, items, min/max, enum/const). Unknown schema features fail closed; no plain-text fallback. Repair OFF by default; enabled repair produces diagnostics. |

Phase 4 tests: device integration test_ocr_model_qwen3_vl (packing/position
semantics, ragged-image vision parity, isolated first vision block, isolated
first decoder layer, full prefill + decode KV/logits parity with DeepStack +
interleaved MRoPE, deterministic greedy tokens vs the oracle, blank/repeated/
adversarial graph behavior) and test_ocr_model_fire_red (recipe contract +
shared-graph parity), both on vk and cuda. Host suites test_ocr_output_table,
test_ocr_output_markdown, test_ocr_output_extract, test_ocr_model_contract.
The synthetic Qwen3-VL geometry (vocab 32 / hidden 32 / 3 decoder layers /
4 Q + 2 KV heads / head dim 8; vision 16 hidden / 2 heads / 2 layers / patch 2
/ merge 2; DeepStack at decoder layers {0,1}) is a DECLARED contract, not an
official checkpoint. Tolerances: 5e-4 abs + 5e-4 rel for linear/norm stages,
3e-3 abs + 3e-3 rel for GELU-erf stages; greedy token ids match the oracle
exactly and repeat-run identically. These are SYNTHETIC parity gates only —
no official checkpoint/processor/template/parser has been inspected, blank/
adversarial tests do NOT establish real-VLM hallucination safety, no VLM
confidence is produced, and every real model remains planned with Session
fail-closed. The unmet official gates are recorded in MODELS.md.

Phases 5-11 add executors/models per the plan (all implemented as synthetic
parity slices; see the per-phase sections below); the Phase 0/1/2/3/4 types
are the stable surface they implement against. No real model is validated.

What Phase 5 adds (CNN OCR executor + PP-OCRv6 contracts + CTC/DB/output; synthetic parity only)

| Piece | File | Purpose |
|---|---|---|
| PP-OCRv6 contracts | models/pp_ocrv6.h/.cpp | Per-size (tiny/small/medium) LCNetV4-style detector + RepLKFPN + DB head and LCNetV4-style recognizer + LightSVTR + CTC geometry/tensor maps, validated independently; inference-only Conv+BN and RepLK folding helpers; real-artifact gate (always unvalidated); NRTR/dropout/loss/training guard |
| CNN executor | runtime/cnn_executor.h/.cpp | REAL detector -> DB postprocess -> rotated polygon line crop -> recognizer -> CTC pipeline; separate preprocessing/detector/crop/recognizer stages; immutable shared weights; per-request workspace; component/pixel/region limits; luisa::span transfers; source-coordinate transforms; blank pages -> valid empty result |
| Public-result bridge | runtime/cnn_executor_result.h/.cpp | Host-only mapping of the plain CNN result to the frozen public RecognizeResult schema (OcrLine/SourceRegion/ModelMetadata); confidence carried through from CTC, never synthesized; `output_tokens` counts CTC kept TOKENS (not UTF-8 bytes; unknown -> 0) |
| CTC ops | ops/ops_ctc.h/.cpp | Device frame log-softmax (temperature, max-subtracted), argmax+per-frame probs, sigmoid, LightSVTR height-collapse, FPN bilinear resize, rotated polygon crop; independent host oracles for each |
| DB postprocess | output/db_postprocess.h/.cpp | CPU baseline: checked binarization, 8-connected components, convex hull + min-area rotated rect, exact convex miter unclip (official area*ratio/perimeter distance), box scoring, optional IoU NMS, degenerate/tiny/empty/adversarial handling, source transforms. NO OpenCV. Explicitly NOT the official contour/pyclipper pipeline (documented in the header) |
| CTC decode | output/ctc_decode.h/.cpp | Declared PP-OCR reference greedy semantics: raw-sequence duplicate collapse before blank removal, dictionary lookup (CJK/emoji), confidence = arithmetic mean of kept-frame probabilities; empty -> no confidence |
| Reading order | output/reading_order.h/.cpp | Deterministic multi-column order, row-tolerance banding for skewed/rotated/wide lines, stable tie-breaking, page-index preservation, canonical '\n' joining |

Phase 5 tests: host unit suites test_ocr_ctc (blank=0/nonzero, repeats,
blank-separated repeats, CJK/emoji, empty frames, invalid ids/probs,
temperature/log-softmax stability, confidence formula), test_ocr_db
(binarize/components/hull/min-area-rect/unclip/scoring/IoU/NMS/transform/
degenerate/adversarial), test_ocr_reading_order (columns/rows/skew/wide/
tie-break/pages/join/empty). Device integration test_ocr_model_pp_ocrv6 on
vk and cuda: per-size config/folding-math gates, detector probability-map
parity vs an UNFOLDED two-step oracle, DB polygons/scores, crop pixels,
per-frame CTC logits, collapsed ids/text/confidence, reading order, blank
pages (no hallucinated lines), repeat-run determinism, wide lines,
multi-page, and pixel/region limit refusals. Review-hardening tests: quad
winding/crop orientation vs an INDEPENDENT source-derived crop oracle,
non-identity FPN resize parity (ragged up/downsample + tails), oversized
FPN-skip and bad height_collapse refusals, frame-count-vs-collapse checks,
and an ON-DISK wrong-layout package refusal through create_from_packages.
Host suite test_ocr_cnn_result covers the public-result bridge incl.
CJK/emoji token counting. The synthetic per-size
architecture and tolerances are recorded in MODELS.md; they are SYNTHETIC
parity gates only. PP-OCRv6 tiny/small/medium all remain planned;
PP-OCRv6_medium is NOT a validated default; no official polygon/CER/
confidence/calibration or RTX 4060 performance gate is met; no runtime
downloads exist.

What Phase 6 adds (layout/region infrastructure + PaddleOCR-VL + OpenDoc; synthetic parity only)

| Piece | File | Purpose |
|---|---|---|
| PP-DocLayout contracts | models/pp_doc_layout.h/.cpp | SEPARATE V2/V3 layout detector configs/tensor maps (folded conv backbone + 1x1 class head (sigmoid) + 1x1 box head); CPU decode (score threshold, clamped exp box decode, NMS AND class-aware NMS, region scoring, stable reading order, source transforms, degenerate/empty handling); real-artifact gate (always unvalidated) |
| Region scheduler | runtime/scheduler.h/.cpp | Token-budgeted region scheduling: one overview/layout pass; crops sorted by approximate patch/token count; vision/prefill batches under max_patches/max_regions; decode batches under max_decode_batch; separate overview/crop/output budgets; restore to page/reading order; bounded detector-output cache keyed by explicit content/geometry hash for recognition retry |
| Layout orchestration | runtime/layout_vlm_executor.h/.cpp | PpDocLayoutExecutor (device conv graph); RegionRecognizer interface with PaddleOcrVlRecognizer (NaViT + adaptive token reduction + MLP connector + ERNIE decoder) and OpenDocRecognizer (PP-DocLayoutV2 + UniRec CNN+CTC, NOT a dense VLM); LayoutVlmExecutor full overview->layout->crop->region->composition orchestration with cancellation/time/memory checks, immediate sequence/workspace release, and a documented cpu_reference layout option |
| PaddleOCR-VL contract | models/paddle_ocr_vl.h/.cpp | Distinct native-resolution NaViT (QKV bias, no QK-Norm/DeepStack), adaptive visual-token reduction + MLP connector, ERNIE-4.5-0.3B-style decoder (rotate-half RoPE, QKV bias); native packing; tensor inventory; real-artifact gate |
| OpenDoc contract | models/open_doc.h/.cpp | PP-DocLayoutV2 + UniRec-0.1B two-stage text/formula/table recognizer (CNN+CTC, no autoregressive KV); tensor inventory; real-artifact gate |
| Host composition bridges | models/paddle_ocr_vl_result.*, models/open_doc_result.*, runtime/layout_result.* | Strict Markdown/JSON/plain/spotting composition; type-specific adapters (text/formula/table); malformed output fails closed (no plain-text fallback); public RecognizeResult with lines/blocks/source page/box/polygon/type/content/reading order; confidence only from validated detector/CTC scores labeled internal/test (no pseudo-probability for generated content) |

Phase 6 tests: host unit suites test_ocr_scheduler (batch packing/sort/tie/
oversized-refusal/decode-batches/restore-order/output-budget/cache-eviction/
hashes), test_ocr_layout_decode (threshold/box-decode/NMS/class-aware/tie/cap/
degenerate/transform/malformed/reading-order/V2-V3-distinct), and
test_ocr_layout_compose (type adapters, strict Markdown/JSON errors, public
schema, confidence provenance, empty/cross-page). Device integration suites on
vk and cuda: test_ocr_layout (detector head-map + decoded box/score/class parity
vs an independent CPU oracle, cpu_reference option, empty/determinism),
test_ocr_model_paddle_ocr_vl (NaViT patches + connector + ERNIE greedy-token
parity, batch-vs-batch-1 equivalence), test_ocr_model_open_doc (UniRec CTC
parity vs an UNFOLDED oracle, batch equivalence, full two-stage pipeline, empty
page), and test_ocr_region_scheduler (batch-vs-single ordered equivalence,
layout cache retry, cancellation, per-page output budget, shuffled-page sort,
document-global region ids, memory accounting, cpu_reference layout,
cross-page). Phase 11 P7 region recognizers execute ragged multi-crop vision
graphs and split them by exact packed-patch capacity; Paddle/OpenDoc still
execute crops one at a time. Text decode remains independent batch-1 and
plan_decode_batches/max_decode_batch remain planning-only. Batch-vs-1 tests
assert actual output equality, including P7 exact-capacity chunking. The synthetic architecture and tolerances are
recorded in MODELS.md; they are SYNTHETIC parity gates only. PP-DocLayoutV2/V3,
PaddleOCR-VL, and OpenDoc all remain planned; PaddleOCR-VL is NOT a validated
complex-document default; no official
layout mAP/reading-order/table/formula/Markdown/JSON/determinism/memory
or RTX 4060 performance gate is met; no runtime downloads exist.

Backend verification note (upstream blocker): the cuda backend passes all Phase
6 suites in both xmake and CMake. The vk backend is currently blocked by an
UPSTREAM post-merge regression unrelated to Phase 6: vkCreateDevice chains
VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT (sType 1000657000), which
the installed Khronos validation layer (1.4.321.1) rejects as an unknown
structure, escalating to a fatal error. The pre-existing baseline
test_ocr_ops_gemm fails identically on vk, confirming this is not a Phase 6
issue. The merged vk backend also fails to compile under CMake's non-unity build
(missing opacity_micromap.h / command_buffer_sync.h includes in device.cpp /
blas.cpp); xmake's unity build masks this. Phase 6 vk parity was verified green
before the upstream merge; no backend edits are made here. Re-run vk once the
upstream device-create chain / validation-layer mismatch is resolved.

What Phase 11 adds (opt-in execution features; synthetic parity only)

| Piece | File | Purpose |
|---|---|---|
| Exact-capacity P7 crop chunking | ocr/runtime/p7_vlm_executor.* | P7 region recognizers record ragged multi-crop vision graphs and split them by EXACT packed-patch capacity (no approximate-cost overflow); Paddle/OpenDoc remain per-crop; batch-vs-1 tests assert actual output equality |
| Optional CUDA graph decode | ocr/runtime/dense_vlm_executor.* | DenseDecodeOptions.enable_cuda_graph: opt-in single-sequence CUDA graph capture/update; unsupported backends use the ordinary stream path with an explicit fallback counter; capture/update failures fail closed (no partial graph output); steps/updates/fallbacks counters |
| Scheduler graph forwarding | ocr/runtime/scheduler.h, layout_vlm_executor.* | SchedulerLimits.enable_cuda_graph forwarded through the orchestrator to each generative region recognizer (default off) |
| P7/Paddle graph counters | ocr/runtime/p7_vlm_executor.h, layout_vlm_executor.h | cuda_graph_steps/updates/fallbacks observability on the P7 slice and PaddleOcrVlRecognizer |
| Optional fused residual+RMSNorm | ocr/runtime/dense_vlm_executor.* | DenseExecutorConfig.enable_fused_residual_rms_norm (default false preserves the reference command sequence) |

Phase 11 boundaries: text decode remains independent batch-1 per region;
plan_decode_batches / max_decode_batch stay planning-only (no batched decode
executor). GDN/MoE/Surya executors have no CUDA graph path. These are opt-in
engine features verified by synthetic parity tests (test_ocr_decode_stop,
test_ocr_region_scheduler, test_ocr_model_paddle_ocr_vl on cuda); they are NOT
real-model support and carry no performance claim.

What Phase 12 adds (offline qualification surface; no device/model required)

| Piece | File | Purpose |
|---|---|---|
| --list-models | ocr/cli.h/.cpp, example_ocr.cpp | CliArgs.list_models short-circuits like --help (no backend/package/input); prints model_support_report_json() and exits 0 BEFORE session validation |
| Support report | ocr/cli.cpp::model_support_report_json | Registry-driven valid JSON: target_count=17, per-target target/model/family/support_state/validated, explicit real_model_validated=false and session_status="unvalidated" |
| Qualification matrix | QUALIFICATION.md | 17-target matrix: synthetic suite/evidence, backend actually measured (RTX 4080 SUPER cuda; historical vk notes only where already documented), real checkpoint/package/precision/mode/performance explicitly NOT QUALIFIED; exact host/device qualification commands and the external-artifact real-row procedure |
| CLI tests | src/tests/unit/ocr/test_ocr_cli.cpp | --list-models parsing + yyjson-validated report (all 17 entries, none validated, fail-closed Session preserved) |

## Build

CMake (from the repository root):

```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build build --target example_ocr lc-example-ocr
ctest --test-dir build -L unit_ocr
```

xmake:

```bash
xmake f -p windows -a x64 -m debug -c
xmake build example_ocr
# All OCR host tests (no GPU/model needed):
xmake build test_ocr_recipe test_ocr_status test_ocr_cli test_ocr_quantization test_ocr_package test_ocr_safetensors test_ocr_tokenizer test_ocr_prompt test_ocr_image test_ocr_document test_ocr_preprocess test_ocr_json_output
xmake run test_ocr_recipe
xmake run test_ocr_package
```
Targets: lc-example-ocr (static engine library), example_ocr (CLI), and
example_ocr_bench (manual, correctness-gated synthetic benchmark harness).
The benchmark protocol, workload catalog, report schema, and real-model
qualification boundary are documented in BENCHMARK.md (BENCHMARK.md). Benchmark
binaries are not CTest timing tests and produce no official-model claims.
Host tests live in src/tests/unit/ocr/ (CTest label unit;unit_ocr) and need no
GPU or model.
model. Frozen SYNTHETIC fixtures live in
`src/tests/data/ocr/` (see `golden_manifest.json`); they are regenerated only
by `tools/generate_fixtures.py`, never by tests. Official
checkpoint/tokenizer/processor golden qualification is UNMET: no official
artifact has been inspected, and every model remains planned.

Offline Python tools (`examples/ocr/tools/`, useful `--help` on each):
`pack_model.py` (streaming safetensors -> .lcocr v1, requires local official
config/revision/license), `validate_package.py` (reopens and fully checks an
artifact, standard-library only), `preprocess.py` / `export_golden.py`
(explicit local official processor/model artifacts; no implicit downloads),
`rasterize_pdf.py` (external PDF library, file/page/pixel limits, no network),
`generate_fixtures.py` + `lcocr_format.py` / `lcocr_writer.py`
(standard-library-only synthetic fixture generation).

## CLI
```text
example_ocr --backend vk --package model.lcocr --input page.png --mode plain
example_ocr --help
example_ocr --list-models
```
The CLI accepts backend, package/manifest, model selector, repeatable inputs,
mode (plain|spotting|markdown|extract), output path, prompt/schema,
region/color guidance, language (exact match), all pixel/token/region limits,
sampling (greedy by default), precision (automatic default = recipe default
else fp32)/cache/offload/graph/fast-path policies, and memory/time budgets.
Arguments are UTF-8 (Windows wmain converts wide argv; POSIX passes UTF-8).
With no validated model it prints the resolved configuration and then a
structured JSON refusal ("code":"unvalidated") — exit code 2. Bad arguments
exit 1 with a useful message; report I/O errors exit 3; --help and
--list-models exit 0 without any device (the latter prints the registry-driven
support report JSON: 17 planned targets, real_model_validated=false,
session_status="unvalidated"). On a future successful recognition the CLI
writes the FULL result schema via ocr/output/json_output.h (the same
serializer the public API uses).

## Design rules (all phases)

- Namespace `luisa::ocr`; Luisa containers (`luisa::string/vector/...`); C++20.
- No project-owned exception raising and no RTTI. Recoverable failures use
  `Status`/`Result`; `LUISA_ASSERT`/`LUISA_ERROR` are for broken invariants only.
- Host-only engine TUs do not include DSL device headers; kernels (later
  phases) are device-free factories.
- One Session owns one model and its declared dependent components; models
  are selectable, not simultaneously resident.
- No silent substitution of model, architecture, prompt, positional encoding,
  or output parser. Unsupported combinations fail early and clearly.
- Generative VLM results report `has_confidence == false` — never a
  pseudo-probability.
- No weights are committed or redistributed; users supply licensed
  checkpoints. No runtime network access.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the engine design,
[MODELS.md](MODELS.md) for the support matrix, [QUALIFICATION.md](QUALIFICATION.md)
for the Phase 12 qualification matrix, and [FORMAT.md](FORMAT.md) for
the planned package/result formats.
