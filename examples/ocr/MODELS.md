OCR model support matrix

Every model is planned. None works yet. (Phase 1 added host
artifact/tokenizer/image/preprocessing/JSON infrastructure; Phase 2 added a
portable FP32 tensor runtime and operators; Phase 3 added a dense
autoregressive decoder core validated ONLY against a synthetic CPU oracle;
Phase 4 added a Qwen3-VL vertical slice + FireRed-OCR recipe contract and
strict structured outputs, again validated ONLY against a synthetic CPU
oracle; Phase 5 added a conventional CNN OCR executor + PP-OCRv6 graph
contracts + CTC/DB/reading-order output infrastructure, again validated ONLY
against a synthetic CPU oracle; Phase 6 added layout/region infrastructure
(PP-DocLayoutV2/V3 + token-budgeted region scheduling), a PaddleOCR-VL
vertical slice (native-resolution NaViT + adaptive visual-token reduction +
MLP connector + ERNIE decoder), and an OpenDoc two-stage vertical slice
(PP-DocLayoutV2 + UniRec CNN+CTC), again validated ONLY against synthetic CPU
oracles; Phase 7 added four custom dense/layout vertical slices (TeleOCR,
GLM-OCR, MinerU, HunyuanOCR) and Phase 8 added the remaining dense ViT models
(dots.mocr, GOT-OCR2.0, olmOCR-2) with a real packed-INT4 W4A32 decoder path,
again validated ONLY against synthetic CPU oracles on cuda; Phase 9 added the
hybrid GDN subsystem (causal Conv1D ring state + delta-rule recurrent/chunked
prefill + hybrid cache + a decoder that explicitly mixes GDN and periodic full
GQA attention layers) and the OvisOCR2 / Surya synthetic vertical slices, again
validated ONLY against synthetic CPU oracles on cuda in xmake AND CMake; Phase
10 added the MoE subsystem (FP32 router + deterministic top-k + permutation +
active-expert reference loop + two shared dense experts) and the DeepSeek-OCR-2
synthetic vertical slice (SAM local perception + both conv downsamplers + visual
LM encoder with causal-flow queries and the exact custom visual/query block
mask), again validated ONLY against synthetic CPU oracles on cuda in xmake AND
CMake. Phase 11 added opt-in execution features (exact-capacity P7 ragged
multi-crop vision graph chunking, optional DenseDecodeOptions.enable_cuda_graph
single-sequence CUDA graph capture/update with explicit fallback counters,
SchedulerLimits.enable_cuda_graph forwarding, P7/Paddle graph counters, and
optional DenseExecutorConfig.enable_fused_residual_rms_norm), again verified
ONLY by synthetic parity on cuda; text decode remains independent batch-1 and
decode-batch planning stays planning-only. Phase 12 added the offline
qualification surface: example_ocr --list-models prints the registry-driven
support report JSON (17 targets, all planned/unvalidated,
real_model_validated=false, session_status="unvalidated") without any
backend/package/input, and QUALIFICATION.md records the honest 17-target
qualification matrix. No model left the planned state.)

No checkpoint has been pinned, inspected, loaded, or run. There are no
validated revisions, fingerprints, hashes, precisions, backends, or
performance numbers anywhere in this repository's OCR engine. The graph
contracts below are transcribed from the requirement as contracts to
verify against official artifacts — they are not verified facts.

Current boundaries (Phases 11-12): the opt-in Phase 11 features (P7
exact-capacity ragged crop chunking; dense/P7/Paddle CUDA graph decode with
capture/update and fallback counters; optional fused residual+RMSNorm) are
synthetic-parity engine features, not real-model support and not performance
claims. Region VISION batching executes for P7 recognizers; Paddle/OpenDoc
remain per-crop; text decode is still independent batch-1 and
plan_decode_batches / max_decode_batch remain planning-only. Paged KV, native
fast paths, GLM MTP, Hunyuan DFlash, batched decode, and grouped persistent
MoE GEMM remain unsupported; the GDN/MoE/Surya executors have no CUDA graph
path. Session::create / Session::recognize fail closed with unvalidated for
every real artifact. The machine-readable support state is
`example_ocr --list-models`; the qualification matrix and the exact
synthetic/real qualification commands live in QUALIFICATION.md.

Phase 6 note: ocr/models/pp_doc_layout.* defines the PP-DocLayoutV2/V3 layout
detector contracts (SEPARATE per-version configs/tensor maps: a folded conv
backbone + a 1x1 class head (sigmoid) + a 1x1 box head), with a CPU decode
(score threshold, box decode with clamped exp log-scale, NMS AND class-aware
NMS, region scoring, stable multi-column reading order, source-coordinate
transforms, degenerate/empty handling). ocr/runtime/scheduler. is the
token-budgeted region scheduler (one overview/layout pass; crops sorted by
approximate patch/token count; vision/prefill batches packed under
max_patches_per_vision_batch / max_regions_per_batch; decode batches under
max_decode_batch; separate overview/crop/output budgets; restore to original
page/reading order) plus a bounded detector-output cache keyed by an explicit
content/geometry hash for recognition retry. ocr/runtime/layout_vlm_executor.
owns the PpDocLayoutExecutor (device conv graph), the RegionRecognizer
interface with TWO distinct implementations -- PaddleOcrVlRecognizer
(native-resolution NaViT + adaptive visual-token reduction + MLP connector +
ERNIE dense decoder) and OpenDocRecognizer (PP-DocLayoutV2 + UniRec CNN+CTC,
NOT a dense VLM) -- and the LayoutVlmExecutor full overview->layout->crop->
region vision->decode->page/document orchestration with cancellation/time/
memory checks, immediate sequence/workspace release, and a documented
cpu_reference layout option (head maps from the independent host oracle; never
a silent fallback to a different model). Host composition bridges
(paddle_ocr_vl_result., open_doc_result., layout_result.) produce strict
Markdown/JSON/plain/spotting output and the public RecognizeResult; malformed
table/formula/JSON output fails closed (no plain-text fallback). Generative
content carries NO confidence; only validated detector/CTC scores are carried,
explicitly labeled internal/test. SMALL SYNTHETIC fixtures execute end-to-end
on vk and cuda and match independent double-accumulating CPU oracles (layout
head maps + decoded boxes/scores/classes, NaViT patches + connector features +
ERNIE greedy tokens, UniRec CTC ids/text, region batching vs batch-1
equivalence, repeat-run determinism). This is a SYNTHETIC parity gate, not
real-model qualification. PaddleOCR-VL is NOT a validated complex-document
default. Session creation and recognition still fail closed with unvalidated
for every real artifact; pp_doc_layout_validate_real_artifact,
paddle_ocr_vl_validate_real_artifact, and open_doc_validate_real_artifact
ALWAYS return unvalidated.

Phase 5 note: ocr/models/pp_ocrv6.* defines the PP-OCRv6 CNN graph contracts
(LCNetV4-style detector backbone + RepLKFPN neck + DB head; LCNetV4-style
recognizer backbone + LightSVTR height-collapse + per-frame SVTR MLP + CTC),
with tiny/small/medium as SEPARATE per-size dimension/tensor-map contracts
validated independently, and inference-only Conv+BN / RepLK folding helpers
(pp_fold_conv_bn / pp_fold_replk). ocr/runtime/cnn_executor.* is a REAL
executing detector -> DB postprocess -> rotated polygon line crop ->
recognizer -> CTC pipeline with separate preprocessing/detector/crop/
recognizer stages, immutable shared weights, a per-request workspace,
component/pixel/region limits, luisa::span transfers, and source-coordinate
transforms back to ORIGINAL page pixels. ocr/ops/ops_ctc.* provides device
frame log-softmax/argmax/probs, sigmoid, height-collapse, FPN resize, and
polygon-crop kernels plus independent host oracles. ocr/output/db_postprocess.*
is a CPU DB baseline (checked binarization, 8-connected components, convex
hull + min-area rotated rect, exact convex miter unclip with the official
area*ratio/perimeter distance, box_score_fast-style scoring, optional IoU NMS,
degenerate/tiny/empty/adversarial handling) with NO OpenCV dependency; it is
explicitly NOT the official cv2.findContours/approxPolyDP/pyclipper pipeline
(see the header for the exact implemented-vs-unqualified boundary).
ocr/output/ctc_decode.* implements the DECLARED PP-OCR reference greedy
semantics (raw-sequence consecutive-duplicate collapse before blank removal,
per-class dictionary lookup incl. CJK/emoji, confidence = ARITHMETIC MEAN of
kept-frame softmax probabilities; empty decode -> has_confidence=false).
ocr/output/reading_order.* is deterministic multi-column reading order with
row-tolerance banding for skewed/rotated/wide lines, stable tie-breaking, and
page-index preservation. A SMALL SYNTHETIC per-size CNN fixture executes
end-to-end on vk and cuda and matches an independent double-accumulating CPU
oracle (folded weights vs UNFOLDED two-step conv->BN / three-branch RepLK
evaluation, detector probability map, DB polygons/scores, crop pixels,
per-frame CTC logits, collapsed ids/text/confidence, reading order), plus
blank-page (no lines, no hallucination), repeat-run determinism, wide-line,
multi-page, and limit-refusal cases. This is a SYNTHETIC parity gate, not
real-model qualification. No NRTR/dropout/loss/training branches exist.
Load-path gates: CnnExecutor::create_from_packages validates BOTH component
package tables on disk (exact names/shapes, conv_oihw for conv weights, kn
for linears, fp32-only, no unexpected tensors) before binding any weight;
installed-weight checks use the separate device-descriptor view (the runtime
maps conv_oihw -> row_major). Detector configs whose FPN lateral skip is
LARGER than the backbone output are refused before allocation (the neck only
upsamples), and non-divisible/zero height_collapse is refused.
Session creation and recognition still fail closed with unvalidated for
every real artifact; pp_ocrv6_validate_real_artifact ALWAYS returns
unvalidated.

Phase 4 note: ocr/models/qwen3_vl.* defines the DISTINCT Qwen3-VL graph
contract (vision patch embed + bidirectional QK-Norm attention + 2D vision
RoPE + GELU MLP + spatial merger; interleaved M-RoPE with explicit
temporal/height/width axis sections; Qwen3 text decoder with QK-Norm, GQA,
NO QKV bias; DeepStack feature projection + residual injection at the exact
configured early decoder layers). ocr/runtime/qwen3_vl_executor.* records
that graph from the Phase 2 operators and the shared Phase 3 dense executor
(additive Qwen3 contract switches; the Phase 3 Qwen2.5-style path is
unchanged). ocr/models/fire_red_ocr.* reuses the SAME graph only through its
own tensor map / prompt / output-parser configuration; no GRPO/training
operators exist. A SMALL SYNTHETIC Qwen3-like slice executes end-to-end on
vk and cuda and matches an independent double-accumulating CPU oracle at
every layered golden point (patches, first vision block, merger, DeepStack
projections/injections, first decoder layer, full-prefill logits, decode
KV/logits, deterministic greedy tokens). This is a SYNTHETIC parity gate,
not real-model qualification: the geometry/tensor names are a declared
synthetic contract, NOT an official checkpoint. Instruct vs Thinking is an
explicit identity boundary (Thinking fails closed). Session creation and
recognition still fail closed with unvalidated for every real artifact.

Phase 3 note: the shared dense decoder components in
ocr/models/qwen2_5_vl.* define and check recipe-driven decoder geometry and
exact tensor-name/layout expectations for Qwen2.5-VL-derived text decoders.
They are NOT a Qwen2.5-VL model and make no claim about olmOCR, TeleOCR, or
any other real model. Qwen2.5-VL and Qwen3-VL are kept distinct: the config
parser fails closed on Qwen3 markers (QK-Norm, etc.) rather than accepting a
different decoder contract, and selects the explicit rotate-half RoPE pair
layout (the Qwen2.5 HF text convention); the synthetic adjacent-pair decoder is
never described as Qwen2.5. The synthetic dense decoder reproduces
deterministic greedy tokens and reference intermediate tensors on vk and
cuda; that is a synthetic parity gate, not real-model qualification. Session
creation and recognition still fail closed with unvalidated.

Support states (plan §1):

- `planned` — contract written down; nothing implemented or verified.
- `load_only` — package loads and validates; inference not qualified.
- `experimental` — runs only under the explicit unsafe override; unvalidated.
- `validated` — qualified for its exact declared checkpoint revision, package
  version, backends, precisions, input kinds, and output modes via load-time
  validation, layered golden tests, deterministic end-to-end tests, and
  resource-limit tests.

A model is "supported" only in the `validated` state, and only for the exact
matrix row that passed. No silent substitution of another model,
architecture, prompt, positional encoding, or output parser is allowed.

## Targets (15 models, 17 selectable targets)

| Target (`--model`) | Family | Intended path & components (contract to verify) | Intended modes | Support |
|---|---|---|---|---|
| `pp_ocrv6:tiny` | CNN | LCNetV4 detector + RepLKFPN + DB head; LCNetV4/LightSVTR recognizer + CTC; size-specific shapes | plain, spotting (polygons, CTC confidence) | planned |
| `pp_ocrv6:small` | CNN | same family, separate size validation | plain, spotting | planned |
| `pp_ocrv6:medium` | CNN | same family; intended validated default for plain/spotting, NOT validated (no official checkpoint; see Phase 5 gates) | plain, spotting | planned |
| `paddle_ocr_vl_1_6` | layout VLM | PP-DocLayoutV3 + reading order/crops; native-resolution NaViT; adaptive MLP connector; ERNIE-4.5-0.3B decoder; full pipeline required (Phase 6 synthetic slice only; NOT a validated complex-document default) | plain, spotting, markdown, extract | planned |
| `ovis_ocr2` | hybrid GDN | dynamic preprocessing; vision merger; Qwen3.5-0.8B hybrid decoder (GDN + periodic full GQA) | plain, markdown | planned |
| `tele_ocr` | dense VLM | Qwen2.5-VL-derived encoder (window/global), MLP aligner, Qwen3-0.6B (QK-Norm, M-RoPE); outer `model_type` is NOT identity | plain, markdown | planned |
| `glm_ocr` | layout VLM | PP-DocLayoutV3; 0.4B CogViT; downsampling connector; GLM-0.5B decoder; MTP deferred | plain, markdown, extract | planned |
| `mineru_2_5_pro_2605_1_2b` | layout VLM | page layout; native crops; 675M NaViT; 2x2 pixel-unshuffle; Qwen2-0.5B; cross-page composition | plain, markdown, extract | planned |
| `hunyuan_ocr_1_5` | custom VLM | native Hunyuan-ViT; adaptive pool/MLP; Hunyuan-0.5B; XD-RoPE (text/height/width/time); DFlash deferred | plain, spotting, markdown, extract | planned |
| `surya_2_5` | hybrid GDN | EfficientViT-Segformer detector + crops; vision merger; 24-layer hybrid decoder (18 GDN + 6 full-attention, contract to verify) | plain, spotting, markdown | planned |
| `open_doc_0_1b` | layout VLM | PP-DocLayoutV2 + UniRec-0.1B two-stage text/formula/table recognizer (not a generic VLM); no autoregressive KV cache; UniRec-specific output contract; retains layout routing (Phase 6 synthetic two-stage slice only) | plain, markdown | planned |
| `fire_red_ocr` | dense VLM | Qwen3-VL-2B vision/merger + Qwen3 decoder; interleaved MRoPE, DeepStack, GQA, QK-Norm; no GRPO operators | plain, markdown, extract | planned |
| `qwen3_vl_2b_instruct` | dense VLM | same reusable Qwen3-VL graph; Instruct recipe (not Thinking) | plain, markdown, extract | planned |
| `deepseek_ocr_2` | MoE VLM | SAM-style local perception (windowed, separate block); BOTH convolutional downsamplers; visual LM encoder + causal-flow queries carrying the EXACT custom visual/query mask; DeepSeek MoE decoder (64 routed experts/top-6/2 shared, exact counts; hidden dims scalable) (Phase 10 synthetic slice only; real artifacts fail closed) | plain, markdown | planned |
| `dots_mocr` | dense VLM | ViT + patch merger; 28-layer dense decoder (GQA 12 query/2 KV heads, RMSNorm/RoPE/SwiGLU, contract to verify); quantized 8 GB path | plain, markdown | planned |
| `got_ocr2_0` | dense VLM | ViTDet/SAM encoder (windowed/global, relative positions); linear connector; Qwen-0.5B; region/color prompt modes; multi-crop composition | plain, markdown | planned |
| `olm_ocr_2` | dense VLM | Qwen2.5-VL-7B encoder/merger/decoder; **English-only**; INT4 + strict memory envelope or explicit host offload | plain, markdown | planned |

## Honesty rules

- No pinned hashes/revisions are invented. Recipe identity fields stay empty
  until an official artifact is inspected and pinned.
- Confidence: conventional OCR (CTC) or explicitly validated spotting/layout
  sources only. Generative VLM results carry `has_confidence == false`.
- RTX 4060 8 GB is a primary target, not a promise that every configuration
  fits; DeepSeek-OCR-2, dots.mocr, and especially olmOCR-2 require restrictive
  budgets and/or quantization. No performance claims exist yet.
- GLM MTP and Hunyuan DFlash are optional later accelerators, never
  prerequisites for correct model support.
- No weights are committed, redistributed, or downloaded at runtime. Users
  supply appropriately licensed checkpoints; license fields are unreviewed
  until pinned.
- The unsafe experimental registry override never marks anything validated
  and never enables inference by itself.
- Language hints match EXACTLY against the declared list; "multi" is a literal
  tag, not a wildcard. Capability lists (modes/languages/coordinates) may only
  expand when actually validated. Declared capability is intent, not support.
- A validated registry match requires support==validated, a fully pinned
  identity (revision + fingerprint + package format version), exact equality
  of all three, and an exact canonical tensor signature (all required/known
  names, no duplicates, no unexpected names). Anything else fails closed.

Qualification gate (still unmet for every model; see QUALIFICATION.md)

A row moves to validated only with: pinned checkpoint revision + package
version; per-backend/precision load validation; layered golden parity;
deterministic greedy end-to-end tokens; blank/hallucination cases; parser and
coordinate validation; memory/time-limit tests; and measured (not estimated)
resource reports. Missing any required row keeps the model experimental.

Phase 4 synthetic vertical slice — exact declared architecture

The synthetic Qwen3-VL slice (qwen3_vl_synthetic_config) is a DECLARED
contract, not an official checkpoint. Geometry:

- Text decoder: vocab 32, hidden 32, 3 layers, 4 query / 2 KV heads (GQA
  group 2), head_dim 8, intermediate 64, RMSNorm eps 1e-5, RoPE theta 1e4,
  max positions 128, EOS 31. QK-Norm (per-head RMSNorm over head_dim) on Q/K,
  NO QKV projection bias, SwiGLU MLP, tied embeddings. Interleaved M-RoPE:
  pairs (d, d+head_dim/2), axis sections {temporal 2, height 1, width 1}.
- Vision: 2 layers, hidden 16, 2 heads, head_dim 8, intermediate 32, patch 2,
  temporal_patch 1, RGB, spatial_merge 2, merger intermediate 32, LayerNorm
  eps 1e-5, RoPE theta 1e4, 2D axial RoPE (adjacent pairs) with 2 height pairs
  + 2 width pairs. Bidirectional attention WITHIN each image (ragged images
  never attend across each other). GELU-erf MLP.
- DeepStack: 2 injections, vision layers {0,1} -> decoder layers {0,1}; each
  injection adds its projected feature to the image token rows AFTER the
  selected decoder layer's residual stream.
- Merger: spatial merge (2x2) -> LayerNorm over merged dim 64 -> fc1 (GELU-erf,
  64->32) -> fc2 (32->32 text hidden).
- Image packing: each source edge scaled uniformly into [min_pixels,
  max_pixels] then rounded to a positive multiple of patch*merge (=4);
  source transform maps preprocessed pixels back to source pixels.

Synthetic parity tolerances (FP32 device vs double-accumulating CPU oracle,
operation-appropriate, plan §10.1): linear/norm/patchify stages 5e-4 +
5e-4*|ref|; stages including the GELU-erf approximation 3e-3 + 3e-3*|ref|.
Deterministic greedy tokens must match the oracle EXACTLY (integer ids) and
repeat-run identically. These are synthetic gates, not official quality
metrics; no CER/IoU/confidence calibration is claimed.

Unmet OFFICIAL gates (Qwen3-VL-2B-Instruct and FireRed-OCR)

Both models remain planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:

1. No pinned official checkpoint revision, config fingerprint, or package
   hash. The synthetic tensor names/geometry are a declared contract, not the
   official checkpoint's; the official tensor map must be verified against a
   real artifact before any real package loads.
2. No official processor parity: the synthetic resize/normalize statistics
   (mean/std 0.5) and packing convention are declared, not verified against
   the official Qwen3-VL image processor.
3. No official chat-template parity: the qwen3_vl_ocr / fire_red_ocr prompt
   layouts are synthetic contracts, not verified against the official
   apply_chat_template. Instruct vs Thinking is enforced as an identity
   boundary, but neither official template is pinned.
4. No official layered goldens: parity is against an independent synthetic
   CPU oracle only, not against official reference intermediate tensors.
5. No official output-parser qualification: the Markdown/HTML/OTSL/LaTeX/JSON
   parsers are strict and host-tested, but not validated against official
   FireRed-OCR reference outputs.
6. No production precision matrix (FP32/BF16/FP16/W4) on the RTX 4060: the
   slice is FP32-only; no quantized production claim, no measured
   performance/memory numbers.
7. No blank/hallucination qualification for a REAL model: the synthetic
   blank/repeated/adversarial tests exercise the implemented synthetic graph
   and strict parsers honestly; they do NOT establish hallucination safety
   for any real VLM, and no VLM confidence is produced.
8. FireRed-OCR real artifacts fail closed unconditionally
   (fire_red_ocr_validate_real_artifact returns unvalidated); only the
   synthetic recipe-driven path is exercised. No GRPO/training nodes exist.

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.

Phase 7 synthetic custom dense/layout vertical slices — exact declared architecture

The Phase 7 models (TeleOCR, GLM-OCR, MinerU, HunyuanOCR) share ONE configurable
slice executor (runtime/p7_vlm_executor.*) whose architecture distinctions are
EXPLICIT configuration (runtime/p7_slice_config.h), never inferred from an outer
model_type string. Each model module owns its identity/tensor-name/output
contract and instantiates the shared graph. All are SYNTHETIC contracts; real
artifacts fail closed through the per-model real-artifact gates.

TeleOCR (tele_ocr_synthetic_config) — DECLARED contract, not an official checkpoint:
- Vision: Qwen2.5-VL-derived window/global tower. 2 layers, hidden 16, 2 heads,
  head_dim 8, inter 32, patch 2; QKV bias, NO vision QK-Norm, 2D axial RoPE
  (2 height + 2 width pairs), GELU-erf MLP. Window layers attend within a 2x2
  patch tile (per-patch custom block mask, block_size 1); layer 1 is GLOBAL
  (single all-ones block). The window-vs-global distinction is device-tested.
- Aligner: spatial 2x2 token packing + LayerNorm + 2-layer GELU MLP (NOT
  Paddle's adaptive pool, NOT MinerU's pixel-unshuffle).
- Text: Qwen3-0.6B-style decoder. vocab 32, hidden 32, 3 layers, 4 Q / 2 KV
  heads, head_dim 8, inter 64; QK-Norm, NO QKV bias, GQA, INTERLEAVED M-RoPE
  (rotate-half pairs, sections {2,1,1}). A Qwen2 decoder is NEVER substituted.
- Optional layout/crop route via the shared LayoutVlmExecutor (PP-DocLayoutV3).
- Camera/deformed input: curved/folded/crumpled metadata + a deterministic
  nearest-neighbor geometric warp (host-tested against an independent oracle).
- Identity: the outer model_type is NEVER identity; tele_ocr_validate_signature
  requires checkpoint revision + config fingerprint + the exact canonical
  tensor-name set. parse_tele_ocr_config refuses a Qwen2 (qkv_bias) decoder.
- Output: plain + markdown only (strict per-type composition); spotting/extract
  fail closed (not validated for TeleOCR). Generative content: no confidence.

Unmet OFFICIAL gates (TeleOCR)

TeleOCR remains planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:
1. No pinned official TeleOCR checkpoint revision, config fingerprint, or
   package hash. The synthetic tensor names/geometry are a declared contract.
2. No official window/global attention schedule verification: the window size,
   global-layer indices, and mask granularity are declared synthetic values,
   not verified against the official TeleOCR/Qwen2.5-VL-derived encoder.
3. No official processor parity: the stretch-resize/normalize statistics and
   packing are declared, not verified against the official TeleOCR processor.
   The curved/folded/crumpled warp is a synthetic deterministic remap, NOT the
   official dewarping pipeline; no official deformed-document accuracy exists.
4. No official Qwen3-0.6B decoder parity: QK-Norm/M-RoPE sections are declared
   synthetic values, not verified against the official checkpoint.
5. No official layered goldens: parity is against independent synthetic CPU
   oracles only (device cuda: 169 asserts / 7 tests pass), not official
   reference intermediate tensors.
6. No official determinism/accuracy qualification for a REAL model; no CER/
   edit-distance, reading-order, or coordinate accuracy on any benchmark.
7. No production precision matrix (FP32-only baseline) and no measured RTX
   4060 8 GB performance/memory numbers.
8. Real artifacts fail closed unconditionally (tele_ocr_validate_real_artifact
   returns unvalidated); only the synthetic path is exercised. No runtime
   downloads exist.

GLM-OCR (glm_ocr_synthetic_config) — DECLARED contract, not an official checkpoint:
- Front end: PP-DocLayoutV3 (the shared layout detector + reading order).
- Vision (CogViT): 2 layers, hidden 16, 2 heads, head_dim 8, inter 32, patch 2;
  learned ABSOLUTE position table (1024 rows), NO vision RoPE, QKV bias, NO
  QK-Norm, bidirectional attention, GELU-erf MLP.
- Connector: learned strided downsampling (spatial 2x2 gather -> learned linear
  projection) + LayerNorm + 2-layer GELU MLP. NOT Paddle's adaptive pool, NOT
  MinerU's pixel-unshuffle, NOT TeleOCR's plain spatial-merge aligner.
- Text (GLM-0.5B-style): vocab 32, hidden 32, 3 layers, 4 Q / 2 KV heads,
  head_dim 8, inter 64; PARTIAL rotary factor 1/2 (rotate-half pairs within the
  first head_dim/2 = 4 rotated elements; inv_freq over the rotated dim; the
  remaining 4 elements pass through), QKV bias, NO QK-Norm, GQA, RMSNorm,
  SwiGLU, plain 1D text positions.
- Ordinary target decoding ONLY. MTP draft/target verification is DEFERRED;
  parse_glm_ocr_config refuses a positive mtp_num_layers requirement.
- Fail-closed identity: a Qwen model_type, Qwen3 qk_norm, full-rotary factor
  1.0, mrope_section, or MoE/GDN fields are all refused.
- Output: plain + markdown + extract (strict); spotting fails closed (not
  validated). Generative content: no confidence.

Unmet OFFICIAL gates (GLM-OCR)

GLM-OCR remains planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:
1. No pinned official GLM-OCR checkpoint revision, config fingerprint, or
   package hash. The synthetic tensor names/geometry are a declared contract.
2. No official CogViT parity: the learned-position table size, patch embed, and
   block conventions are declared synthetic values, not verified against the
   official 0.4B CogViT.
3. No official downsampling-connector parity: the stride/projection layout is a
   declared synthetic convention, not verified against the official connector.
4. No official GLM-0.5B decoder parity: the partial-rotary factor, pair family,
   and layer conventions are declared synthetic values, not verified against
   the official GLM decoder. The exact declared GLM position/layer conventions
   remain an unmet official gate.
5. No official processor parity (stretch-resize/normalize statistics declared).
6. No official layered goldens: parity is against independent synthetic CPU
   oracles only (device cuda: 143 asserts / 5 tests pass), not official
   reference intermediate tensors.
7. No official determinism/accuracy qualification for a REAL model; no CER/
   table/JSON/reading-order accuracy on any benchmark.
8. No production precision matrix (FP32-only) and no measured RTX 4060 8 GB
   performance/memory numbers.
9. MTP is deferred and unimplemented; ordinary target decode is the only path.
   Real artifacts fail closed unconditionally (glm_ocr_validate_real_artifact
   returns unvalidated). No runtime downloads exist.

MinerU (mineru_synthetic_config) — DECLARED contract, not an official checkpoint:
- Front end: overview page-layout pass + native-resolution region crops (the
  shared LayoutVlmExecutor).
- Vision (675M-NaViT-shaped): 2 layers, hidden 16, 2 heads, head_dim 8, inter
  32, patch 2; QKV bias, NO QK-Norm, 2D axial RoPE (2 height + 2 width pairs),
  GELU-erf MLP, bidirectional attention within one image.
- Connector: 2x2 PIXEL-UNSHUFFLE (token-major vision output transposed to
  channel-major NCHW, pixel_unshuffle, transposed back to token rows; token
  channel order (ci, ry, rx) matching ref_pixel_unshuffle) + LayerNorm +
  2-layer GELU MLP. NOT Paddle's adaptive-pool connector.
- Text (Qwen2-0.5B-style): vocab 32, hidden 32, 3 layers, 4 Q / 2 KV heads,
  head_dim 8, inter 64; QKV bias, NO QK-Norm, GQA, RMSNorm, SwiGLU,
  ADJACENT-pair M-RoPE (Qwen2-VL style, sections {2,1,1}; NOT Qwen3 interleaved).
- Reading-order restoration + CROSS-PAGE composition (pages ascending by source
  page_index, reading order within each page; device-tested with reversed input).
- Output: plain + markdown + extract (strict); spotting fails closed (not
  validated). Generative content: no confidence.

Unmet OFFICIAL gates (MinerU)

MinerU remains planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:
1. No pinned official MinerU2.5-Pro checkpoint revision, config fingerprint, or
   package hash. The "675M NaViT" / "Qwen2-0.5B" labels name the intended real
   components, not a qualified checkpoint; the synthetic geometry is declared.
2. No official NaViT parity: the native-resolution packing, 2D-RoPE sections,
   and block conventions are declared synthetic values.
3. No official pixel-unshuffle channel-ordering verification: the (ci, ry, rx)
   token channel order is the declared synthetic convention (matching the
   Phase 2 ref_pixel_unshuffle), not verified against the official MinerU
   connector. The official ordering is an unmet gate.
4. No official Qwen2-0.5B decoder parity: the adjacent-MRoPE sections and layer
   conventions are declared synthetic values.
5. No official processor parity (stretch-resize/normalize statistics declared).
6. No official layered goldens: parity is against independent synthetic CPU
   oracles only (device cuda: 465 asserts / 5 tests pass), not official
   reference intermediate tensors.
7. No official determinism/accuracy qualification for a REAL model; no CER/
   table/formula/reading-order/cross-page accuracy on any benchmark.
8. No production precision matrix (FP32-only) and no measured RTX 4060 8 GB
   performance/memory numbers.
9. Real artifacts fail closed unconditionally (mineru_validate_real_artifact
   returns unvalidated). No runtime downloads exist.

HunyuanOCR (hunyuan_ocr_synthetic_config) — DECLARED contract, not an official checkpoint:
- Vision (native Hunyuan-ViT): 2 layers, hidden 16, 2 heads, head_dim 8, inter
  32, patch 2; QKV bias, NO QK-Norm, 2D axial RoPE (2 height + 2 width pairs),
  SiLU MLP (the Hunyuan vision activation), bidirectional attention.
- Connector: ADAPTIVE average pool to a token-budget grid (<= 16 visual tokens)
  + LayerNorm + 2-layer GELU MLP. NOT MinerU's pixel-unshuffle, NOT GLM's
  learned downsampling.
- Text (Hunyuan-0.5B-style): vocab 32, hidden 32, 3 layers, 4 Q / 2 KV heads,
  head_dim 8, inter 64; QKV bias, NO QK-Norm, GQA, RMSNorm, SwiGLU, XD-RoPE
  over FOUR axes (text, height, width, time), adjacent pairs, sections
  {1,1,1,1}. A Qwen M-RoPE decoder is never substituted.
- Ordinary target decoding ONLY. DFlash is DEFERRED; parse_hunyuan_ocr_config
  refuses a positive dflash_num_layers requirement.
- Output adapters: plain + markdown + extract (strict) + a spotting adapter
  that carries source coordinates from the layout regions. Translation is NOT
  implemented and fails closed. NO fake confidence: generative content carries
  has_confidence == false (device-tested).

Unmet OFFICIAL gates (HunyuanOCR)

HunyuanOCR remains planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:
1. No pinned official HunyuanOCR-1.5 checkpoint revision, config fingerprint,
   or package hash. The synthetic tensor names/geometry are a declared contract.
2. No official Hunyuan-ViT parity: the SiLU MLP, 2D-RoPE sections, and block
   conventions are declared synthetic values, not verified against the official
   native Hunyuan-ViT.
3. No official adaptive-pool/MLP connector parity: the token-budget grid and
   pool convention are declared synthetic values.
4. No official Hunyuan-0.5B decoder parity: the XD-RoPE axis sections and layer
   conventions are declared synthetic values, not verified against the official
   decoder. The exact XD-RoPE text/height/width/time semantics remain an unmet
   official gate.
5. No official processor parity (stretch-resize/normalize statistics declared).
6. No official layered goldens: parity is against independent synthetic CPU
   oracles only (device cuda: 278 asserts / 5 tests pass), not official
   reference intermediate tensors.
7. No official spotting/extraction/translation qualification: the spotting
   adapter carries layout source coordinates (synthetic), not validated
   official spotting confidence/coordinates; translation is unimplemented.
8. No official determinism/accuracy qualification for a REAL model; no CER/
   spotting/JSON accuracy on any benchmark.
9. No production precision matrix (FP32-only) and no measured RTX 4060 8 GB
   performance/memory numbers.
10. DFlash is deferred and unimplemented; ordinary target decode is the only
    path. Real artifacts fail closed unconditionally
    (hunyuan_ocr_validate_real_artifact returns unvalidated). No runtime
    downloads exist.

Phase 8 synthetic remaining dense ViT vertical slices — exact declared architecture

The Phase 8 models (dots.mocr, GOT-OCR2.0, olmOCR-2) reuse the shared dense
executor (runtime/dense_vlm_executor.*) and the configurable Phase 7 slice
(runtime/p7_vlm_executor.*) with ADDITIVE contract switches. Each model module
owns its identity/tensor-name/output contract. All are SYNTHETIC contracts; real
artifacts fail closed through the per-model real-artifact gates. No official
checkpoint has been inspected, pinned, or run; no RTX 4060 performance claim is
made.

Additive engine support (old behavior unchanged; all prior suites stay green):
- DenseLinearStorage (models/qwen2_5_vl.h): an explicit per-linear weight storage
  policy. fp32 is the default (exact Phase 3/4/7 path). int4_sym / int4_asym
  select the W4A32 path: PACKED grouped-INT4 code reads with FP32 activations and
  FP32 accumulation (Phase 1/2 semantics, dequantization inside the GEMM/GEMV
  kernels). Embeddings, norms, and QKV biases stay FP32; a quantized recipe ships
  a standalone packed lm_head (tied embeddings are refused for W4A32). This is
  NEVER a relabeled FP32 path: the packed words + per-group scales/zero-points
  are read on device. FP16/BF16 arithmetic and W4A16 remain refused.
- A named Qwen2.5-VL rotate-half M-RoPE variant (make_rope_mrope_rotate_half_kernel)
  distinct from the Qwen3 interleaved-MRoPE identity; selected by the explicit
  text_qwen2_5_contract switch (QKV bias + no QK-Norm + rotate-half M-RoPE).
- Vision decomposed relative-position bias (P7VisionConfig.rel_pos_window): a
  learned per-head table [heads, 2w-1, 2w-1] expanded host-side to [heads, rows,
  rows] and added to the attention scores (ViTDet/SAM). DISTINCT from RoPE and
  from absolute position embeddings; never substituted for them.
- Connector RMSNorm switch (P7ConnectorConfig.rms_norm): the Qwen2.5-VL merger
  convention (weight only, no bias), distinct from the Phase 7 LayerNorm default.

dots.mocr (dots_mocr_synthetic_config) — DECLARED contract, not an official checkpoint:
- Vision: 2-layer ViT, hidden 16, 2 heads, head_dim 8, inter 32, patch 2; QKV
  bias, NO QK-Norm, 2D axial RoPE (2 height + 2 width pairs), bidirectional,
  GELU-erf MLP. Connector: spatial-merge 2 patch merger + LayerNorm + 2-layer
  GELU MLP.
- Decoder: the FULL contract is 28 layers, 12 query / 2 KV heads (GQA), RMSNorm,
  1D RoPE (rotate-half pairs), SwiGLU. The executing synthetic slice uses a
  SCALED geometry (3 layers, 4 Q / 2 KV, head_dim 8, hidden 32) with the SAME
  conventions; the 28/12/2 contract is validated EXPLICITLY
  (dots_mocr_declared_decoder_config + the contract test). The declared full
  geometry is hidden 1536 / head_dim 128 / inter 8960 / vocab 151936.
- W4A32 quantized path on the memory-constrained configuration: packed INT4
  decoder linears + standalone packed lm_head, FP32 activations/accumulation.
  The quantized arm is compared against a DEQUANTIZED-weight oracle (not the
  FP32 arm); the FP32-vs-quantized logit delta is reported as a synthetic
  quality measure (not official accuracy).
- Memory preflight (dots_mocr_preflight_decoder): checked decoder weight +
  workspace + KV bytes, required-vs-available report, fails BEFORE allocation.
- Output: plain + markdown only; spotting/extract fail closed. No confidence.

GOT-OCR2.0 (got_ocr2_synthetic_config) — DECLARED contract, not an official checkpoint:
- Vision: 2-layer ViTDet/SAM, hidden 16, 2 heads, head_dim 8, inter 32, patch 2;
  WINDOW/GLOBAL attention (window 2 patches, global layer {1}), decomposed
  RELATIVE-POSITION bias (window 4), NO vision RoPE, QKV bias, NO QK-Norm,
  GELU-erf MLP. The relative bias is device-tested to change the output (a
  zero-bias tower differs from the learned-bias tower).
- Connector: a plain LINEAR projection (LayerNorm + 2-layer GELU MLP, 16 -> 32,
  no spatial reduction).
- Decoder: Qwen-0.5B-style, 3 layers, 4 Q / 2 KV, head_dim 8, hidden 32; QKV
  bias, NO QK-Norm, 1D RoPE (rotate-half), SwiGLU.
- Prompt modes: plain ("OCR: "), formatted ("OCR with format: "), region
  ("OCR with region: [x0,y0,x1,y1]" normalized), color ("OCR with color: <name>").
  Exact prompt/parser round-trip; invalid guidance/mode is REFUSED (never a
  silent plain fallback). Source region transforms map normalized boxes to
  source pixels (clamped, degenerate refused).
- Multi-crop composition: crops composed in reading order with source boxes
  preserved; formatted mode validates Markdown strictly (malformed -> error).
- Output: plain + markdown; region/color are prompt modes, not composition
  modes. No confidence.

olmOCR-2 (olm_ocr2_synthetic_config) — DECLARED contract, not an official checkpoint:
- Vision: 2-layer Qwen2.5-VL-style window/global tower, hidden 16, 2 heads,
  head_dim 8, inter 32, patch 2; 2D axial RoPE, QKV bias, NO QK-Norm, GELU-erf
  MLP. Merger: spatial-merge 2 + RMSNorm (Qwen2.5-VL convention) + 2-layer GELU
  MLP.
- Decoder: the FULL contract is Qwen2.5-VL-7B (28 layers, 28 Q / 4 KV, hidden
  3584, head_dim 128, inter 18944, vocab 152064), QKV bias, NO QK-Norm,
  rotate-half M-RoPE (sections {t,h,w}), SwiGLU. The executing synthetic slice
  uses a SCALED geometry (3 layers, 4 Q / 2 KV, head_dim 8, hidden 32) with the
  SAME Qwen2.5 contract. The declared 7B geometry is a CONTRACT, NOT a claim
  that a real 7B model fits in 8 GB.
- ENGLISH-ONLY: olm_ocr2_check_language accepts "en"/empty and refuses every
  other language (including "multi"); a non-English config language list is
  refused at parse time. No bilingual claim.
- The declared 8 GB path REQUIRES INT4 (require_int4): FP32 is refused for that
  path (allowed only for the small synthetic parity slice). Strict memory
  preflight (olm_ocr2_preflight) reports vision + decoder + workspace + KV bytes
  and fails BEFORE allocation with a required-vs-available report. The declared
  7B INT4 footprint at 32768 context exceeds 8 GiB and is refused (measured
  preflight arithmetic, NOT an estimate-based fit claim).
- Host offload (explicit_host) is NOT implemented and fails closed as
  unsupported BEFORE any allocation/accounting, with no performance claim.
- Output: plain + markdown (English-only); spotting/extract fail closed. No
  confidence.

Synthetic parity tolerances (FP32 device vs double-accumulating CPU oracle,
operation-appropriate, plan §10.1): linear/norm/patchify stages 5e-4 +
5e-4|ref|; vision/connector/decoder/logits/KV stages 3e-3 + 3e-3|ref| (GELU-erf
approximation); the W4A32 quantized arm vs the dequantized-weight oracle 3e-3 +
3e-3|ref|. Deterministic greedy tokens must match the oracle EXACTLY (integer
ids) and repeat-run identically. These are synthetic gates, not official quality
metrics; no CER/IoU/confidence calibration is claimed.

Unmet OFFICIAL gates (dots.mocr, GOT-OCR2.0, olmOCR-2)

All three models remain planned. The following official gates are NOT met and no
official artifact has been inspected, pinned, or run:
1. No pinned official checkpoint revision, config fingerprint, or package hash
   for any of the three models. The synthetic tensor names/geometry are declared
   contracts, not the official checkpoints'; the official tensor maps must be
   verified against real artifacts before any real package loads.
2. No official processor parity: the synthetic resize/normalize statistics
   (mean/std 0.5) and packing conventions are declared, not verified against the
   official dots.mocr / GOT-OCR2.0 / olmOCR-2 processors.
3. No official graph parity: the dots.mocr 28/12/2 geometry, the GOT ViTDet/SAM
   window schedule + relative-position table size, and the olmOCR-2 Qwen2.5-VL-7B
   dimensions are declared synthetic contracts, not verified against the official
   reference implementations. The exact official relative-position / M-RoPE
   semantics remain unmet gates.
4. No official chat-template parity: the GOT prompt strings and the dots/olm
   prompt layouts are synthetic contracts, not verified against the official
   apply_chat_template.
5. No official layered goldens: parity is against independent synthetic CPU
   oracles only, not official reference intermediate tensors.
6. No official output-parser qualification: the strict Markdown/composition
   parsers are host-tested but not validated against official reference outputs.
7. No production precision matrix on the RTX 4060 and no measured performance/
   memory numbers. The W4A32 path is exercised on the SMALL SYNTHETIC slice only;
   the declared 7B/28-layer memory reports are PREFLIGHT ARITHMETIC (required
   bytes), NOT measured resident memory, and NOT a claim that a real model fits.
8. No blank/hallucination qualification for a REAL model: the synthetic
   blank/repeated/malformed tests exercise the implemented synthetic graphs and
   strict parsers honestly; they do NOT establish hallucination safety for any
   real VLM, and no VLM confidence is produced.
9. olmOCR-2 host offload is unimplemented (fails closed unsupported); no 8 GB
   offload performance claim exists. English-only is enforced; no bilingual
   support is claimed.
10. Real artifacts fail closed unconditionally (dots_mocr_validate_real_artifact,
    got_ocr2_validate_real_artifact, olm_ocr2_validate_real_artifact all return
    unvalidated); only the synthetic recipe-driven paths are exercised. No
    runtime downloads exist.

Phase 8 verification results (cuda; xmake AND CMake; debug; post-review)

- New device suites (identical counts in both build systems):
  test_ocr_model_dots_mocr: 6 tests / 113 asserts;
  test_ocr_model_got_ocr2: 8 tests / 143 asserts;
  test_ocr_model_olm_ocr2: 5 tests / 109 asserts. Total 19 tests / 365 asserts.
- Review fixes (S1-S4) landed with tests: S1 quantizable-linears are restricted
  to the decoder prefix + lm_head (visual.* can never qualify); the package
  load gates (dots_mocr_validate_packages / olm_ocr2_validate_packages) are
  wired into the load path BEFORE upload and tested (vision FP32 + decoder
  INT4 accepted; visual INT4 refused; decoder-linear-left-FP32 refused;
  unexpected names refused). S2 measured weight residency includes the
  executor-owned tied-head transpose (store bytes alone omit it); preflight
  weight/workspace/KV plans are asserted EQUAL to measured device bytes.
  S3 the packed linear resolver defensively checks packed-word / scale /
  zero-point buffer counts before binding (mirroring the FP32 dense-size
  gate); a malformed-DeviceWeight negative test is not reachable through the
  public WeightStore API (batches are built from validated packages), so the
  gate is defense-in-depth, documented as such. S4 olmOCR-2 `languages`
  present-but-non-array (scalar/object/null) is refused; non-string entries
  refused; tested. GOT extras: got_ocr2_preflight memory report (asserted
  equal to measured arenas/KV), page_index carried through multi-crop
  composition (pages ascending, reading order within page), and
  locale-independent d.dddd region coordinate format/parse (integer math;
  comma decimals/exponents/signs/wrong precision refused; tested).
- Affected prior suites re-run green on cuda in BOTH build systems (24 suites:
  decode prefill/cache/sampling/stop, qwen3_vl, fire_red, ops rope/gemm/
  attention/norm/vision, layout, paddle_ocr_vl, open_doc, region_scheduler,
  tele_ocr, glm_ocr, mineru, hunyuan, pp_ocrv6, runtime_device, plus host
  decoder_config/model_contract/memory_budget). The Phase 7 four model suites
  remain exactly 187+167+465+300 = 1119 asserts. All 26 host suites pass
  (CTest -L unit_ocr: 26/26; the same binaries under xmake: 26/26).
- vk remains upstream blocked (the vk backend aborts inside its own
  debug-utils validation callback at device creation; the same blocker
  documented for Phase 6). No backend edits were made; cuda is the verified
  backend for Phase 8.
- MEASURED synthetic memory (actual device buffer bytes on cuda, NOT model
  estimates; small slices, max_vision_patches 4096, prompt 6 + 8 decode;
  S2-corrected: weights = store bytes + executor-owned tied-head transpose):
  dots.mocr FP32: weights 152128 B (store 148032 + owned 4096), vision arena
  22316544 B, decoder arena 478976 B, KV 5392 B; total 22953040 B (~21.9 MiB).
  olmOCR-2 FP32: weights 151872 B (store 147776 + owned 4096), vision arena
  22316544 B, decoder arena 481024 B, KV 5392 B; total 22954832 B (~21.9 MiB).
  GOT-OCR2.0 FP32 (prompt 18 + 8 decode, rel-bias arena): weights 146384 B
  (store 142288 + owned 4096), combined arenas 157406208 B, KV 10000 B;
  total 157562592 B (~150.3 MiB; the rel-bias scratch is heads*4096^2 floats,
  a declared worst-case capacity bound). The host preflight plans are
  asserted EQUAL to the actual executor weight/arena/KV bytes.
- dots.mocr W4A32 quality (synthetic random weights, group 128, FP32
  activations/accumulation, packed INT4 reads verified against a dequantized
  oracle at 3e-3+3e-3|ref|): FP32-vs-INT4 last-row logits max_abs delta
  2.1151211, max per-element rel delta 6.198368 (near-zero logits), relative
  L2 error 0.4100526; declared synthetic gate rel-L2 < 0.50. This is a
  synthetic quantization-noise measurement on random small-scale weights,
  NOT official accuracy; greedy decode tokens match the dequantized oracle
  exactly and repeat-run identically.
- olmOCR-2 declared 7B INT4 PREFLIGHT ARITHMETIC at 32768 context (required
  bytes, NOT measured residency, NOT a fit claim): total 9862594128 B
  (~9.19 GiB) = vision 500288 + decoder 5937412096 + workspace 166585344 +
  KV 3758096400. This EXCEEDS 8 GiB (8589934592 B) and is refused BEFORE any
  allocation with the required-vs-available report. No claim that a real 7B
  model fits or runs is made; host offload is unimplemented (unsupported).

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.

Phase 5 synthetic CNN pipeline — exact declared architecture

The synthetic PP-OCRv6 fixture (pp_ocrv6_synthetic_config) is a DECLARED
contract, not an official checkpoint. Per-size geometry (each size validated
independently; nothing shared blindly):
- Channel base: tiny=4, small=6, medium=8. Detector input: 64/96/128 square.
  Recognizer line-crop input: 64/96/128 wide x 16 high.
- Detector backbone (3 stages): 3x3 stride-(2,2) conv (3->base, ReLU);
  asymmetric 3x3 stride-(2,1) DEPTHWISE conv (base->base, ReLU); large 5x5
  stride-(1,1) DEPTHWISE conv (base->2*base, hard-swish). Total stride (4,2).
- RepLKFPN neck: lateral 1x1 conv on the stage-1 skip (base->base); bilinear
  upsample (half-pixel, align_corners=false) to the backbone resolution;
  channel concat (2*base + base); folded RepLK 7x7 conv (3*base->2*base,
  ReLU) merged from a large 7x7 branch + a center-padded 3x3 branch + an
  identity-BN impulse branch (per-output-channel identity on input channel
  oc % in_c — a declared synthetic convention).
- DB head: 1x1 conv (2*base->1) + sigmoid probability map.
- Recognizer backbone (3 stages): 3x3 stride-(2,2) conv (3->base, ReLU);
  asymmetric 3x3 stride-(2,1) conv (base->2*base, ReLU); 3x3 stride-(2,1)
  DEPTHWISE conv (2*base->2*base, hard-swish). Feature height 2, width
  input/2 (32/48/64 frames).
- LightSVTR: height-collapse reshape (r=2) [1,c,2,W] -> [W, 2c]
  (collapsed channels 16/24/32); per-frame SVTR MLP fc1 (GELU-erf,
  collapsed->4*base) + fc2 (bias, 4*base->collapsed); CTC linear head
  collapsed->16 classes (blank id 0; 15 non-blank entries incl. CJK 中/文
  and emoji 😀).
- Preprocessing: STRETCH-resize to the fixed detector input (independent x/y
  scales, bilinear half-pixel, NO aspect preservation and NO padding). This
  is a synthetic convention, NOT official PP-OCRv6 preprocessing parity.
- Recognizer frames = feature_width * (feature_height / height_collapse);
  the height-collapse factor is part of the frame count (arena/GEMM sizing
  and the device kernel agree).
- Result counters: `output_tokens` is the CTC KEPT-TOKEN count (one per
  dictionary entry emitted after collapse), NOT a UTF-8 byte count; lines
  without a CTC count contribute 0 (unknown). CJK/emoji are one token each.
- Fixture weights: deterministic LCG; box-filter conv init (1/(icg*kh*kw) +
  small jitter) with near-identity BN (a well-conditioned magnitude-
  propagating synthetic init); the DB head is calibrated to a luminance
  detector (gain on channel 0, identity BN, bias shift placing the class
  median midpoint at the binarize threshold) so the two bright test bars
  are detected deterministically. The SAME transformed weights feed both
  the device path and the oracle; calibration is fixture design, not a
  model claim.

Synthetic parity tolerances (FP32 device vs double-accumulating CPU oracle,
operation-appropriate, plan §10.1): detector backbone/fused/RepLK/probability
stages 3e-3 + 3e-3*|ref| (multi-stage FP32 conv chain); host folded-vs-
unfolded RepLK check 1e-3 + 1e-3*|ref|; folding-math unit gates 1e-4/1e-5;
CTC logits 3e-3 + 3e-3*|ref| (declared portable GELU-erf on both arms);
DB polygon coordinates 0.75 source px, scores 5e-3; collapsed text/ids must
match EXACTLY; confidence 2e-3; repeat runs must be bit-identical. These are
synthetic gates, not official quality metrics.

Unmet OFFICIAL gates (PP-OCRv6 tiny/small/medium)

All three sizes remain planned. The following official gates are NOT met and
no official artifact has been inspected, pinned, or run:
1. No pinned official checkpoint revision, config fingerprint, or package
   hash for any size. The synthetic tensor names/geometry are a declared
   contract, not the official checkpoints'; the official tensor maps must be
   verified against real artifacts before any real package loads.
2. No official preprocessing parity: the synthetic resize/normalize
   statistics (mean/std 0.5, bilinear half-pixel) and the fixed detector/
   recognizer input sizes are declared, not verified against the official
   PP-OCRv6 processors (aspect resize/pad buckets, per-size line heights).
3. No official polygon qualification: the DB baseline is a correct
   connected-component/convex-box implementation, NOT the official
   findContours/approxPolyDP/minAreaRect + pyclipper pipeline; no official
   polygon IoU / detection recall/precision is measured.
4. No official CER/edit-distance or line recall/precision on any benchmark;
   no scene-text vs document separation; no official accuracy claim.
5. No official confidence qualification/calibration: the CTC confidence is
   the declared reference arithmetic-mean formula, tested against the
   synthetic oracle only; no calibration against official outputs exists.
6. No official layered goldens: parity is against an independent synthetic
   CPU oracle only, not official reference intermediate tensors.
7. No production precision matrix (FP32-only baseline; no FP16/BF16/W4
   claim) and no measured RTX 4060 8 GB performance/memory numbers (cold/
   warm load, pages/second, resident weights, peak workspace, H2D/D2H bytes
   all unmeasured).
8. No blank/hallucination qualification for a REAL model: the synthetic
   blank/repeated/wide tests exercise the implemented synthetic graph
   honestly; they do not establish real-model hallucination safety.
9. PP-OCRv6_medium is NOT a validated default: the plan's Phase 5 exit gate
   (medium validated for plain text/spotting, tiny/small validated
   separately) is UNMET until official artifacts pass the gates above.
   pp_ocrv6_validate_real_artifact returns unvalidated unconditionally;
   Session::create/recognize fail closed for every real artifact; no runtime
   downloads exist.

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.
Phase 6 synthetic layout/region vertical slices — exact declared architecture

The Phase 6 fixtures are DECLARED synthetic contracts, not official checkpoints.

PP-DocLayoutV2/V3 layout detector (pp_doc_layout_synthetic_config):
- V2: input 64x64, channel base 4, 5 classes (text/title/table/formula/figure).
  V3: input 96x96, channel base 6, 8 classes (+header/footer/list). The two
  versions are SEPARATE contracts (different input sizes, channel widths, class
  lists, and class-head tensor shapes); nothing is shared blindly.
- Backbone: 3x3 stride-2 conv (3->base, ReLU) then 3x3 stride-2 depthwise conv
  (base->2base, ReLU). Total stride 4. Class head: 1x1 conv (2base->num_classes)
  + sigmoid. Box head: 1x1 conv (2base->4) (dx, dy, log_w, log_h).
- CPU decode: per-cell best class (lowest index wins ties), score threshold,
  box center ((cx+0.5+dx)*stride, (cy+0.5+dy)*stride), size stride*exp(clamp(log,
  -cap,cap)), source transform, NMS / class-aware NMS (score-descending, stable
  index tie-break), region cap, min-box-side rejection, stable multi-column
  reading order (reuses output/reading_order). Empty/all-background maps yield
  zero regions; non-finite probabilities/deltas fail closed.

PaddleOCR-VL region recognizer (paddle_ocr_vl_synthetic_config):
- NaViT vision: 2 layers, hidden 16, 2 heads, head_dim 8, intermediate 32,
  patch 2, RGB, LayerNorm eps 1e-5, RoPE theta 1e4, 2D axial RoPE (adjacent
  pairs) with 2 height + 2 width pairs. QKV WITH bias, NO QK-Norm, NO DeepStack
  (distinct from Qwen3-VL). Bidirectional attention WITHIN each image. GELU-erf
  MLP.
- Adaptive visual-token reduction: adaptive average pool over the patch grid to
  a token-budget grid (<= 16 visual tokens, aspect-preserving floor, >= 1 per
  axis). MLP connector: LayerNorm over vision hidden + fc1 (GELU-erf, 16->32) +
  fc2 (32->32 text hidden).
- ERNIE-4.5-0.3B-style decoder (shared Phase 3 dense contract): vocab 32, hidden
  32, 3 layers, 4 Q / 2 KV heads, head_dim 8, intermediate 64, RMSNorm eps 1e-5,
  RoPE theta 1e4, rotate-half 1D RoPE, QKV bias, NO QK-Norm, SwiGLU, tied
  embeddings, max positions 256, EOS 31.
- Native-resolution packing: each source edge scaled uniformly into [16, 4096]
  pixels then rounded to a positive multiple of patch (2); no spatial-merge
  factor (token reduction happens in the connector).

OpenDoc two-stage recognizer (open_doc_synthetic_config):
- Stage 1: PP-DocLayoutV2 (above). Stage 2: UniRec-0.1B CNN+CTC region
  recognizer reusing the shared PP-OCRv6 tiny recognizer geometry (LCNetV4-style
  backbone + LightSVTR height-collapse + per-frame SVTR MLP + CTC head, 16
  classes, blank 0) and a small multilingual dictionary incl. LaTeX/OTSL tokens.
  NOT a dense VLM: no autoregressive KV decoding on this path. Type-specific
  adapters: text/title/list -> verbatim; formula -> validated LaTeX body;
  table -> validated HTML/OTSL. Malformed formula/table output fails closed.

Region scheduling (RegionScheduler): crops sorted by approximate patch count
descending (stable region_id tie-break); first-fit-decreasing vision/prefill
packing under max_patches_per_vision_batch AND max_regions_per_batch; an
oversized crop is refused (never silently split); restore to (page_index,
reading_order, region_id). region_id is DOCUMENT-GLOBAL (monotonic across pages
in input order). Returned pages are sorted by source page_index (stable) before
composition, so a shuffled input span still yields document order. Detector
outputs cached under an explicit content/geometry FNV-1a key (FIFO-bounded);
a retry hit never re-runs layout. No prefix sharing across unrelated crops.

Output budgets: max_output_tokens_per_region is the single per-region cap,
forwarded to the recognizer (a generative recognizer stops at the cap and marks
truncated). max_output_tokens_per_page is enforced DURING accumulation: once the
page total reaches it, later regions are not recognized and a visible
output_truncated diagnostic is emitted.

EXECUTION (Phase 11): plan_vision_batches is an approximate-cost packing plan.
P7 recognizers record ragged multi-crop vision graphs and split them by exact
packed-patch capacity; Paddle/OpenDoc still process their planned crops one at
a time. Every recognizer uses a fresh KV cache and independent batch-1 text
decode. plan_decode_batches / max_decode_batch remain planning-only; no batched
decode executor exists. Batch-vs-1 tests assert actual output equality.

Layout coordinate contract (M1): pp_doc_layout_decode multiplies each box by the
backbone stride exactly once, producing boxes in DETECTOR-INPUT pixels. The
orchestrator's source transform is therefore source/input (overview source size
over detector input size), NOT source/map -- a source/map scale would
double-count the stride. An absolute-geometry host regression at a known cell
and a full-pipeline crop-content test (bright bars must be hit) pin this.

Memory accounting (S6): the per-region KV cache create receives a MemoryBudget
preflighted against the caller's device limit (weights + workspace + KV). The
run result reports weight_bytes, workspace_bytes, kv_cache_peak_bytes (0 for the
CTC-only OpenDoc path), and peak_device_bytes = weights + workspace + KV peak;
the public bridge reports peak_device_bytes (previously workspace alone).

Synthetic parity tolerances (FP32 device vs double-accumulating CPU oracle,
operation-appropriate, plan §10.1): layout conv chain / class probs / box deltas
3e-3 + 3e-3|ref|; decoded box coordinates 0.75 source px, scores 5e-3; NaViT
patches 5e-4 + 5e-4|ref|, connector features 3e-3 + 3e-3|ref| (GELU-erf);
ERNIE greedy tokens and UniRec CTC ids/text must match EXACTLY; region batching
vs batch-1 must produce equivalent ordered outputs; repeat runs bit-identical.
The host preprocessing oracle mirrors the declared FLOAT stretch-resize +
normalize convention exactly (half-pixel centers, clamp edges) so the conv-chain
parity isolates the graph from preprocessing. These are synthetic gates, not
official quality metrics.

Unmet OFFICIAL gates (PP-DocLayoutV2/V3, PaddleOCR-VL, OpenDoc)

All remain planned. The following official gates are NOT met and no official
artifact has been inspected, pinned, or run:
1. No pinned official checkpoint revision, config fingerprint, or package hash
   for any layout/PaddleOCR-VL/OpenDoc/UniRec model. The synthetic tensor
   names/geometry are declared contracts, not the official checkpoints'.
2. No official layout mAP or reading-order accuracy on any benchmark (e.g.
   PubLayNet/DocLayNet); the synthetic detector is calibrated for deterministic
   fixture detections, not qualified detection quality.
3. No official processor parity: the synthetic stretch-resize/normalize
   statistics (mean/std 0.5) and packing conventions are declared, not verified
   against the official PP-DocLayout/PaddleOCR-VL/OpenDoc processors.
4. No official table/formula/Markdown/JSON qualification: the strict parsers are
   host-tested but not validated against official reference outputs; no table
   cell/structure accuracy, formula delimiter/content accuracy, or Markdown
   structural accuracy is measured on a real benchmark.
5. No official layered goldens: parity is against independent synthetic CPU
   oracles only, not official reference intermediate tensors.
6. No official determinism qualification for a REAL model: the synthetic
   repeat-run/batch-equivalence tests exercise the implemented synthetic graphs
   honestly; they do not establish real-model deterministic decode.
7. No production precision matrix (FP32-only baseline; no FP16/BF16/W4 claim)
   and no measured RTX 4060 8 GB performance/memory numbers (layout latency,
   regions/second, pages/second, resident weights, peak workspace, KV bytes,
   H2D/D2H bytes all unmeasured).
8. PaddleOCR-VL is NOT a validated complex-document default: the plan's Phase 6
   exit gate (PaddleOCR-VL validated for complex documents) is UNMET until
   official artifacts pass the gates above. No real checkpoint is loaded; no
   runtime downloads exist.
9. OpenDoc real artifacts fail closed unconditionally
   (open_doc_validate_real_artifact returns unvalidated); only the synthetic
   two-stage path is exercised. UniRec is never forced into a dense VLM/KV path.

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.

Phase 9 hybrid GDN subsystem + OvisOCR2 + Surya — exact declared architecture

The Phase 9 fixtures are DECLARED synthetic contracts, not official checkpoints.
The hybrid subsystem lives in ocr/ops/ops_gdn.* (device kernels + independent
double-accumulation CPU oracles), ocr/runtime/hybrid_gdn_config.* (host-safe
geometry/tensor contract), ocr/runtime/hybrid_gdn_executor.* (hybrid cache,
executor, host decode controller, HybridVlmGraph), ocr/models/ovis_ocr2.* and
ocr/models/surya.* + ocr/runtime/surya_executor.*. The vision tower reuses the
Phase 7 graph through an ADDITIVE P7VlmExecutor::create_vision_only (identical
vision code path; only the embedded dense text executor is skipped). All prior
suites stay green. Real artifacts fail closed unconditionally.

GDN layer math (the exact synthetic contract, ops/ops_gdn.h):
- Projections (row-major packed [K,N], no bias): q,k,v,z = x @ W{q,k,v,z}
  [T,H*d]; a,b = x @ W{a,b} [T,H].
- Causal depthwise Conv1D + SiLU applied separately to q/k/v (groups ==
  channels C = H*d, kernel K in [2,16], stride 1, dilation 1, left zero pad
  K-1): u_t[c] = bias[c] + sum_j w[c,j] * x_{t-(K-1)+j}[c] (negative absolute
  positions read 0); y = silu(u). RING STATE: per layer/request/group the last
  K-1 PRE-activation values, slot s = absolute_position mod (K-1); a slot is
  overwritten only AFTER the output reading it was computed (race-free).
- REQUIRED L2/QK normalization per head over d: q,k <- q,k / max(||.||_2, l2_eps).
- Gates per (token,head): beta = sigmoid(b); softplus(x) = x>20 ? x :
  log(1+exp(x)); g = -exp(A_log[h]) * softplus(a + dt_bias[h]) (log decay <= 0);
  alpha = exp(g).
- Recurrent state S[h] is [d,d], layout S[k_index, v_index]. Delta rule per
  token: S <- alpha*S; err[v] = v[v] - sum_k S[k,v] k[k]; S[k,v] += beta*k[k]*
  err[v]; o[v] = sum_k S[k,v] q[k]. Straight recurrent prefill loops tokens in
  order; decode is T=1. NO sequence x sequence matrix is ever materialized on
  the recurrent path.
- Chunked prefill (WY/UT-transform chunk algebra, correctness-first SERIAL
  forward substitution inside a chunk; explicitly NOT a parallel scan and NOT
  a sequential rename — a different arithmetic path that must match the
  recurrent path within tolerance): with c_i = sum_{j<=i} g_j,
  L[i,j] = beta_i <k_i,k_j> exp(c_i-c_j) (j<i);
  w_i = beta_i exp(c_i) k_i - sum_{j<i} L[i,j] w_j;
  u_i = beta_i v_i - sum_{j<i} L[i,j] u_j; ue_i = u_i - S_prev^T w_i;
  S_new = exp(c_T) S_prev + sum_i exp(c_T-c_i) k_i ue_i^T;
  o_i = exp(c_i) S_prev^T q_i + sum_{j<=i} <q_i,k_j> exp(c_i-c_j) ue_j.
  All exponent arguments are <= 0 (FP32-safe; underflow to 0 is the limit).
- Gated output normalization per (token,head): y = o/sqrt(mean(o^2)+norm_eps)
  * norm_w * silu(z); executor output path: x += concat_heads(y) @ Wo, then the
  shared RMSNorm + SwiGLU MLP (+residual).

Hybrid cache (plan §7.2; runtime/hybrid_gdn_executor.*):
- Distinct stores, never collapsed: contiguous FP32 K/V for FULL-ATTENTION
  layers only; per-layer/head/request GDN recurrent [H,d,d]; per-layer/request
  Conv ring [3*H*d, K-1] (q/k/v thirds); device-visible DecodeState
  (kv_len/positions/step/state) reused from Phase 3.
- Checked capacity math from ACTUAL dimensions:
  KV = full_layers * requests * tokens * kv_heads * head_dim * 2 * bytes;
  GDN = gdn_layers * requests * H * d * d * bytes;
  Conv = gdn_layers * requests * 3*H*d * (K-1) * bytes; metadata = 16 B (batch 1).
  Preflighted against a MemoryBudget BEFORE allocation; create() synchronously
  zeroes recurrent/conv/metadata so an unbound request never reads stale state.
- Completion/eviction/reset/reuse: record_finished() writes state=finished,
  kv_len=0 (a released slot can never be attended); release() frees the host
  slot immediately; reset() zeroes recurrent+conv+metadata; HybridVlmGraph
  resets the cache before every run (no cross-request leaks, device-tested).
- Paged KV remains OPTIONAL/UNSUPPORTED (PagedKvCache fails closed).

Layer schedule: EXPLICIT per-layer data (HybridLayerKind), never inferred. The
executor mixes GDN layers and full GQA attention layers in ONE decoder (not
KV-only): full layers run QKV(+bias)/optional QK-Norm/1D RoPE/KV append/causal
prefill or single-query decode attention/o+residual; GDN layers run the math
above. Dense embedding (+mm scatter), final RMSNorm, LM head (tied transpose
owned buffer or standalone), greedy/seeded sampling/EOS/stop via the REUSED
Phase 3 DeterministicSampler + Phase 1 IncrementalDetokenizer. Prefill and
decode are separate command-recording loops; bounded output; per-record
heap-owned upload staging retained through completion.

OvisOCR2 (ovis_ocr2_synthetic_config) — DECLARED contract:
- FULL declared decoder: 24 layers, 18 GDN + 6 full attention (full at 0-based
  {3,7,11,15,19,23}), hidden 1024, 8 Q / 2 KV heads, head_dim 128, inter 3072,
  vocab 151936, GDN 8 heads x 128, conv K=4, rotate-half RoPE theta 1e6,
  RMSNorm eps 1e-6. ovis_ocr2_validate_schedule enforces 24/18/6 EXACTLY; the
  executing slice is SCALED and must NOT pass the full gate (tested).
- Executing synthetic slice: vision 2-layer ViT hidden 16 / 2 heads / head_dim
  8 / inter 32 / patch 2, 2D axial RoPE, bidirectional, GELU-erf; spatial-merge
  2 merger + LayerNorm + 2-layer GELU MLP (16*4 -> 32 -> 16); dynamic packing
  (patch 2, grid multiple 2, max 4096 px). Decoder: 6 layers = 4 GDN + 2 full
  (full at {3,5}), hidden 16, 4 Q / 2 KV, head_dim 4, inter 32, vocab 24, EOS
  23, GDN 2 heads x 4, conv K=4, adjacent-pair RoPE theta 1e4, eps 1e-5, tied
  embeddings, QKV bias, no QK-Norm, 1-axis text positions.
- Executing graph: image -> host preprocess -> device ViT+merger -> mm-scatter
  into the prompt -> hybrid prefill/decode -> text -> strict plain/markdown
  composition (spotting/extract fail closed). Generative text carries NO
  confidence. Signature gate = exact tensor inventory + revision + fingerprint
  (outer model_type never identity); real artifacts always unvalidated.
- Memory preflight (ovis_ocr2_preflight_decoder): checked weights (+tied-head
  transpose) + workspace bound + hybrid cache (KV+GDN+Conv+metadata),
  required-vs-available, fails BEFORE allocation.

Surya (surya_synthetic_config) — DECLARED contract:
- FULL declared decoder: same 24-layer / 18-GDN / 6-full schedule (validated
  EXPLICITLY by surya_validate_schedule), hidden 1024, 8 Q / 2 KV, head_dim
  128, inter 3072, vocab 65536, GDN 8x128, conv K=4. The executing slice runs
  fewer channels (scaled) but the schedule contract is validated/tested.
- Detector (EfficientViT-Segformer-style, declared synthetic): 32x32 input,
  stem 3->4 3x3 s2 p1, stage1 4->4 3x3 s2 p1, stage2 4->4 3x3 s1 p1 (declared
  hard_swish backbone; the parity fixture uses identity activations for
  controllability), DB head 4->1 1x1 + sigmoid -> 8x8 probability map. Host
  stretch-resize/normalize (mean/std 0.5, half-pixel bilinear). CPU DB
  postprocess (shared output/db_postprocess): binarize 0.3, box score 0.4,
  unclip 1.4, min 2 px; map->source affine scale = source/map per axis.
- Per line: axis-aligned source crop (min 4x4, clamped) -> dynamic packing ->
  the SAME vision merger + hybrid decoder slice as OvisOCR2 (6 layers 4 GDN +
  2 full) -> greedy text. Reading order: stable top-to-bottom then
  left-to-right; canonical "\n" join. Blank pages yield ZERO lines (tested).
- Confidence: ONLY the DETECTOR mean probability inside the unclipped box
  (DbBox::score, in [0,1], labeled detector-score). NO generative
  pseudo-confidence. Polygons/bounds in ORIGINAL source coordinates (unclip
  may legitimately expand past the page edge). Outputs: plain + spotting +
  markdown via the shared strict composer; extract fails closed; NO fabricated
  table HTML (the synthetic reference defines line polygons + text only).

Declared tolerances (FP32 device vs independent double-accumulation oracles;
per-element |got-ref| <= atol + rtol*|ref|; NaN must match NaN):
- GDN conv+silu: atol=rtol=2e-3; ring state EXACT (0 tolerance).
- GDN recurrent/chunk kernels: atol=rtol=3e-3 (outputs, per-token states,
  every chunk boundary); gates atol=rtol=1e-5 (beta) / 1e-4 (g); gated norm
  1e-4; zero-beta state passthrough 1e-6.
- Full hybrid decoder (layered prefill logits, per-step decode logits, GDN
  recurrent/conv state, full KV, chunk boundaries at sizes 1/5/13):
  atol=rtol=4e-3.
- OvisOCR2 / Surya pipelines (vision merger features, detector probability
  map): atol=rtol=5e-3.
- Deterministic production tokens: EXACT equality (greedy repeat-run, chunked
  == recurrent token sequences, seeded sampling replay); no tolerance.

Phase 9 review fixes (S1/S2) — exact state-safety and text contracts:
- S1 GDN state reset on EVERY from-scratch prefill. HybridGdnExecutor::
  record_state_reset zeroes THIS request's recurrent matrix AND Conv ring for
  every GDN layer through the already-compiled zero-fill shader, recorded right
  after the successful bind_sequence and BEFORE any chunk work, in the SAME
  command list ahead of every layer dispatch (ordering guaranteed by the list;
  no host sync). This is required because the delta rule ACCUMULATES onto the
  state and the Conv reads previous taps: the kernels do not overwrite the whole
  store. record_finished() is therefore documented as METADATA-ONLY (state=
  finished, kv_len=0, position=0) and release() only frees the host slot; stale
  GDN bytes may remain in a released slot and are cleared by the next prefill.
  reset() still clears recurrent + conv + metadata for ALL sequences, and a
  caller that bypasses prefill MUST reset() first. Full-attention KV is never
  zeroed (attention is bounded by kv_len / the causal prefix). HybridVlmGraph's
  per-run reset() is now belt-and-braces, not the safety mechanism.
- S2 Surya tokenizer plumbing. SuryaConfig::tokenizer is a BORROWED const
  Tokenizer* with an explicit lifetime contract (caller-owned; must outlive
  create() AND every recognize(); never copied/freed/retained). create() REFUSES
  a null tokenizer or a vocabulary smaller than the decoder's, and recognize()
  re-checks it, so a text-capable run can never silently emit empty content. It
  is passed through to HybridVlmGraph::run (used only inside generate()). Line
  EOS now comes from the declared decoder contract instead of a hardcoded id. A
  line that produced tokens but no text (and hit neither EOS nor a stop string)
  is an internal_error, not a blank line; a genuinely token-less line and a blank
  PAGE (zero lines) remain valid empty results.

Phase 9 verification (post-review): cuda, xmake AND CMake, debug (vk remains
upstream blocked; no backend edits). New suites (tests/asserts, identical in both
build systems): gdn_conv 3/31, gdn_recurrent 4/26, gdn_chunked 3/25, gdn_cache
12/237 (full hybrid decoder layered parity, controller EOS/stop/seeded sampling
wiring, per-chunk-boundary state comparison, cache capacity/isolation/eviction/
reset/reuse, paged-unsupported, plus the S1 reused-cache leak regression),
model_ovis_ocr2 8/49 (includes ragged non-square image -> vision -> hybrid decode
-> page Markdown end to end), model_surya 10/61 (detector parity, source
coordinates, blank page, repeat-run, plus the S2 detokenized-text and
tokenizer-required regressions). Total NEW: 40 tests / 429 asserts per build
system. Regression: all 26 host suites (2507 asserts / 240 tests; CTest unit_ocr
26/26) and 23 affected prior device suites (4537 asserts / 167 tests) green in
both build systems (Phase 7/8 model suites, layout, region scheduler, ops,
decode, runtime). example_ocr builds in both systems.

S1/S2 regression strength (negative controls, run then reverted; no probe code
remains): with the prefill zeroing disabled, gdn_hybrid_reused_cache_no_leak
FAILS on run B's prefill AND decode logits vs a fresh cache; with the tokenizer
dropped, surya_line_text_detokenized FAILS on per-line text and composed document
content. The S1 regression uses NONZERO GDN weights (not the self-loop zero-weight
fixture, which would mask the leak), runs two generate() calls on ONE cache with
NO reset() between them, and requires BIT-EXACT equality (0 tolerance) of the
second run's prefill logits, decode logits, tokens, and final recurrent + Conv
ring state against fresh-cache references; it also asserts the two requests reach
DIFFERENT states so the equality is non-degenerate.

Phase 9 unmet official gates (all models remain planned):
1. No official OvisOCR2 / Qwen3.5-0.8B / Surya checkpoint pinned, inspected, or
   run; ovis_ocr2_validate_real_artifact and surya_validate_real_artifact fail
   closed unconditionally. The 24/18/6 schedule, head geometry, and every
   tensor name/shape are DECLARED contracts to verify against real artifacts.
2. The GDN equations above are the declared synthetic contract (delta rule +
   softplus decay + gated RMSNorm); official GDN semantics (exact gate
   parameterization, normalization placement, conv grouping) remain unverified.
3. No official processor parity: OvisOCR2 dynamic preprocessing and the Surya
   detector resize/normalize statistics are declared, not verified against
   official processors. The Surya detector is a scaled CNN stand-in for
   EfficientViT-Segformer (no MBConv/attention stages, no official weights).
4. No official layered goldens, deterministic real-model decode, detection
   quality (polygon IoU / line recall), confidence calibration, or OCR accuracy
   (CER/benchmarks) — parity is against independent synthetic CPU oracles only.
5. FP32-only baseline; no FP16/BF16/W4 hybrid path, no paged KV, no batching
     beyond batch 1, no CUDA graphs for the GDN path (Phase 11 added opt-in
     CUDA graph decode only to the dense/P7/Paddle executors), and NO measured
     RTX 4060 performance/memory numbers. The chunked prefill is
     correctness-first serial substitution inside
   chunks — NOT chunk-parallel; no performance claim is made for it.
6. Surya HTML/table output is intentionally NOT emitted (undefined by the
   synthetic reference); only line polygons/blocks/text are defined.

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.

Phase 10 DeepSeek-OCR-2 MoE vertical slice — exact declared architecture

The Phase 10 fixtures are DECLARED synthetic contracts, not official checkpoints.
The MoE subsystem lives in ocr/ops/ops_moe.* (device kernels + independent
double-accumulation CPU oracles), ocr/runtime/moe_vlm_executor.* (the complete
vision + MoE decode graph), and ocr/models/deepseek_ocr2.* (host-safe identity /
tensor / mask / output contract + memory preflight + grounded-Markdown adapter).
Real artifacts fail closed unconditionally (deepseek_ocr2_validate_real_artifact
returns unvalidated). No prior suite changed behaviour.

EXACT expert contract (full config AND the synthetic slice; hidden dims may
scale, the expert count and top-k may NOT):
- 64 routed experts, top-6 routing, TWO shared dense experts. Enforced by
  deepseek_ocr2_validate_expert_contract and asserted in both the host contract
  suite and the device expert suite. The declared full geometry
  (deepseek_ocr2_declared_decoder_config) is hidden 2048, 12 layers, 16 heads,
  head_dim 128, moe/shared intermediate 1408, vocab 129280 — a DECLARED geometry
  used to validate the contract, NOT a verified checkpoint.

Synthetic slice (deepseek_ocr2_synthetic_config), all values declared:
- Vision perception (SAM-style LOCAL): 2 windowed ViT blocks, hidden 16, 2 heads,
  head_dim 8, intermediate 32, patch 2, 3 channels, LayerNorm eps 1e-5, QKV bias,
  no QK-Norm, GELU-erf MLP, 2D axial RoPE (theta 10000, 2 height pairs),
  window_global mask with window_patch_size 2 and global layer {1} (layer 0
  windowed). Perception is a SEPARATE stage from the visual LM encoder and never
  carries the custom visual/query mask.
- BOTH convolutional downsamplers, applied in order: Conv2D 16->16 then 16->16,
  kernel 3, stride 2, pad 1, dilation 1, groups 1, with bias. Token rows are
  transposed to NCHW, convolved, and transposed back. Weights use the row_major
  layout tag with an OIHW shape (the shared WeightStore maps every non-kn/nk
  package layout to the row_major compute tag, exactly like the Phase 5/9 CNNs).
- Visual LM encoder (Qwen2-style): 2 pre-LayerNorm blocks over
  visual_after_down + num_queries rows, hidden 16 (= down2.out_channels),
  2 heads, head_dim 8, intermediate 32, eps 1e-5, QKV + o bias, GELU-erf MLP,
  NO RoPE (the causal-flow structure is carried entirely by the custom mask).
- Causal-flow queries: 4 learned query embeddings appended after the visual rows.
- Connector: LayerNorm + fc1 (bias, GELU-erf) + fc2 (bias) over the QUERY rows
  only, 16 -> 32 -> 16 (text hidden). Emits num_queries == 4 visual tokens.
- Packing: patch 2, grid_multiple 4 (two stride-2 downsamplers), max_pixels
  32*32, max_vision_patches 256.
- MoE text decoder: vocab 24, hidden 16, 2 layers, 4 query heads / 2 KV heads,
  head_dim 4, RMSNorm eps 1e-6, 1D RoPE rotate-half pairs (theta 10000),
  max_position_embeddings 128, EOS id 23, moe_intermediate 16,
  shared_intermediate 16, standalone (untied) lm_head, prefill_chunk_rows 64,
  max_seq_len 128. Every decoder layer is full-attention with contiguous KV.

EXACT custom visual/query block mask (mandatory; never a generic causal mask):
built host-side as DATA with block_size 1 over rows = visual_after_down + 4.
- visual row i attends every visual row and NO query row (bidirectional among
  visual tokens only);
- query row (visual + q) attends ALL visual rows plus query rows with index <= q
  (INCLUDING itself) and no later query.
The device suite asserts this structure element-by-element AND compares the
device encoder attention output per layer against the independent CPU oracle
that consumes the same mask. The text decoder attention is causal (that is the
autoregressive contract); the custom mask is vision-only.

MoE math (the exact synthetic contract, ops/ops_moe.h):
- Router: logits[t,e] = sum_h x[t,h] * W_router[h,e] (FP32, packed [K,N], no
  bias); probs = max-subtracted FP32 softmax over experts.
- Top-k: k selection passes per token, each excluding already-chosen experts;
  strictly-greater comparison so an equal probability selects the LOWEST expert
  index (deterministic tie-break). The k selected probabilities are renormalized
  to sum to 1 (zero-sum guard writes zeros).
- Histogram counts[e] over slots (slot = token*top_k + j); offsets = exclusive
  prefix of counts; permutation walks slots in ASCENDING order per expert, so
  order within an expert is stable. perm_slot / perm_token / slot_pos (inverse).
- Gather: gathered[s,:] = x[perm_token[s],:]. Reference expert loop over ACTIVE
  experts only (counts[e] > 0): gate/up projections, SwiGLU silu(gate)*up, down
  projection on that expert's contiguous gathered rows. Inactive experts are
  skipped entirely. NO grouped persistent GEMM (still absent after Phase 11;
  Phase 11 added CUDA graph decode only to the dense/P7/Paddle paths).
- Restore: out[t,d] = shared0[t,d] + shared1[t,d]
  + sum_j weights[t*top_k+j] * expert_out[slot_pos[t*top_k+j], d], then added to
  the residual stream. Both shared experts are dense over ALL token rows.

Synthetic parity tolerances (FP32 device vs double-accumulating CPU oracle,
per-element |got-ref| <= atol + rtol*|ref|):
- ops_moe router (softmax / top-k weights) and permutation gather/scatter:
  atol = rtol = 2e-5. Expert ids / counts / offsets / perm_token / slot_pos are
  compared EXACTLY (integer equality, 0 tolerance).
- ops_moe full expert layer (routed expert_out, shared0, shared1, final out):
  atol = rtol = 2e-4.
- DeepSeek model slice (vision connector features, per-layer custom-mask encoder
  attention, router logits/probs, selected weights, routed expert_out, both
  shared contributions, moe_out, prefill last-row logits, decode KV K/V state):
  atol = rtol = 5e-3. Selected experts, permutation counts/perm_token/slot_pos,
  and the greedy token sequence are compared EXACTLY.
- Deterministic repeat: two full runs of the same request produce IDENTICAL
  token sequences (exact equality), and decode greedy tokens match the
  independent token-by-token CPU oracle at EVERY step.
- Custom visual/query mask BYTES downloaded from the device are compared with
  EXACT equality (0 tolerance) against an independent structural expectation, and
  the canonical builder is checked against that same expectation.
- Grounded-Markdown coordinates are asserted with a 1e-3 absolute pixel tolerance
  after the [0,1000] -> page-pixel scaling; marker stripping is exact string
  containment, and every malformed/out-of-range case must return a failing Status.

MEASURED memory (synthetic slice, max_seq_len 128, debug, cuda — allocated-byte
measurement, NOT an estimate; preflight and measurement agree exactly):
- vision weights 59,648 B; decoder weights 423,488 B; workspace arena 898,560 B;
  KV cache 16,400 B (K+V for 2 layers x 128 tokens x 2 KV heads x head_dim 4 x
  2 x 4 B = 16,384 B, plus 16 B DecodeState metadata).
- TOTAL 1,398,096 B (~1.33 MiB). The device suite asserts measured == planned for
  each component and that a 1 KiB explicit budget refuses BEFORE allocation with
  required-vs-available details. deepseek_ocr2_workspace_plan is the SINGLE
  shared arena plan used by both reserve_workspace and the preflight, so the
  preflighted workspace equals the measured arena capacity by construction.
- This is a small SYNTHETIC configuration. It is NOT an 8 GB claim: no official
  checkpoint was measured, and no RTX 4060 residency/performance number exists.

Grounded Markdown adapter (host, output contract):
- deepseek_ocr2_parse_grounded accepts
  `<|ref|>text<|/ref|><|det|>[[x0,y0,x1,y1], ...]<|/det|>` with integer
  coordinates in [0,1000] normalized page space, scaled to source pixels by the
  page dimensions. Plain Markdown without grounding is accepted.
- FAILS CLOSED (invalid_argument) on: unclosed <|ref|>, a ref not followed by
  <|det|>, unclosed <|det|>, a non-list payload, a box without 4 comma-separated
  integers, a coordinate > 1000, a degenerate box (x1<=x0 or y1<=y0), an empty
  box list, and trailing data after the list. Invalid grounding is never
  silently dropped or downgraded to plain text.
- deepseek_ocr2_compose supports plain + markdown only (shared strict layout
  composition); spotting/extract fail closed unsupported. Generative content
  carries NO confidence.

Phase 10 independent-review fixes (all must-fix + should-fix, each with a
permanent test; no Phase 11 / backend work). Boundary contracts now enforced:
- Config parsing refuses num_attention_heads == 0 (and a non-dividing hidden_size)
  BEFORE deriving head_dim; an explicit head_dim does not bypass the guard.
- The MoE decoder refuses tie_word_embeddings (standalone lm_head only), and the
  preflight no longer adds nonexistent tied-head transpose bytes.
- A conv downsampler can never ENLARGE the grid: stride 1 requires
  2*pad + 1 <= kernel; stride >= 2 is checked at the smallest legal input, where
  out-in is maximal. record_vision also checks the concrete grid per stage, so
  max_vision_patches + num_queries stays an actual workspace upper bound.
- record_vision requires patch_count == checked grid_h*grid_w (rows derive from
  that product) and refuses zero source width/height, empty pixels, and zero
  resized edges BEFORE the preprocess clamp/scale math.
- create_cache refuses max_seq_len > decoder.max_position_embeddings and 0.
- MemoryLimits are enforced end-to-end (weights + the exact workspace arena + KV
  at the actual max sequence) in create() and create_cache(), before allocation.
- The device routing readback is validated by the SHARED
  validate_moe_routing_tables (counts sum == slots, exclusive-prefix offsets,
  offset+count <= slots) before it drives any subview; it returns a Status. The
  host MoE oracles validate x extents / perm_token indices (gather), expert_out
  extent / destination slots (scatter), and reject tokens == 0 before dividing.
- The MoE input validators run on the actual execution path.
- Upload staging is retained on EVERY exit path: caller-owned lists get the
  completion callback (a post-error commit is safe), executor-owned lists are
  committed AND drained on error (releasing staging and satisfying the runtime's
  committed-before-destroy rule).
- The router softmax seeds its running max from a REAL element instead of a
  finite -3.0e38 sentinel, so a FINITE extreme row (e.g. all -3.2e38) can no
  longer underflow every exp() into a zero sum and yield NaN; an explicit
  zero-sum guard matches the oracle. Non-finite rows still write zeros.
- deepseek_ocr2_compose GROUNDS every region: markers are stripped into canonical
  content, the union box plus per-box polygon corners populate the source
  geometry, and malformed/out-of-range grounding fails closed. Grounding
  coordinates are the CLOSED [0,1000] (1000 == the full page edge) and trailing
  commas are rejected.
- Test independence: the decoder oracle consumes the INDEPENDENT vision-oracle
  features (device features separately asserted equal); the device custom-mask
  BYTES are downloaded and compared to an independent structural expectation
  while executor and oracle share ONE canonical builder
  (MoeVlmExecutor::build_visual_query_mask); the empty-assignment case runs the
  real device router/top-k/histogram/offsets/permutation chain and drives the
  expert loop from the validated DEVICE counts; the composed-route test compares
  EXACT device-vs-host expert ids from the SAME x/W via the host projection.

Phase 10 verification (post-review): cuda, xmake AND CMake, debug (vk remains
upstream blocked; no backend edits). New suites (tests/asserts, identical in both
build systems): deepseek_contract 11/105 (host), moe_router 7/755,
moe_permutation 4/59, moe_experts 2/12, model_deepseek_ocr2 12/193 —
36 tests / 1124 asserts total. All 27 host suites (2612 asserts / 251 tests;
CTest unit_ocr 27/27) and the FULL OCR device set of 36 suites
(6652 asserts / 265 tests) are green in BOTH systems, so the 32 prior device
suites (5633 asserts / 240 tests) show no regression; example_ocr builds in both
systems.

Two REAL defects were found by the new suites and fixed (no probe code remains):
1. GEMV epilogue fall-through: the small-M GEMV path implements only
   none/bias/residual. The connector runs at M == num_queries == 4 <= kGemvSmallM,
   so its `bias_gelu_erf` epilogue silently fell through to the no-epilogue GEMV,
   dropping BOTH the bias and the GELU activation. record_linear now restricts
   GEMV to the epilogues it implements; gated epilogues always take GEMM. The
   vision-feature parity assert failed before the fix and passes after.
2. Invalid CUDA block size: the single-thread offsets kernel used
   set_block_size(1,1,1), which CUDA rejects (must be a multiple of 32, <= 1024).
   It now uses 32 with the ocr_linear_id() == 0u guard.

Phase 10 unmet official gates (DeepSeek-OCR-2 remains planned):
1. No official DeepSeek-OCR-2 checkpoint pinned, inspected, or run;
   deepseek_ocr2_validate_real_artifact fails closed unconditionally. Every
   dimension, tensor name/shape/layout, and the declared full geometry are
   contracts to verify against real artifacts. The exact tensor/config signature
   gate exists (deepseek_ocr2_validate_signature) but has only been exercised
   with the synthetic inventory.
2. The official visual architecture is UNVERIFIED: the SAM-style local perception
   block conventions, the TWO downsamplers' exact channel/kernel/stride schedule,
   the visual LM encoder depth/normalization/RoPE convention, the causal-flow
   query count and initialization, and the connector reduction are all DECLARED
   synthetic values. In particular the encoder uses NO RoPE and the query count
   is 4; the official query count/positions are unknown.
3. The custom visual/query mask SEMANTICS are the declared reference (visual
   bidirectional; query sees all visual + previous queries including itself).
   Whether the official reference includes the query's own position, and any
   per-layer mask variation, remain unverified against official code.
4. The official MoE routing semantics are UNVERIFIED: FP32 router without bias,
   plain softmax then top-6 renormalization, lowest-index tie-break, no expert
   bias/correction term, no group-limited routing, no auxiliary-loss-free bias,
   and the exact shared-expert scaling are declared synthetic choices. The
   64/top-6/two-shared COUNTS are the requirement contract, not a verified
   checkpoint fact.
5. No official processor parity (resize/normalize statistics, patch packing,
   prompt template, tokenizer) and no official layered goldens: parity is against
   independent synthetic CPU oracles only. No official deterministic real-model
   decode, no OCR accuracy (CER/benchmarks), no grounding-coordinate accuracy
   against real documents, and no blank/hallucination qualification for a REAL
   model.
6. FP32-only baseline. No FP16/BF16/W4 MoE path (no quantization A/B arm was
     added because no quantized MoE path exists), no paged KV, no batching beyond
     batch 1, no CUDA graphs for the MoE path (Phase 11 added opt-in CUDA graph
     decode only to the dense/P7/Paddle executors), no grouped persistent GEMM,
     no MTP/DFlash, and NO
     measured RTX 4060 performance or 8 GB residency claim. Prefill is single-chunk
   (prompt must fit prefill_chunk_rows); the MoE active-expert loop synchronizes
   the stream per layer to read routing counts, which is correctness-first and
   makes NO performance claim.

Until these gates are met with official artifacts, Session::create and
Session::recognize fail closed with unvalidated for every real model, and the
registry marks every recipe planned.
