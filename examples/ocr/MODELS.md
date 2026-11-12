OCR model support matrix

Every model is planned. None works yet. (Phase 1 added host
artifact/tokenizer/image/preprocessing/JSON infrastructure; Phase 2 added a
portable FP32 tensor runtime and operators; Phase 3 added a dense
autoregressive decoder core validated ONLY against a synthetic CPU oracle;
Phase 4 added a Qwen3-VL vertical slice + FireRed-OCR recipe contract and
strict structured outputs, again validated ONLY against a synthetic CPU
oracle; Phase 5 added a conventional CNN OCR executor + PP-OCRv6 graph
contracts + CTC/DB/reading-order output infrastructure, again validated ONLY
against a synthetic CPU oracle. No model left the planned state.)

No checkpoint has been pinned, inspected, loaded, or run. There are no
validated revisions, fingerprints, hashes, precisions, backends, or
performance numbers anywhere in this repository's OCR engine. The graph
contracts below are transcribed from the requirement as contracts to
verify against official artifacts — they are not verified facts.

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
| `paddle_ocr_vl_1_6` | layout VLM | PP-DocLayoutV3 + reading order/crops; native-resolution NaViT; adaptive MLP connector; ERNIE-4.5-0.3B decoder; full pipeline required | plain, spotting, markdown, extract | planned |
| `ovis_ocr2` | hybrid GDN | dynamic preprocessing; vision merger; Qwen3.5-0.8B hybrid decoder (GDN + periodic full GQA) | plain, markdown | planned |
| `tele_ocr` | dense VLM | Qwen2.5-VL-derived encoder (window/global), MLP aligner, Qwen3-0.6B (QK-Norm, M-RoPE); outer `model_type` is NOT identity | plain, markdown | planned |
| `glm_ocr` | layout VLM | PP-DocLayoutV3; 0.4B CogViT; downsampling connector; GLM-0.5B decoder; MTP deferred | plain, markdown, extract | planned |
| `mineru_2_5_pro_2605_1_2b` | layout VLM | page layout; native crops; 675M NaViT; 2x2 pixel-unshuffle; Qwen2-0.5B; cross-page composition | plain, markdown, extract | planned |
| `hunyuan_ocr_1_5` | custom VLM | native Hunyuan-ViT; adaptive pool/MLP; Hunyuan-0.5B; XD-RoPE (text/height/width/time); DFlash deferred | plain, spotting, markdown, extract | planned |
| `surya_2_5` | hybrid GDN | EfficientViT-Segformer detector + crops; vision merger; 24-layer hybrid decoder (18 GDN + 6 full-attention, contract to verify) | plain, spotting, markdown | planned |
| `open_doc_0_1b` | layout VLM | PP-DocLayoutV2 + UniRec-0.1B two-stage text/formula/table recognizer (not a generic VLM); no autoregressive KV cache; UniRec-specific output contract; retains layout routing | plain, markdown | planned |
| `fire_red_ocr` | dense VLM | Qwen3-VL-2B vision/merger + Qwen3 decoder; interleaved MRoPE, DeepStack, GQA, QK-Norm; no GRPO operators | plain, markdown, extract | planned |
| `qwen3_vl_2b_instruct` | dense VLM | same reusable Qwen3-VL graph; Instruct recipe (not Thinking) | plain, markdown, extract | planned |
| `deepseek_ocr_2` | MoE VLM | SAM-style local perception (windowed, separate block); downsamplers; visual LM encoder + causal-flow queries carrying the custom visual/query mask; DeepSeek MoE decoder (64 routed experts/top-6/2 shared, contract to verify) | plain, markdown | planned |
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

Qualification gate (later phases)

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