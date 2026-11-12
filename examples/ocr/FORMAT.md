OCR formats (Phase 1 implemented)

This document describes the on-disk and result formats implemented in Phase 1.
The binary package (.lcocr v1), the safetensors inspection reader, the feature
bundle (.ocrfeat v1), and the result JSON serializer are implemented and tested
host-side. NO real model is validated; every recipe remains `planned`
(see MODELS.md). The formats below are exercised only by small SYNTHETIC
fixtures under src/tests/data/ocr.

.lcocr package v1 (implemented: ocr/package.h/.cpp)

Little-endian binary package. Layout (all offsets/sizes are u64):

  [0..64)   fixed header:
            magic "LCOCR1\0\0" (8 bytes), u32 format_version=1,
            u32 header_size=64, u64 manifest_offset, u64 manifest_size,
            u64 table_offset, u64 table_size, u64 payload_base, u64 file_size.
  payloads  tensor blobs, each starting at a 256-byte aligned absolute offset;
            padding bytes are zero; no overlap; the payload region is fully
            covered (no trailing gap).
  manifest  UTF-8 JSON (identity/dependencies/versions; see below).
  table     u64 tensor_count, then per-tensor records + name bytes:
            u32 name_size, u32 rank, u32 encoding, u32 layout, u32 quant_kind,
            u32 group_size, u32 reserved(0), u64 element_count,
            u64 shape[rank], u64 strides[rank] (element strides, row-major),
            u64 payload_offset, u64 payload_size,
            u64 scales_offset, u64 scales_size,
            u64 zero_points_offset, u64 zero_points_size,
            name bytes (UTF-8, no NUL).

The header is written last by the streaming writer; the reader validates the
whole structure eagerly (header/manifest/table) but reads payload bytes ON
DEMAND, so opening a multi-GB package never copies the file into host memory.

Region order is header, payloads, manifest, table. The table is the LAST
region: table_offset + table_size must equal file_size (trailing bytes after
the table are rejected by both the C++ reader and validate_package.py). All
padding between regions must be zero.

EMPTY packages (zero tensors) are supported with a canonical layout shared by
the C++ and Python writers/readers: the payload region is empty zero padding
from the header end up to the 256-byte-aligned manifest offset, so
payload_base == manifest_offset (aligned), followed by the manifest and a
count-0 table. Both writers produce this layout; both readers and the Python
validator accept it and reject any nonzero padding or misalignment.

ZERO-ELEMENT tensors (element_count == 0, e.g. shape [0,4]) use one canonical
rule in C++/Python/validator: a zero-size payload/scales/zero-point range may
have offset 0 (no blob) OR any 256-byte aligned in-bounds position within the
payload region [payload_base, manifest_offset]. Both writers emit the aligned
current position; zero-size ranges never participate in overlap/coverage.
Misaligned or out-of-region nonzero offsets are rejected.

Encodings (table `encoding`):
  0 fp32, 1 fp16, 2 bf16 (dense; 4/2/2 bytes per element),
  3 int4_packed (grouped; quant_kind selects symmetric/asymmetric),
  4 int8 (grouped symmetric).

Quantization (ocr/quantization.h/.cpp):
- INT4 codes are unsigned nibbles 0..15, two per byte, LOW nibble first;
  odd counts pad the final high nibble with 0 (a nonzero pad is rejected).
- Groups are contiguous along flattened row-major order; the final group may
  be partial. group_count = ceil(N / group_size). Group size is 64 or 128.
- Asymmetric: dequant = (code - zero_point[g]) * scale[g]; zero_point in 0..15
  stored as one byte per group.
- Symmetric: zero point is implicitly 8 (code 8 == 0.0); no zero-point bytes.
- Scales are FP32, finite and non-negative (validated). INT8 codes are signed
  -127..127 (never -128), dequant = code * scale[g].
- FP16/BF16 conversions are explicit round-to-nearest-even; FP16 is NEVER
  reinterpreted as BF16, and INT8 is NEVER reinterpreted as grouped INT4.

Manifest JSON (yyjson; unknown/duplicate keys rejected):
  format_version, model (semantic target, e.g. "glm_ocr", "pp_ocrv6:medium"),
  recipe_version, checkpoint_revision, config_fingerprint, license, producer,
  producer_version, preprocessing_version, tokenizer_version,
  output_parser_version, component ("" = whole model),
  dependencies[] {component, package, sha256},
  source_hashes[] {path, algorithm, digest}.

Fail-closed validation: malformed/truncated/overflow/duplicate/overlap/trailing
data, misaligned payloads, nonzero padding, nonfinite scales, out-of-range zero
points, unsupported encoding/rank/group size all return Status/Result errors
(never LUISA_ERROR on ordinary input). Rank is arbitrary within kPackageMaxRank
(8). Tensor lookup is by NAME.

The registry matches the exact config/tensor signature; an outer HF model_type
alone never selects a recipe. Unknown revisions load only through the explicit
unsafe experimental override and remain unvalidated; production default is
fail-closed. Component-selective loading/unload and device weight upload arrive
in later phases; the Phase 1 reader is host-only.

Safetensors inspection reader (implemented: ocr/safetensors_reader.h/.cpp)

Checked host-side reader for packaging tools and C++ tests, NOT a production
inference path. Parses the u64 header length + JSON metadata + tensor table;
validates dtypes/ranks/shapes/offset ranges, duplicate names, file bounds,
overlap/gaps/trailing data. Supported inspection dtypes: F32, F16, BF16, I8, U8,
I16, I32, I64, U16, U32, U64, BOOL; others fail closed with a repack instruction.
Data bytes are read on demand; F32/F16/BF16 decode to FP32 with explicit
conversion (no reinterpretation).

.ocrfeat feature bundle v1 (implemented: ocr/feature_bundle.h/.cpp)

Versioned checked interchange for official-AutoProcessor differential tests
(produced by tools/preprocess.py). A DEBUG path, not required for ordinary
input. Same header/payload/table discipline as .lcocr:

  [0..64)  header: magic "OCRF" (u32), u32 version=1, u32 header_size=64,
           u32 reserved(0), u64 meta_offset, u64 meta_size,
           u64 tensor_table_offset, u64 tensor_table_size,
           u64 payload_base, u64 file_size.
  meta     UTF-8 JSON: producer, processor, processor_revision,
           source_image_hash, decode_settings (deterministic decode JSON text).
  table    u64 count, then per tensor: u32 name_size, u32 dtype, u32 rank,
           u32 reserved(0), u64 element_count, u64 shape[rank],
           u64 payload_offset, u64 payload_size, name bytes.
  payloads 256-byte aligned, zero padding, no overlap/trailing data.

Feature dtypes: f32, i32, i64, u8.

Result JSON (implemented: ocr/output/json_output.h/.cpp)

The CLI writes the FULL RecognizeResult schema (ocr/api.h) on success and a
strict status/error form on failure. The serializer uses yyjson with correct
ownership (every dynamic string/key copied into the document; writer buffer
freed with the matching allocator). Non-finite floats are REJECTED (never
emitted as invalid JSON). The success form includes:

  ok, text,
  lines[] {text, source{page_index, box{x0,y0,x1,y1}, polygon[{x,y}]},
           reading_order, has_confidence, confidence?},
  blocks[] {type, content, source, reading_order, has_confidence, confidence?},
  metadata {model_id, pp_size, target_name, checkpoint_revision, package_hash,
            recipe_version, backend, weight/activation/cache_precision,
            cache/offload/graph/fast_path_policy, max_overview_pixels,
            max_crop_pixels, max_output_tokens_per_region/page,
            enabled_optimizations[], support_state, validated,
            router{used, router_id, initial_model, final_model, reason}},
  diagnostics[] {severity, code, message},
  counters {load_ms, preprocess_ms, vision_ms, decode_ms, postprocess_ms,
            total_ms, pages_processed, regions_processed, patches_processed,
            output_tokens, peak_device_bytes, peak_host_bytes,
            resident_weight_bytes}.

The failure form is {ok:false, code, message, detail}.

Rules frozen in Phase 0 and preserved:

- Source coordinates are in original page pixel space; transforms
  (overview resize, layout crop, perspective warp, tile composition,
  cross-page merging) must preserve them. Phase 1 preprocess records a
  source-coordinate transform (scale/offset per axis) as provenance.
- has_confidence is explicit. Generative VLMs report false — never a
  pseudo-probability.
- Metadata precision fields are RESOLVED CONCRETE values (never "automatic");
  the policy fields record what was actually applied. Router/fallback
  provenance is explicit (router.used=false means no routing/fallback); there
  is no implicit fallback.
- Truncation, output repair, offload, and fallback are always visible as
  diagnostics; a failed Markdown/extract response is never silently converted
  into apparent successful plain text.
- Empty/no-text pages return a valid empty result from conventional OCR
  (tested separately against VLM hallucination in later phases).

Tokenizer / prompt / detokenizer (implemented)

- tokenizer.{h,cpp}: yyjson HF tokenizer.json parsing of explicitly validated
  subsets only: byte-level BPE (GPT-2 byte encoder/decoder, ranks, added/special
  tokens, GPT-2 pre-tokenization) and SentencePiece-style Unigram with byte
  fallback + metaspace. Unsupported normalizer/pre-tokenizer/post-processor/
  model semantics fail closed (no silent approximation, no general Jinja).
  Invalid UTF-8 input is REJECTED (no lossy replacement). Added tokens match
  longest-first; special tokens are suppressed on decode by default.
- Unigram metaspace convention (validated against official tokenizers 0.22.2
  / sentencepiece 0.2.2): with add_prefix_space=true (the default), encode
  prepends ONE structural U+2581 to the FIRST chunk of the input. Decode
  strips exactly ONE leading structural U+2581 and maps every remaining
  U+2581 to a space: decode([▁hello])='hello', [▁hello,▁world]='hello world',
  [▁,▁hello]=' hello' (only one stripped). decode(encode(t)) == t.
  BATCH/INCREMENTAL EQUIVALENCE is the gate: whole-sequence decode strips
  once; the incremental detokenizer strips exactly once at STREAM level
  (first decoded content), NEVER per token. add_prefix_space=false is honored
  (no prefix); a replacement other than U+2581 is REJECTED; add_bos_token/
  add_eos_token are honored when present (the <s>/</s> presence heuristic
  applies only when the flags are absent). <0xHH> byte-fallback pieces are
  excluded from ordinary Viterbi literal matching, and multi-byte byte
  fallback chains keep forward byte order.
- NFC boundary (validated subset): full NFC composition is NOT implemented.
  With a BPE NFC normalizer, any input containing a combining mark OR any
  NFC non-invariant character is REJECTED at encode time (fail closed). The
  rejection set is the COMPLETE single-character NFC non-invariant set
  (Unicode 15.1, 73 ranges): all singleton mappings (U+2126 OHM, U+212A
  KELVIN, U+212B ANGSTROM, U+037E, U+0374, U+0387, CJK compatibility
  ideographs, ...) plus every canonical decomposition. Official HF maps
  these before encoding (expected IDs pinned in the frozen official
  differential fixture); our subset never passes their raw bytes through.
  Precomposed NFC-invariant text passes unchanged. A Unigram NFC normalizer
  is rejected at parse time.
- prompt.{h,cpp}: recipe-ID-specific builders; unknown template IDs fail closed.
- detokenizer.{h,cpp}: incremental UTF-8 across token boundaries, EOS and stop
  strings without splitting multibyte output or leaking stop text. Output is
  byte-identical to whole-sequence decode of the same token stream.

These are exercised by frozen SYNTHETIC tokenizer fixtures AND one frozen
OFFICIAL tokenizer differential (official_tokenizer_diff.json, generated once
from local HF tokenizers 0.22.2 + sentencepiece 0.2.2; tests only READ it,
never regenerate). This is a tokenizer-tool differential on synthetic
vocabs — NOT an official model checkpoint output; model/processor golden
gates remain unmet.

Image / document / preprocess (implemented)

- image.{h,cpp}: checked JPEG/PNG decode via the repository stb_image dependency
  with byte/pixel/dimension limits, degenerate rejection, RGB/BGR conversion,
  explicit row stride, and a checked JPEG EXIF orientation parser + transform
  (stb alone does not orient). Memory transfers use luisa::span.
- document.{h,cpp}: externally-rasterized page images + a checked JSON manifest
  with page/pixel limits. NO in-process PDF parser (tools/rasterize_pdf.py
  rasterizes outside the inference process).
- preprocess.{h,cpp}: checked orientation/resize (bilinear/area)/normalize/
  packed-ragged image+patch metadata with 64-bit checked counts and
  source-coordinate transform provenance. spec.order is honored: the source
  is converted to the requested RGB/BGR order BEFORE resampling, so output
  channel c (and mean[c]/std_dev[c]) corresponds to the requested order's
  channel c. This is the production reference path; the official-processor
  differential path is feature_bundle + tools/preprocess.py.
- tools/pack_model.py --quantize accepts none|fp16|bf16|int4-symmetric|
  int4-asymmetric|int8. fp16/bf16 are dense storage conversions (RNE; Python
  BF16 NaN stays a quiet NaN, never carries into infinity) and produce
  packages the C++ reader decodes bit-identically to the Python validator.

Test data layout (Phase 1)

src/tests/data/ocr/ holds frozen small SYNTHETIC images, tokenizer cases,
package/safetensors fixtures, and golden_manifest.json (which labels every
artifact synthetic vs official; the official list is empty). The path is
compiled into the host tests as LUISA_OCR_TEST_DATA_DIR (both build systems).
Goldens are NEVER auto-created or overwritten by tests; the only writer is
tools/generate_fixtures.py (synthetic) and tools/export_golden.py (official,
explicit local artifacts, offline). Temporary corruption copies in tests live
in the system temp dir and are cleaned up.