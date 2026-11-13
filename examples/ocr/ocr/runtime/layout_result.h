#pragma once
// OCR engine -- layout document -> public RecognizeResult bridge (Phase 6).
// HOST-ONLY header: includes api.h (host AttentionMaskKind). Never include this
// in a device TU that also includes ops_attention.h.
//
// Maps a plain LayoutDocument + LayoutRunResult to the frozen public
// RecognizeResult schema. Lines/blocks carry source page/box/polygon/type/
// content/reading order. Confidence is carried through ONLY for an explicitly
// validated detector/CTC score source (labeled internal/test via a diagnostic);
// generative content keeps has_confidence=false (never a pseudo-probability).
#include "ocr/api.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Assemble a public RecognizeResult from a composed layout document + run.
// `metadata` carries the resolved identity/precision/support fields (the caller
// fills them from the recipe/session). The bridge fills text/lines/blocks/
// counters/diagnostics. When any block carries a confidence score, an
// internal/test confidence-source diagnostic is appended so the score is never
// mistaken for a calibrated generative probability.
[[nodiscard]] RecognizeResult layout_document_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                        const ModelMetadata &metadata);

// Additive Phase 7: shared STRICT layout composition for the custom
// dense/layout variants (TeleOCR / GLM-OCR / MinerU / HunyuanOCR). Composes a
// plain LayoutRunResult into a LayoutDocument for the requested mode:
//   plain    : region texts joined in reading order (pages ascending).
//   spotting : same as plain, with source coordinates on lines/blocks.
//   markdown : per-type Markdown blocks (title -> heading, table -> validated
//              HTML/OTSL, formula -> validated LaTeX, text/list -> paragraphs).
//   extract  : the concatenated region content must be valid JSON conforming
//              to `extract_schema` (required); canonical text is serialized JSON.
// Malformed table/formula/JSON output is an ERROR, never a plain-text fallback.
// Empty pages produce a valid empty document. Generative content carries NO
// confidence. This is the same strict contract as the Phase 6 Paddle compose,
// shared so the Phase 7 models do not duplicate (or silently diverge from) it.
[[nodiscard]] Result<LayoutDocument> layout_compose_strict(const LayoutRunResult &run,
                                                           LayoutOutputMode mode,
                                                           luisa::string_view extract_schema);

}// namespace luisa::ocr