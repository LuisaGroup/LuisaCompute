#pragma once
// OCR engine -- PaddleOCR-VL host composition + public bridge (Phase 6).
// HOST-ONLY header: includes api.h (host AttentionMaskKind). Never include this
// in a device TU that also includes ops_attention.h.
//
// Composes a plain LayoutRunResult into a LayoutDocument (canonical Markdown /
// plain / JSON per mode) and maps it to the frozen public RecognizeResult.
// Region content is parsed/validated per type with STRICT structured-output
// rules: a malformed table/formula/JSON is an ERROR, never a plain-text
// fallback. Generative content carries NO confidence (has_confidence=false).
#include <luisa/core/stl/string.h>

#include "ocr/api.h"
#include "ocr/models/paddle_ocr_vl.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Compose a plain layout run into a LayoutDocument for the requested mode.
//   plain    : region texts joined in reading order (per page, pages ascending).
//   spotting : same as plain, but lines/blocks carry source coordinates.
//   markdown : per-type Markdown blocks (title -> heading, table -> validated
//              HTML/OTSL, formula -> validated LaTeX, text/list -> paragraphs).
//   extract  : the concatenated region content must be valid JSON conforming to
//              `extract_schema` (required); the canonical text is the serialized
//              JSON. A non-conforming document fails closed.
// Malformed region output fails closed (invalid_argument); no plain-text
// fallback. Empty pages produce a valid empty document.
[[nodiscard]] Result<LayoutDocument> paddle_ocr_vl_compose(const LayoutRunResult &run,
                                                           LayoutOutputMode mode,
                                                           luisa::string_view extract_schema);

// Map a composed LayoutDocument + run metadata to the public RecognizeResult.
// `metadata` carries the resolved identity/precision/support fields (the caller
// fills them from the recipe/session); the bridge fills lines/blocks/text/
// counters/diagnostics. Confidence is carried through ONLY when the region
// score is an explicitly validated detector/CTC source (labeled internal/test);
// generative content keeps has_confidence=false.
[[nodiscard]] RecognizeResult paddle_ocr_vl_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                      const ModelMetadata &metadata);

}// namespace luisa::ocr