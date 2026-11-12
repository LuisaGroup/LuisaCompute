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

}// namespace luisa::ocr