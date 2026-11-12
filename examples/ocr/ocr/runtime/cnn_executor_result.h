#pragma once
// OCR engine -- CNN executor public-API bridge (Phase 5). HOST-ONLY.
//
// Maps the plain CnnRunResult (runtime/cnn_executor.h, device-safe) to the
// frozen public RecognizeResult schema (ocr/api.h). This header includes
// api.h and must ONLY be included by host TUs (never a device TU that also
// includes the ops enums, whose host AttentionMaskKind would clash).
#include "ocr/api.h"
#include "ocr/runtime/cnn_executor.h"

namespace luisa::ocr {

// Assemble a public RecognizeResult from a plain CNN run. `metadata` carries
// the resolved identity/precision/support fields (the caller fills them from
// the recipe/session); the bridge fills lines/blocks/text/counters. Lines are
// emitted in reading order across pages (pages ascending, then reading_order).
// Confidence is carried through from CTC unchanged (never synthesized).
[[nodiscard]] RecognizeResult cnn_result_to_public(const CnnRunResult &run, const ModelMetadata &metadata);

}// namespace luisa::ocr