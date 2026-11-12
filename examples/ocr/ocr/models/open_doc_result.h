#pragma once
// OCR engine -- OpenDoc host adapters + composition + public bridge (Phase 6).
// HOST-ONLY header: includes api.h (host AttentionMaskKind). Never include this
// in a device TU that also includes ops_attention.h.
//
// OpenDoc is a TWO-STAGE pipeline (PP-DocLayoutV2 + UniRec CNN+CTC), NOT a dense
// VLM. The UniRec stage produces raw CTC text per region; the type-specific
// adapters here turn that into canonical content (text / LaTeX formula /
// HTML-or-OTSL table). Malformed formula/table output is an ERROR, never a
// plain-text fallback. CTC confidence is an explicitly validated score source,
// labeled internal/test by the public bridge.
#include <luisa/core/stl/string.h>

#include "ocr/api.h"
#include "ocr/models/open_doc.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Adapt a raw UniRec CTC decode for a region into type-specific canonical
// content. STRICT:
//   * text/title/list/header/footer/figure/other: decoded text verbatim (trimmed);
//     an empty decode yields empty content (a blank region, not an error).
//   * formula: the decoded text must be valid LaTeX (wrapped in a recognized
//     delimiter pair); canonical content is the body WITHOUT outer delimiters.
//     A malformed formula is an ERROR (invalid_argument), never plain text.
//   * table: the decoded text must parse as HTML or OTSL; canonical content is
//     the validated table source. A malformed table is an ERROR, never plain text.
[[nodiscard]] Result<luisa::string> open_doc_adapt_region(LayoutBlockType block_type,
                                                          luisa::string_view decoded_text);

// Compose a plain layout run into a LayoutDocument for the requested mode
// (plain / spotting / markdown). OpenDoc does NOT declare extract mode; passing
// extract fails closed (unsupported). Malformed region output fails closed.
[[nodiscard]] Result<LayoutDocument> open_doc_compose(const LayoutRunResult &run, LayoutOutputMode mode);

// Map a composed LayoutDocument + run metadata to the public RecognizeResult.
[[nodiscard]] RecognizeResult open_doc_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                 const ModelMetadata &metadata);

}// namespace luisa::ocr