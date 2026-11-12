#pragma once
// OCR engine -- PP-OCRv6 size variant identity (Phase 6 extraction).
//
// DEVICE-SAFE header: the PpOcrSize enum was originally declared in
// model_recipe.h, but that header also declares the host-only AttentionMaskKind
// enum, which clashes with the ops enum in device translation units. The CNN
// graph contract (models/pp_ocrv6.h) needs ONLY this size enum, so it lives here
// where both host and device TUs can include it without pulling model_recipe.h.
// model_recipe.h re-exports it (via this include) so existing callers are
// unchanged.
#include <cstdint>

#include <luisa/core/stl/string.h>

namespace luisa::ocr {

// Explicit PP-OCRv6 size variants: one family, three checkpoints. `none` for
// every non-PP-OCRv6 model.
enum class PpOcrSize : uint32_t {
    none = 0,
    tiny,
    small,
    medium,
};

[[nodiscard]] luisa::string_view pp_ocr_size_name(PpOcrSize size) noexcept;

}// namespace luisa::ocr