#pragma once
// OCR engine -- Qwen3-VL host preprocessing bridge (Phase 4). Host-only.
//
// Isolates the Phase 1 preprocess reference (preprocess.h -> api.h ->
// model_recipe.h, whose host-only AttentionMaskKind must never meet the ops
// enum in a device TU) behind a small host-only surface. ONLY host-only TUs
// may include this header. The Qwen3-VL executor .cpp forward-declares the
// function over the shared plain data types (runtime/qwen3_vl_image.h) and
// never includes this header.
#include <cstdint>

#include <luisa/core/stl/memory.h>

#include "ocr/runtime/qwen3_vl_image.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Preprocess one image on the host through the Phase 1 reference (the exact
// declared bilinear convention). Fails closed on invalid input.
[[nodiscard]] Result<Qwen3VlPreprocessedImage> qwen3_vl_preprocess_image_host(
    const Qwen3VlImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev);

}// namespace luisa::ocr
