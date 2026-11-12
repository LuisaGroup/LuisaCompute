#pragma once
// OCR engine -- Qwen3-VL image data types (Phase 4). Host-only, device-safe.
//
// Plain data structs shared by the Qwen3-VL executor and the host
// preprocessing bridge. This header pulls ONLY the host contract
// (models/qwen3_vl.h) -- never preprocess.h/api.h (host AttentionMaskKind)
// and never the ops enums -- so both host and device TUs may include it.
#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen3_vl.h"

namespace luisa::ocr {

// One image input to the vision graph: already-decoded source pixels (HWC
// uint8 RGB) plus the packing metadata computed host-side.
struct Qwen3VlImageInput {
    luisa::vector<uint8_t> pixels;// HWC uint8, source size
    uint32_t source_width{0};
    uint32_t source_height{0};
    Qwen3VlPackedImage packed;
};

// Host-side preprocessing result for one image (normalized CHW pixels).
struct Qwen3VlPreprocessedImage {
    uint32_t width{0};
    uint32_t height{0};
    luisa::vector<float> pixels;// CHW normalized
};

}// namespace luisa::ocr
