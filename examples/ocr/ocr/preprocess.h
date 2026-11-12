#pragma once
// OCR engine -- production preprocessing (Phase 1). Host-only reference path.
//
// Checked orientation/resize/normalize/packed-ragged image+patch transforms
// with 64-bit checked counts and source-coordinate transform provenance. The
// GPU production path (later phases) must match this reference bit-for-bit for
// the declared resize convention. This is NOT the official-processor
// differential path (that is feature_bundle + tools/preprocess.py).
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/api.h"
#include "ocr/image.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Resize convention. Phase 1 implements exact bilinear (half-pixel centers)
// and area (box) resampling; the recipe declares which one it validated.
enum class ResizeConvention : uint32_t {
    bilinear = 0,
    area = 1,
};

[[nodiscard]] luisa::string_view resize_convention_name(ResizeConvention convention) noexcept;

struct PreprocessSpec {
    ResizeConvention resize{ResizeConvention::bilinear};
    uint32_t target_width{0};  // 0 = keep source width
    uint32_t target_height{0}; // 0 = keep source height
    // Aspect-preserving longest-edge bound (0 = disabled). Applied before the
    // explicit target size; the result is rounded to a multiple of pad_to.
    uint32_t max_long_edge{0};
    uint32_t pad_to_multiple{1};
    PixelOrder order{PixelOrder::rgb};
    // Per-channel normalization: out = (pixel/255 - mean) / std. Empty = none.
    luisa::vector<float> mean;
    luisa::vector<float> std_dev;
    // Patch grid (0 = no patch metadata).
    uint32_t patch_size{0};
    uint64_t max_pixels{64ull * 1024ull * 1024ull};
};

// Source-coordinate transform provenance: maps a point in the preprocessed
// image back to the ORIGINAL page pixel space (scale + offset per axis).
struct CoordinateTransform {
    float scale_x{1.0f};
    float scale_y{1.0f};
    float offset_x{0.0f};
    float offset_y{0.0f};

    [[nodiscard]] SourcePoint to_source(float x, float y) const noexcept {
        return SourcePoint{x * scale_x + offset_x, y * scale_y + offset_y};
    }
};

// One preprocessed image: normalized CHW float pixels + patch metadata.
struct PreprocessedImage {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t channels{3};
    luisa::vector<float> pixels;// CHW, normalized
    uint32_t patch_size{0};
    uint32_t patches_w{0};
    uint32_t patches_h{0};
    uint64_t patch_count{0};
    CoordinateTransform transform;
};

// Preprocess one decoded image (checked). Rejects degenerate sizes, invalid
// normalization, and pixel-count overflow.
[[nodiscard]] Result<PreprocessedImage> preprocess_image(const DecodedImage &image, const PreprocessSpec &spec);

// Packed ragged batch: images concatenated along the patch axis with offsets
// (no maximum-page padding). All counts are checked 64-bit.
struct RaggedImageBatch {
    luisa::vector<PreprocessedImage> images;
    luisa::vector<uint64_t> patch_offsets;// size == images.size()+1
    uint64_t total_patches{0};
    uint64_t total_pixels{0};
};

[[nodiscard]] Result<RaggedImageBatch> pack_ragged_batch(
    luisa::span<const DecodedImage> images, const PreprocessSpec &spec,
    uint64_t max_total_patches);

}// namespace luisa::ocr